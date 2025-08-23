# -*- encoding: utf-8 -*-
"""
@File    :  manager.py
@Time    :  2025/08/23 15:48:46
@Author  :  Kevin Wang
@Desc    :  TODO: 考慮改成 Async 版本，需要再 postgresql 加裝 asyncpg 插件
"""
import uuid
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import create_engine, Column, String, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, mapped_column, sessionmaker, Session

from sqlalchemy import (create_engine,
                        text,
                        )

from infra.session.base import AbstractSessionManager, ConversationSession
from infra.llm_providers.base import ChatMessage

def make_chat_session_class(base, schema:str):
    class ChatSessionModel(base):
        __table_args__ = {"schema": schema}
        __tablename__ = "chat_sessions"

        session_id = mapped_column(String(256), primary_key=True)
        messages = mapped_column(JSONB, default=lambda: [])
        context = mapped_column(JSONB, default=lambda: {})
        created_at = mapped_column(DateTime, default=datetime.now)
        updated_at = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)
    return ChatSessionModel

class PostgresSessionManager(AbstractSessionManager):
    def __init__(self,
                 host:str="localhost",
                 port:int=5432,
                 dbname:str="postgres",
                 schema:str="default_session_manager",
                 user:str="postgres",
                 password:str="",
                 session_ttl_hours:int=24,
                 echo:bool=False,
                 ):
        self._Base = declarative_base()
        dsn = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}"
        self.ChatSessionModel = make_chat_session_class(self._Base, schema)

        self.engine = create_engine(dsn, echo=echo)
        self.sessionmaker = sessionmaker(bind=self.engine)
        self.session_ttl = timedelta(hours=session_ttl_hours)

        # 建立 schema
        with self.engine.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))

    def _ensure_tables(self):
        with self.engine.begin() as conn:
            self._Base.metadata.create_all(conn)

    def create_session(self,
                       session_id:Optional[str]=None,
                       ) -> str:
        self._ensure_tables()
        if session_id is None:
            session_id = str(uuid.uuid4())

        with self.sessionmaker() as session:
            result = session.query(self.ChatSessionModel).filter_by(session_id=session_id).first()
            if result:
                result.updated_at = datetime.now()
            else:
                chat_session = self.ChatSessionModel(
                    session_id=session_id,
                    messages=[],
                    context={},
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
                session.add(chat_session)
            session.commit()
            return session_id

    def get_session(self,
                    session_id:str,
                    ) -> Optional[ConversationSession]:
        self._ensure_tables()
        cutoff_time = datetime.now() - self.session_ttl
        with self.sessionmaker() as session:
            chat_session = (
                session.query(self.ChatSessionModel)
                .filter_by(session_id=session_id)
                .filter(self.ChatSessionModel.updated_at > cutoff_time)
                .first()
            )
            if chat_session:
                messages = [ChatMessage(**msg) for msg in chat_session.messages]
                return ConversationSession(
                    session_id=chat_session.session_id,
                    messages=messages,
                    context=chat_session.context,
                    created_at=chat_session.created_at,
                    updated_at=chat_session.updated_at,
                )
        return None

    def add_message(self,
                    session_id:str,
                    message:ChatMessage,
                    ) -> None:
        self._ensure_tables()
        with self.sessionmaker() as session:
            chat_session = session.query(self.ChatSessionModel).filter_by(session_id=session_id).first()
            if chat_session:
                messages = chat_session.messages or []
                messages.append(message.model_dump())
                chat_session.messages = messages
                chat_session.updated_at = datetime.now()
                session.commit()

    def delete_session(self,
                       session_id:str,
                       ) -> bool:
        self._ensure_tables()
        with self.sessionmaker() as session:
            chat_session = session.query(self.ChatSessionModel).filter_by(session_id=session_id).first()
            if chat_session:
                session.delete(chat_session)
                session.commit()
                return True
            return False

    def close(self):
        self.engine.dispose()
