# -*- encoding: utf-8 -*-
"""
@File    :  test_manager.py
@Time    :  2025/08/23 16:36:24
@Author  :  Kevin Wang
@Desc    :  PostgresSessionManager 單元測試（同步版）
"""

import os
import uuid
from datetime import datetime, timedelta

import pytest
from dotenv import load_dotenv
from sqlalchemy import text

from conftest import SKIP_POSTGRES_TESTS
from infra.session.manager import PostgresSessionManager
from infra.session.base import ConversationSession
from infra.llm_providers.base import ChatMessage

load_dotenv()

@pytest.fixture(scope="function")
def session_manager():
    # 每個測試都建立一個新的 schema
    schema_name = f"test_schema_{uuid.uuid4().hex[:8]}"
    mgr = PostgresSessionManager(host=os.getenv("MY_POSTGRE_HOST", "localhost"),
                                 port=int(os.getenv("MY_POSTGRE_PORT", "5432")),
                                 dbname=os.getenv("MY_POSTGRE_DB_NAME", "postgres"),
                                 schema=schema_name,
                                 user=os.getenv("MY_POSTGRE_USERNAME", "postgres"),
                                 password=os.getenv("MY_POSTGRE_PASSWORD", "postgres"),
                                 session_ttl_hours=24,
                                 )
    mgr._ensure_tables()
    yield mgr
    # 清理 schema
    with mgr.engine.begin() as conn:
        conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE'))

@pytest.mark.skipif(SKIP_POSTGRES_TESTS, reason="Skipping PGVector tests")
class TestPostgresSessionManager:
    def test_create(self,
                    session_manager:PostgresSessionManager,
                    ):
        session_id = session_manager.create_session()
        session = session_manager.get_session(session_id)
        assert session is not None
        assert isinstance(session, ConversationSession)
        assert str(session.session_id) == str(session_id)
        assert session.messages == []
        assert isinstance(session.created_at, datetime)

    def test_create_with_custom_id(self,
                                   session_manager:PostgresSessionManager,
                                   ):
        custom_id = str(uuid.uuid4())
        returned_id = session_manager.create_session(session_id=custom_id)
        assert str(returned_id) == str(custom_id)
        session = session_manager.get_session(custom_id)
        assert session is not None

    def test_add(self,
                 session_manager:PostgresSessionManager,
                 ):
        session_id = session_manager.create_session()
        msg = ChatMessage(role="user", content="Hello", extra={})
        session_manager.add_message(session_id, msg)
        session = session_manager.get_session(session_id)
        assert len(session.messages) == 1
        assert session.messages[0].role == "user"
        assert session.messages[0].content == "Hello"

    def test_expired_session(self,
                             session_manager:PostgresSessionManager,
                             ):
        session_id = session_manager.create_session()
        session_manager.session_ttl = timedelta(seconds=0)  # 人為調短 ttl
        # 人為回寫 updated_at 為過期
        with session_manager.sessionmaker() as s:
            s.execute(text(f"UPDATE {session_manager.ChatSessionModel.__table__.schema}.chat_sessions "
                           f"SET updated_at = :updated WHERE session_id = :session_id"
                           ),
                      {"updated": datetime.now() - timedelta(days=2),
                       "session_id": session_id,
                       })
            s.commit()
        session = session_manager.get_session(session_id)
        assert session is None

    def test_delete(self,
                    session_manager:PostgresSessionManager,
                    ):
        session_id = session_manager.create_session()
        ok = session_manager.delete_session(session_id)
        assert ok is True
        session = session_manager.get_session(session_id)
        assert session is None

    def test_delete_nonexist(self,
                             session_manager:PostgresSessionManager,
                             ):
        assert session_manager.delete_session("does-not-exist") is False

    def test_multi_session_isolate(self,
                                   session_manager:PostgresSessionManager,
                                   ):
        id1 = session_manager.create_session()
        id2 = session_manager.create_session()
        session_manager.add_message(id1, ChatMessage(role="user", content="A", extra={}))
        session_manager.add_message(id2, ChatMessage(role="user", content="B", extra={}))
        s1 = session_manager.get_session(id1)
        s2 = session_manager.get_session(id2)
        assert s1.messages[0].content == "A"
        assert s2.messages[0].content == "B"
