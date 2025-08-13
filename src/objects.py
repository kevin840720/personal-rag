# -*- encoding: utf-8 -*-
"""
@File: objects.py
@Time: 2025/01/13 11:05:32
@Author: Kevin Wang
@Desc: 自定義物件
"""

from abc import (ABC,
                 abstractmethod,
                 )
from datetime import datetime
from enum import Enum
from typing import (Any,
                    Dict,
                    List,
                    Optional,
                    Union,
                    )
from uuid import UUID

from docling_core.transforms.chunker import DocChunk
from docling_core.types.doc.document import DoclingDocument
from pydantic import (ConfigDict,
                      Field,
                      field_validator,
                      )
from pydantic.dataclasses import dataclass
from pydantic import BaseModel

class FileType(Enum):
    STRING="string"
    PDF="pdf"
    DOCX="docx"
    XLSX="xlsx"
    MARKDOWN="markdown"

class PDFSourceType(str, Enum):  # TODO: 考慮要不要與 FileType 合併
    UNKNOWN = "unknown"
    PDF = "PDF"
    MS_WORD = "Microsoft Word"
    MS_EXCEL = "Microsoft Excel"
    LO_WRITER = "LibreOffice Writer"
    LO_CALC = "LibreOffice Calc"
    MARKDOWN = "Markdown"

class DocumentMetadata(BaseModel):
    file_type:FileType
    file_name:str
    title:Optional[str]=None
    author:Optional[str]=None
    subject:Optional[str]=None  # 或是 description
    created_at:Optional[datetime]=None
    modified_at:Optional[datetime]=None
    is_chunk:bool=False  # 是否已經被 chunked
    chunk_info:Optional[dict]=None  # 如果是 chunked，則包含 chunk 資訊
    extra:Optional[dict]=None  # 其他附加資訊

    def to_dict(self) -> Dict[str, Union[str, int, float, List, Dict]]:
        """輸出字典格式"""
        result = {'file_type': self.file_type.value,
                  'file_name': self.file_name,
                  }
        if self.title is not None: result['title'] = self.title
        if self.author is not None: result['author'] = self.author
        if self.subject is not None: result['subject'] = self.subject
        if self.created_at is not None: result['created_at'] = self.created_at.isoformat() if self.created_at else None
        if self.modified_at is not None: result['modified_at'] = self.modified_at.isoformat() if self.modified_at else None
        if self.is_chunk: result['is_chunk'] = self.is_chunk
        if self.chunk_info is not None:
            for key, value in self.chunk_info.items():
                result[f"chunk_info.{key}"] = value
        if self.extra is not None:
            for key, value in self.extra.items():
                result[f"extra.{key}"] = value
        return result

    @classmethod
    def from_dict(cls, data:Dict[str, Union[str, int, float, List, Dict]]) -> "DocumentMetadata":
        """從字典建立 DocumentMetadata 物件。"""
        def parse_dt(value):
            return datetime.fromisoformat(value) if isinstance(value, str) else value
        
        for key in data.copy():
            if key.startswith("chunk_info."):
                # 將 chunk_info 的鍵值對移到 chunk_info 字典中
                if "chunk_info" not in data:
                    data["chunk_info"] = {}
                data["chunk_info"][key[len("chunk_info."):]] = data[key]
            if key.startswith("extra."):
                # 將 extra 的鍵值對移到 extra 字典中
                if "extra" not in data:
                    data["extra"] = {}
                data["extra"][key[len("extra."):]] = data[key]

        return cls(file_type=FileType(data["file_type"]) if "file_type" in data else FileType.STRING,
                   file_name=data["file_name"] if "file_name" in data else "",
                   title=data.get("title"),
                   author=data.get("author"),
                   subject=data.get("subject"),
                   created_at=parse_dt(data.get("created_at")),
                   modified_at=parse_dt(data.get("modified_at")),
                   is_chunk=data.get("is_chunk", False),
                   chunk_info=data.get("chunk_info"),
                   extra=data.get("extra"),
                   )

class PDFMetadata(DocumentMetadata):
    source:Optional[PDFSourceType] = None
    producer:Optional[str] = None  # 實際輸出/轉換成 PDF 的軟體或函式庫

    def to_dict(self) -> Dict[str, Union[str,int,float,List,Dict]]:
        result = super().to_dict()
        if self.source is not None: result['source'] = self.source.value
        if self.producer is not None: result['producer'] = self.producer
        return result

    @classmethod
    def from_dict(cls,
                  data:Dict[str, Union[str,int,float,List,Dict]],
                  ) -> "PDFMetadata":
        base = DocumentMetadata.from_dict(data)

        # NOTE: 注意大小寫會影響 Enum 的解析
        try:
            source = PDFSourceType(data["source"]) if data.get("source") else None
        except ValueError:
            source = None
        return cls(file_type=base.file_type,
                   file_name=base.file_name,
                   title=base.title,
                   author=base.author,
                   subject=base.subject,
                   created_at=base.created_at,
                   modified_at=base.modified_at,
                   source=source,
                   producer=data.get("producer"),
                   )

class Document(DoclingDocument):
    id:UUID
    content:str
    metadata:DocumentMetadata
    embedding:Optional[List[float]]=None  # TODO: remove this, already move to Chunk

    __pydantic_config__ = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> Dict[str, Union[str,int,float,List,Dict]]:
        """回傳字典格式"""
        result = {'id': self.id,
                  'metadata': self.metadata.to_dict(),
                  'content': self.content,
                  **self.export_to_dict()  # 確保 DoclingDocument 的屬性也被包含
                  }
        return result

    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        """確保內容不為空"""
        if not v or not v.strip():
            raise ValueError('文件內容不能為空')
        return v.strip()

    @field_validator('embedding')
    @classmethod
    def validate_embedding(cls, v):
        """驗證 embedding 向量"""
        if v is not None:
            if not v:
                raise ValueError('embedding 不能為空列表')
            if not all(isinstance(x, (int, float)) for x in v):
                raise ValueError('embedding 必須是數值列表')
        return v

class Chunk(BaseModel):
    id:UUID
    content:str
    metadata:DocumentMetadata
    embedding:Optional[List[float]]=None
    _doc_chunk:Optional[DocChunk]=None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> Dict[str, Union[str,int,float,List,Dict]]:
        """回傳字典格式"""
        result = {'id': self.id,
                  'metadata': self.metadata.to_dict(),
                  'content': self.content,
                  }
        return result

    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        """確保內容不為空"""
        if not v or not v.strip():
            raise ValueError('文件內容不能為空')
        return v.strip()

    @field_validator('embedding')
    @classmethod
    def validate_embedding(cls, v):
        """驗證 embedding 向量"""
        if v is not None:
            if not v:
                raise ValueError('embedding 不能為空列表')
            if not all(isinstance(x, (int, float)) for x in v):
                raise ValueError('embedding 必須是數值列表')
        return v
