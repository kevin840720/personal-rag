# -*- encoding: utf-8 -*-
"""
@File    :  no_chunk.py
@Time    :  2025/08/30
@Author  :  Kevin Wang
@Desc    :  A passthrough chunker that wraps loader outputs directly into Chunk objects.
"""

from warnings import warn
from typing import Any, List, Optional
import uuid

from chunking.base import ChunkProcessor
from objects import Chunk, DocumentMetadata


class NoChunkProcessor(ChunkProcessor):
    """A no-op processor that creates a single Chunk from given content/metadata.

    Usage patterns supported:
      - process(doc=None, metadata=..., content=...)  # explicit content
      - process(doc="...text...", metadata=...)      # treat doc as content if content not provided
    """

    def process(self,
                doc:Any,
                metadata:DocumentMetadata,
                content:Optional[str]=None,
                ) -> List[Chunk]:
        text = content or (doc if isinstance(doc, str) else "")
        if not text:
            # Ensure we always produce a valid content for Chunk; raise clear error otherwise
            warn("NoChunkProcessor requires non-empty text via `content` or string `doc`.")

        # Mark metadata as chunked and add basic chunk_info
        meta = metadata.model_copy(deep=True)
        meta.is_chunk = True
        meta.chunk_info = {'chunk_index': 0,
                           'total_chunks': 1,
                           'chunk_size': len(text),
                           }

        # Deterministic UUID based on content + metadata for idempotency
        chunk_id = uuid.uuid5(uuid.NAMESPACE_DNS,
                              f"{text}|{meta.model_dump_json()}"
                              )

        return [Chunk(id=chunk_id,
                      content=text,
                      metadata=meta,
                      )]

