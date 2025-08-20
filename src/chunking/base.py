# -*- encoding: utf-8 -*-
"""
@File    :  base.py
@Time    :  2025/01/18 13:44:19
@Author  :  Kevin Wang
@Desc    :  None
"""

from abc import (ABC,
                 abstractmethod,
                 )
from typing import (Any,
                    List,
                    )

from objects import Chunk


class ChunkProcessor(ABC):
    """文件分塊處理器介面"""
    @abstractmethod
    def process(self, doc:Any, **kwargs) -> List[Chunk]:
        pass
