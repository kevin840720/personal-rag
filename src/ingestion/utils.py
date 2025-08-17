# -*- encoding: utf-8 -*-
"""
@File    :  utils.py
@Time    :  2025/06/18 13:21:56
@Author  :  Kevin Wang
@Desc    :  None
"""

import sys
from typing import (Any,
                    Iterator,
                    Literal,
                    Optional,
                    )

import pandas as pd
from docling_core.transforms.serializer.base import (BaseDocSerializer,
                                                     BaseTableSerializer,
                                                     SerializationResult,
                                                     )
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.types.doc.document import TableItem
from docling_core.transforms.serializer.html import HTMLTableSerializer
from docling_core.transforms.serializer.markdown import (ContentLayer,
                                                         MarkdownDocSerializer,
                                                         MarkdownParams,
                                                         )
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import (DoclingDocument,
                                             DOCUMENT_TOKENS_EXPORT_LABELS,
                                             DEFAULT_CONTENT_LAYERS,
                                             )
from docling_core.types.doc.labels import DocItemLabel


class JSONTableSerializer(BaseTableSerializer):
    """JSON-specific table item serializer for Docling HybridChunker

    可以參考 docling_core.transforms.serializer 實做
    TODO: 考慮要不要將所有 doc process 的
    """
    def _split_df(self,
                  df:pd.DataFrame,
                  by:Literal["column", "index"],
                  batch_size:int,
                  ) -> Iterator[pd.DataFrame]:
        """
        將 DataFrame 按欄（column）或列（index）分批切分。

        Args:
            df (pd.DataFrame): 要切分的原始資料表。
            by (Literal["column", "index"]): 指定切分方向：
                - "column"：按欄切（每次取 batch_size 欄位）
                - "index"：按列切（每次取 batch_size 筆資料）
            batch_size (int): 每一批切幾個欄或列。

        Raises:
            Iterator[pd.DataFrame]: 回傳一個逐批產生子資料表的 generator。

        Yields:
            ValueError: 當 by 參數不是 "column" 或 "index" 時拋出。
        """
        if by == "column":
            for i in range(0, len(df.columns), batch_size):
                yield df.iloc[:, i:i+batch_size]
        elif by == "index":
            for i in range(0, len(df), batch_size):
                yield df.iloc[i:i+batch_size, :]
        else:
            raise ValueError(f"Unsupported slice mode: {by}")

    def _col_name_to_first_row(self,
                               df:pd.DataFrame,
                               ) -> pd.DataFrame:
        """將 column name 改寫成首列 (first row) 的值。
           如果是 MultiIndex，則會將所有層級的名稱合併成一個 tuple，再放入首列。

        假設原始 df 長這樣：        套用函數後會變成：
           name1   name2                      col1    col2
        0      1      11       column_name   name1   name2
        1      2      21    ->           0       1      11
        2     10     101                 1       2      21
        3     20     201                 2      10     101
                                         3      20     201
        """
        df.columns = df.columns.to_list()  # 確保 columns 是單層結構
        new_df = df.T.reset_index(drop=False, names=["column_name"]).T
        new_df.columns = [f"col{i:03d}" for i in range(len(df.columns))]  # 將 columns 轉為 col_0, col_1, ...
        return new_df

    def _row_name_to_first_col(self,
                               df:pd.DataFrame,
                               ) -> pd.DataFrame:
        """將 index name 改寫成首行 (first col) 的值。
           如果是 MultiIndex，則會將所有層級的名稱合併成一個 tuple，再放入首行。

        假設原始 df 長這樣：             套用函數後會變成：
                  0      1               row_name     0      1
        name0     1     11          row_0   name0     1     11
        name1     2     21    ->    row_1   name1     2     21
        name2    10    101          row_2   name2    10    101
        name3    20    201          row_3   name3    20    201
        """
        df.index = df.index.to_list()
        new_df = df.reset_index(drop=False, names=["row_name"])
        new_df.index = [f"row{i:03d}" for i in range(len(df))]  # 將 index 轉為 row000, row001, ...
        return new_df

    def serialize(self,
                  *,
                  item: TableItem,
                  doc_serializer: BaseDocSerializer,
                  doc: DoclingDocument,
                  **kwargs: Any,
                  ) -> SerializationResult:
        """
        預設按 columns 分批輸出 JSON，遇到重複欄位自動加上 col_num，
        如果整個 columns 失敗，fallback 按 records 且每筆記錄加上 row_num。
        """
        df = item.export_to_dataframe()

        # 由於原生 dataframe 中，index/column name 可能是「純數字」，這會使 LLM 在閱讀時產生模糊空間，因此用 col_{j} 與 row_{i} 來代替
        # 原先的 index/column name 則會一併被視作 value，放在第一行與第一列。
        # （這同時也能避免因為 index/column name 重複而導致 `to_json` 的序列化失敗問題）
        df = self._row_name_to_first_col(df)
        df = self._col_name_to_first_row(df)

        batch = kwargs.get("batch_size", 30)
        orient = kwargs.get("orient", "columns")

        parts = []
        if orient == "columns":
            for i, sub in enumerate(self._split_df(df.copy(), by="column", batch_size=batch), 1):
                txt = sub.to_json(orient="columns", force_ascii=False, indent=2)
                parts.append(f"Table Part {i:03d}\n{txt}")
        else:
            for i, sub in enumerate(self._split_df(df.copy(), by="index", batch_size=batch), 1):
                txt = sub.to_json(orient="records", force_ascii=False, indent=2)
                parts.append(f"Table Part {i:03d}\n{txt}")
        result = "\n\n".join(parts)
        return create_ser_result(text=result, span_source=item)

class UserDefinedDocSerializer(MarkdownDocSerializer):
    table_serializer:BaseTableSerializer = JSONTableSerializer() 

def export_to_user_text(doc:DoclingDocument,
                        from_element:int=0,
                        to_element:int=sys.maxsize,
                        labels:Optional[set[DocItemLabel]] = None,
                        escape_underscores:bool=True,
                        image_placeholder:str="<!-- image -->",
                        enable_chart_tables:bool=True,
                        image_mode:ImageRefMode=ImageRefMode.PLACEHOLDER,
                        indent:int=4,
                        text_width:int=-1,
                        page_no:Optional[int]=None,
                        included_content_layers:Optional[set[ContentLayer]]=None,
                        page_break_placeholder:Optional[str]=None,  # e.g. "<!-- page break -->",
                        include_annotations:bool=True,
                        mark_annotations:bool=False,
                        ) -> str:

        my_labels = labels if labels is not None else DOCUMENT_TOKENS_EXPORT_LABELS
        my_layers = (
            included_content_layers
            if included_content_layers is not None
            else DEFAULT_CONTENT_LAYERS
        )
        serializer = UserDefinedDocSerializer(
            doc=doc,
            params=MarkdownParams(
                labels=my_labels,
                layers=my_layers,
                pages={page_no} if page_no is not None else None,
                start_idx=from_element,
                stop_idx=to_element,
                escape_underscores=escape_underscores,
                image_placeholder=image_placeholder,
                enable_chart_tables=enable_chart_tables,
                image_mode=image_mode,
                indent=indent,
                wrap_width=text_width if text_width > 0 else None,
                page_break_placeholder=page_break_placeholder,
                include_annotations=include_annotations,
                mark_annotations=mark_annotations,
            ),
        )
        ser_res = serializer.serialize()
        return ser_res.text