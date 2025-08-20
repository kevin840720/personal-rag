# -*- encoding: utf-8 -*-
"""
@File    :  test_base.py
@Time    :  2025/01/15 14:26:27
@Author  :  Kevin Wang
@Desc    :  None
"""

import uuid

import pytest
from objects import DocumentMetadata, FileType, Chunk
from docling_core.types.doc import DocumentOrigin, DoclingDocument, GroupItem, ContentLayer, SectionHeaderItem, TextItem
from chunking.docling import DoclingChunkProcessor
from docling_core.types.doc.labels import (
    CodeLanguageLabel,
    DocItemLabel,
    GraphCellLabel,
    GraphLinkLabel,
    GroupLabel,
    PictureClassificationLabel,
)

import pytest
from docling_core.types.doc import (
    DoclingDocument, GroupItem, ContentLayer, SectionHeaderItem, TextItem, TitleItem,
    ListGroup, ListItem, RefItem
)
import uuid

@pytest.fixture
def sample_doc():
    return DoclingDocument(
        schema_name="DoclingDocument",
        version="1.5.0",
        name="判斷日文自他動詞",
        id=uuid.UUID("8fb497b6-0ae4-528b-af7c-233c9f05f593"),
        origin=DocumentOrigin(
            mimetype="text/markdown",
            binary_hash=16290215813565461083,
            filename="判斷日文自他動詞.md",
            uri=None,
        ),
        furniture=GroupItem(
            self_ref="#/furniture",
            parent=None,
            children=[],
            content_layer=ContentLayer.FURNITURE,
            name="_root_",
            label=GroupLabel.UNSPECIFIED,
        ),
        body=GroupItem(
            self_ref="#/body",
            parent=None,
            children=[RefItem(cref=f"#/texts/{i}") for i in range(11)] + [RefItem(cref="#/groups/0")],
            content_layer=ContentLayer.BODY,
            name="_root_",
            label=GroupLabel.UNSPECIFIED,
        ),
        groups=[
            ListGroup(
                self_ref="#/groups/0",
                parent=RefItem(cref="#/body"),
                children=[RefItem(cref="#/texts/11"), RefItem(cref="#/texts/12"), RefItem(cref="#/texts/13")],
                content_layer=ContentLayer.BODY,
                name="list",
                label=GroupLabel.LIST,
            )
        ],
        texts=[
            TitleItem(
                self_ref="#/texts/0",
                parent=RefItem(cref="#/body"),
                children=[],
                content_layer=ContentLayer.BODY,
                label=DocItemLabel.TITLE,
                prov=[],
                orig="判斷日文自他動詞",
                text="判斷日文自他動詞",
                formatting=None,
                hyperlink=None,
            ),
            SectionHeaderItem(
                self_ref="#/texts/1",
                parent=RefItem(cref="#/body"),
                children=[],
                content_layer=ContentLayer.BODY,
                label=DocItemLabel.SECTION_HEADER,
                prov=[],
                orig="動詞有以下4種類：",
                text="動詞有以下4種類：",
                formatting=None,
                hyperlink=None,
                level=1,
            ),
            SectionHeaderItem(
                self_ref="#/texts/2",
                parent=RefItem(cref="#/body"),
                children=[],
                content_layer=ContentLayer.BODY,
                label=DocItemLabel.SECTION_HEADER,
                prov=[],
                orig="（1）純自動詞",
                text="（1）純自動詞",
                formatting=None,
                hyperlink=None,
                level=2,
            ),
            TextItem(
                self_ref="#/texts/3",
                parent=RefItem(cref="#/body"),
                children=[],
                content_layer=ContentLayer.BODY,
                label=DocItemLabel.PARAGRAPH,
                prov=[],
                orig="只有自動詞用法。動作或變化僅限於主體，不及物。 例：花が咲く、友達と会う、デパートができる、道を歩く。",
                text="只有自動詞用法。動作或變化僅限於主體，不及物。 例：花が咲く、友達と会う、デパートができる、道を歩く。",
                formatting=None,
                hyperlink=None,
            ),
            SectionHeaderItem(
                self_ref="#/texts/4",
                parent=RefItem(cref="#/body"),
                children=[],
                content_layer=ContentLayer.BODY,
                label=DocItemLabel.SECTION_HEADER,
                prov=[],
                orig="（2）純他動詞",
                text="（2）純他動詞",
                formatting=None,
                hyperlink=None,
                level=2,
            ),
            TextItem(
                self_ref="#/texts/5",
                parent=RefItem(cref="#/body"),
                children=[],
                content_layer=ContentLayer.BODY,
                label=DocItemLabel.PARAGRAPH,
                prov=[],
                orig="只有他動詞用法。動作必定及物，需要賓語。 例：本を読む、ご飯を食べる。",
                text="只有他動詞用法。動作必定及物，需要賓語。 例：本を読む、ご飯を食べる。",
                formatting=None,
                hyperlink=None,
            ),
            SectionHeaderItem(
                self_ref="#/texts/6",
                parent=RefItem(cref="#/body"),
                children=[],
                content_layer=ContentLayer.BODY,
                label=DocItemLabel.SECTION_HEADER,
                prov=[],
                orig="（3）自他對立動詞",
                text="（3）自他對立動詞",
                formatting=None,
                hyperlink=None,
                level=2,
            ),
            TextItem(
                self_ref="#/texts/7",
                parent=RefItem(cref="#/body"),
                children=[],
                content_layer=ContentLayer.BODY,
                label=DocItemLabel.PARAGRAPH,
                prov=[],
                orig="一組成對存在，自動詞表自然狀態或變化，他動詞表人為施加。必須成對記憶。 例：止まる／止める、流れる／流す。",
                text="一組成對存在，自動詞表自然狀態或變化，他動詞表人為施加。必須成對記憶。 例：止まる／止める、流れる／流す。",
                formatting=None,
                hyperlink=None,
            ),
            SectionHeaderItem(
                self_ref="#/texts/8",
                parent=RefItem(cref="#/body"),
                children=[],
                content_layer=ContentLayer.BODY,
                label=DocItemLabel.SECTION_HEADER,
                prov=[],
                orig="（4）自他同形動詞",
                text="（4）自他同形動詞",
                formatting=None,
                hyperlink=None,
                level=2,
            ),
            TextItem(
                self_ref="#/texts/9",
                parent=RefItem(cref="#/body"),
                children=[],
                content_layer=ContentLayer.BODY,
                label=DocItemLabel.PARAGRAPH,
                prov=[],
                orig="同一動詞，依語境可為自動詞或他動詞。數量很少，遇到直接記。 例：吹く（風が吹く／笛を吹く）、閉じる（戸が閉じる／目を閉じる）。",
                text="同一動詞，依語境可為自動詞或他動詞。數量很少，遇到直接記。 例：吹く（風が吹く／笛を吹く）、閉じる（戸が閉じる／目を閉じる）。",
                formatting=None,
                hyperlink=None,
            ),
            SectionHeaderItem(
                self_ref="#/texts/10",
                parent=RefItem(cref="#/body"),
                children=[],
                content_layer=ContentLayer.BODY,
                label=DocItemLabel.SECTION_HEADER,
                prov=[],
                orig="自他對立動詞辨識：看結尾～す/～れる/～あ段る",
                text="自他對立動詞辨識：看結尾～す/～れる/～あ段る",
                formatting=None,
                hyperlink=None,
                level=1,
            ),
            ListItem(
                self_ref="#/texts/11",
                parent=RefItem(cref="#/groups/0"),
                children=[],
                content_layer=ContentLayer.BODY,
                label=DocItemLabel.LIST_ITEM,
                prov=[],
                orig="[<RawText children='① 結尾～す → 都是他動詞'>]",
                text="[<RawText children='① 結尾～す → 都是他動詞'>]",
                formatting=None,
                hyperlink=None,
                enumerated=False,
                marker="",
            ),
            ListItem(
                self_ref="#/texts/12",
                parent=RefItem(cref="#/groups/0"),
                children=[],
                content_layer=ContentLayer.BODY,
                label=DocItemLabel.LIST_ITEM,
                prov=[],
                orig="[<RawText children='② 結尾～れる → 都是自動詞'>]",
                text="[<RawText children='② 結尾～れる → 都是自動詞'>]",
                formatting=None,
                hyperlink=None,
                enumerated=False,
                marker="",
            ),
            ListItem(
                self_ref="#/texts/13",
                parent=RefItem(cref="#/groups/0"),
                children=[],
                content_layer=ContentLayer.BODY,
                label=DocItemLabel.LIST_ITEM,
                prov=[],
                orig="[<RawText children='③ 結尾～あ段る → 都是自動詞'>]",
                text="[<RawText children='③ 結尾～あ段る → 都是自動詞'>]",
                formatting=None,
                hyperlink=None,
                enumerated=False,
                marker="",
            ),
        ],
        pictures=[],
        tables=[],
        key_value_items=[],
        form_items=[],
        pages={},
    )


@pytest.fixture
def sample_metadata():
    return DocumentMetadata(
        file_type=FileType.STRING,
        file_name="判斷日文自他動詞",
    )

class TestDoclingChunkProcessor:
    def test_process_docling_chunk_basic(self, sample_doc, sample_metadata):
        processor = DoclingChunkProcessor()
        chunks = processor.process(sample_doc, sample_metadata)
        assert isinstance(chunks, list)
        assert all(hasattr(chunk, "content") for chunk in chunks)
        # chunk 欄位基本型別檢查
        for chunk in chunks:
            assert isinstance(chunk.content, str)
            assert hasattr(chunk.metadata, "is_chunk")
            assert "chunk_index" in chunk.metadata.chunk_info
            assert "chunk_size" in chunk.metadata.chunk_info
            assert "chunk_tokens" in chunk.metadata.chunk_info
        # 至少有一塊
        assert len(chunks) == 5

    def test_process_docling_chunk_tokens_match(self, sample_doc, sample_metadata):
        processor = DoclingChunkProcessor()
        chunks = processor.process(sample_doc, sample_metadata)
        for chunk in chunks:
            token_count = processor.tokenizer.count_tokens(chunk.content)
            assert chunk.metadata.chunk_info["chunk_tokens"] == token_count