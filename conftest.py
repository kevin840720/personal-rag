# -*- encoding: utf-8 -*-
"""
@File:
    conftest.py
@Time:
    2023/08/24 13:34:58
@Author:
    Kevin Wang
@Desc:
    None
@Ref:
    https://stackoverflow.com/questions/42996270/change-pytest-rootdir
"""

from pathlib import Path
import sys

SKIP_ELASTICSEARCH_TESTS = False
SKIP_PGVECTOR_TESTS = False
SKIP_REAL_FILE_TEST = False

src_path = Path(__file__).parent.joinpath('src')
sys.path.append(src_path.__str__())