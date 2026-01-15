# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for LanceDB VDB functionality."""

import json
import os
import shutil
import sys
import tempfile
import unittest
from concurrent.futures import Future
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from langchain_core.documents import Document


# Mock lancedb at import time for tests
mock_lancedb = MagicMock()
sys.modules['lancedb'] = mock_lancedb

from nvidia_rag.utils.vdb.lancedb.lancedb_vdb import LanceDBVDB


class TestLanceDBVDB(unittest.TestCase):
    """Test cases for LanceDBVDB class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_uri = self.temp_dir
        self.collection_name = "test_collection"
        self.embedding_model = Mock()
        self.meta_dataframe = pd.DataFrame(
            {"source": ["doc1"], "field1": ["value1"]}
        )
        self.meta_source_field = "source"
        self.meta_fields = ["field1"]
        # Reset the mock for each test
        mock_lancedb.reset_mock()
        self.mock_db = Mock()
        mock_lancedb.connect.return_value = self.mock_db

    def tearDown(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init(self):
        """Test LanceDBVDB initialization."""
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768

        lancedb_vdb = LanceDBVDB(
            db_uri=self.db_uri,
            collection_name=self.collection_name,
            embedding_model=self.embedding_model,
            config=mock_config,
            hybrid=False,
            dense_dim=768,
        )

        self.assertEqual(lancedb_vdb._db_uri, self.db_uri)
        self.assertEqual(lancedb_vdb.collection_name, self.collection_name)
        self.assertEqual(lancedb_vdb._embedding_model, self.embedding_model)
        mock_lancedb.connect.assert_called_with(self.db_uri)

    def test_collection_name_property(self):
        """Test collection_name property getter and setter."""
        lancedb_vdb = LanceDBVDB(
            db_uri=self.db_uri,
            collection_name="initial_name",
        )

        self.assertEqual(lancedb_vdb.collection_name, "initial_name")

        lancedb_vdb.collection_name = "new_name"
        self.assertEqual(lancedb_vdb.collection_name, "new_name")

    def test_table_exists(self):
        """Test _table_exists method."""
        self.mock_db.table_names.return_value = ["existing_table", "another_table"]

        lancedb_vdb = LanceDBVDB(db_uri=self.db_uri, collection_name="test")

        self.assertTrue(lancedb_vdb._table_exists("existing_table"))
        self.assertTrue(lancedb_vdb._table_exists("another_table"))
        self.assertFalse(lancedb_vdb._table_exists("non_existing"))

    def test_create_collection_new(self):
        """Test create_collection method when collection doesn't exist."""
        self.mock_db.table_names.return_value = []

        lancedb_vdb = LanceDBVDB(db_uri=self.db_uri, collection_name="test")
        lancedb_vdb.create_collection("new_collection", dimension=768)

        self.mock_db.create_table.assert_called()

    @patch("nvidia_rag.utils.vdb.lancedb.lancedb_vdb.logger")
    def test_create_collection_exists(self, mock_logger):
        """Test create_collection when collection already exists."""
        self.mock_db.table_names.return_value = ["existing_collection"]

        lancedb_vdb = LanceDBVDB(db_uri=self.db_uri, collection_name="test")
        lancedb_vdb.create_collection("existing_collection", dimension=768)

        # create_table should NOT be called for existing collection
        # (it's called once during __init__ for metadata schema init attempt)
        mock_logger.info.assert_called()

    def test_check_collection_exists(self):
        """Test check_collection_exists method."""
        self.mock_db.table_names.return_value = ["existing"]

        lancedb_vdb = LanceDBVDB(db_uri=self.db_uri, collection_name="test")

        self.assertTrue(lancedb_vdb.check_collection_exists("existing"))
        self.assertFalse(lancedb_vdb.check_collection_exists("non_existing"))

    def test_create_index(self):
        """Test create_index method."""
        self.mock_db.table_names.return_value = []

        lancedb_vdb = LanceDBVDB(
            db_uri=self.db_uri,
            collection_name="test_collection",
            dense_dim=768,
        )
        lancedb_vdb.create_index(collection_name="test_collection", dimension=768)

        self.mock_db.create_table.assert_called()

    @patch("nvidia_rag.utils.vdb.lancedb.lancedb_vdb.logger")
    def test_write_to_index(self, mock_logger):
        """Test write_to_index method."""
        mock_table = Mock()
        self.mock_db.table_names.return_value = ["test_collection"]
        self.mock_db.open_table.return_value = mock_table

        cleaned_records = [
            {
                "text": "test text 1",
                "vector": [0.1, 0.2, 0.3],
                "source": {"source_name": "doc1.pdf"},
                "content_metadata": {"title": "Test Doc 1"},
            },
            {
                "text": "test text 2",
                "vector": [0.4, 0.5, 0.6],
                "source": {"source_name": "doc2.pdf"},
                "content_metadata": {"title": "Test Doc 2"},
            },
        ]

        # Mock nv_ingest_client.util.milvus module
        mock_milvus_module = MagicMock()
        mock_milvus_module.cleanup_records.return_value = cleaned_records
        mock_milvus_module.pandas_file_reader.return_value = None
        sys.modules['nv_ingest_client'] = MagicMock()
        sys.modules['nv_ingest_client.util'] = MagicMock()
        sys.modules['nv_ingest_client.util.milvus'] = mock_milvus_module

        lancedb_vdb = LanceDBVDB(
            db_uri=self.db_uri,
            collection_name="test_collection",
        )

        records = [{"raw": "record1"}, {"raw": "record2"}]
        lancedb_vdb.write_to_index(records, collection_name="test_collection")

        mock_milvus_module.cleanup_records.assert_called_once()
        mock_table.add.assert_called_once()

    def test_retrieval_not_implemented(self):
        """Test retrieval method raises NotImplementedError."""
        lancedb_vdb = LanceDBVDB(db_uri=self.db_uri, collection_name="test")

        with self.assertRaises(NotImplementedError):
            lancedb_vdb.retrieval(["query1", "query2"])

    def test_reindex_not_implemented(self):
        """Test reindex method raises NotImplementedError."""
        lancedb_vdb = LanceDBVDB(db_uri=self.db_uri, collection_name="test")

        with self.assertRaises(NotImplementedError):
            lancedb_vdb.reindex([{"record": "data"}])

    def test_run(self):
        """Test run method."""
        self.mock_db.table_names.return_value = []

        lancedb_vdb = LanceDBVDB(
            db_uri=self.db_uri,
            collection_name="test_collection",
        )
        lancedb_vdb.create_index = Mock()
        lancedb_vdb.write_to_index = Mock()

        records = [{"test": "data"}]
        lancedb_vdb.run(records)

        lancedb_vdb.create_index.assert_called_once()
        lancedb_vdb.write_to_index.assert_called_once_with(
            records, collection_name="test_collection"
        )

    def test_run_async_with_list(self):
        """Test run_async method with list input."""
        self.mock_db.table_names.return_value = []

        lancedb_vdb = LanceDBVDB(
            db_uri=self.db_uri,
            collection_name="test_collection",
        )
        lancedb_vdb.create_index = Mock()
        lancedb_vdb.write_to_index = Mock()

        records = [{"test": "data"}]
        result = lancedb_vdb.run_async(records)

        self.assertEqual(result, records)
        lancedb_vdb.create_index.assert_called_once()
        lancedb_vdb.write_to_index.assert_called_once()

    def test_run_async_with_future(self):
        """Test run_async method with Future input."""
        self.mock_db.table_names.return_value = []

        lancedb_vdb = LanceDBVDB(
            db_uri=self.db_uri,
            collection_name="test_collection",
        )
        lancedb_vdb.create_index = Mock()
        lancedb_vdb.write_to_index = Mock()

        records = [{"test": "data"}]
        mock_future = Mock(spec=Future)
        mock_future.result.return_value = records

        result = lancedb_vdb.run_async(mock_future)

        self.assertEqual(result, records)
        mock_future.result.assert_called_once()

    def test_get_collection(self):
        """Test get_collection method."""
        mock_table = Mock()
        mock_table.count_rows.return_value = 100
        self.mock_db.table_names.return_value = ["collection1", "collection2"]
        self.mock_db.open_table.return_value = mock_table

        lancedb_vdb = LanceDBVDB(db_uri=self.db_uri, collection_name="test")
        lancedb_vdb.create_metadata_schema_collection = Mock()
        lancedb_vdb.create_document_info_collection = Mock()
        lancedb_vdb.get_metadata_schema = Mock(return_value=[])
        lancedb_vdb.get_document_info = Mock(return_value={})

        result = lancedb_vdb.get_collection()

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["collection_name"], "collection1")
        self.assertEqual(result[0]["num_entities"], 100)
        lancedb_vdb.create_metadata_schema_collection.assert_called_once()
        lancedb_vdb.create_document_info_collection.assert_called_once()

    @patch("nvidia_rag.utils.vdb.lancedb.lancedb_vdb.logger")
    def test_delete_collections_success(self, mock_logger):
        """Test delete_collections method with successful deletion."""
        self.mock_db.table_names.return_value = ["collection1", "collection2"]

        lancedb_vdb = LanceDBVDB(db_uri=self.db_uri, collection_name="test")
        lancedb_vdb._delete_from_system_table = Mock()

        result = lancedb_vdb.delete_collections(["collection1", "collection2"])

        self.assertEqual(result["total_success"], 2)
        self.assertEqual(result["total_failed"], 0)
        self.assertEqual(self.mock_db.drop_table.call_count, 2)

    @patch("nvidia_rag.utils.vdb.lancedb.lancedb_vdb.logger")
    def test_delete_collections_not_found(self, mock_logger):
        """Test delete_collections method when collection not found."""
        self.mock_db.table_names.return_value = []

        lancedb_vdb = LanceDBVDB(db_uri=self.db_uri, collection_name="test")

        result = lancedb_vdb.delete_collections(["non_existing"])

        self.assertEqual(result["total_success"], 0)
        self.assertEqual(result["total_failed"], 1)
        self.mock_db.drop_table.assert_not_called()

    def test_create_metadata_schema_collection(self):
        """Test create_metadata_schema_collection method."""
        self.mock_db.table_names.return_value = []

        lancedb_vdb = LanceDBVDB(db_uri=self.db_uri, collection_name="test")
        lancedb_vdb._metadata_schema_table_initialized = False
        lancedb_vdb.create_metadata_schema_collection()

        self.mock_db.create_table.assert_called()
        self.assertTrue(lancedb_vdb._metadata_schema_table_initialized)

    def test_create_metadata_schema_collection_already_initialized(self):
        """Test create_metadata_schema_collection when already initialized."""
        lancedb_vdb = LanceDBVDB(db_uri=self.db_uri, collection_name="test")
        lancedb_vdb._metadata_schema_table_initialized = True
        self.mock_db.reset_mock()

        lancedb_vdb.create_metadata_schema_collection()

        self.mock_db.create_table.assert_not_called()

    def test_add_metadata_schema(self):
        """Test add_metadata_schema method."""
        mock_table = Mock()
        self.mock_db.table_names.return_value = ["metadata_schema"]
        self.mock_db.open_table.return_value = mock_table

        lancedb_vdb = LanceDBVDB(db_uri=self.db_uri, collection_name="test")
        lancedb_vdb._metadata_schema_table_initialized = True

        metadata_schema = [{"name": "title", "type": "string"}]
        lancedb_vdb.add_metadata_schema("test_collection", metadata_schema)

        mock_table.add.assert_called_once()

    def test_get_metadata_schema_found(self):
        """Test get_metadata_schema when schema is found."""
        mock_table = Mock()
        mock_search = Mock()
        mock_where = Mock()
        mock_limit = Mock()

        mock_df = pd.DataFrame({
            "metadata_schema": [json.dumps([{"name": "title", "type": "string"}])]
        })
        mock_limit.to_pandas.return_value = mock_df
        mock_where.limit.return_value = mock_limit
        mock_search.where.return_value = mock_where
        mock_table.search.return_value = mock_search

        self.mock_db.table_names.return_value = ["metadata_schema"]
        self.mock_db.open_table.return_value = mock_table

        lancedb_vdb = LanceDBVDB(db_uri=self.db_uri, collection_name="test")

        result = lancedb_vdb.get_metadata_schema("test_collection")

        self.assertEqual(result, [{"name": "title", "type": "string"}])

    def test_get_metadata_schema_not_found(self):
        """Test get_metadata_schema when schema is not found."""
        self.mock_db.table_names.return_value = []

        lancedb_vdb = LanceDBVDB(db_uri=self.db_uri, collection_name="test")

        result = lancedb_vdb.get_metadata_schema("test_collection")

        self.assertEqual(result, [])

    def test_create_document_info_collection(self):
        """Test create_document_info_collection method."""
        self.mock_db.table_names.return_value = []

        lancedb_vdb = LanceDBVDB(db_uri=self.db_uri, collection_name="test")
        lancedb_vdb._document_info_table_initialized = False
        lancedb_vdb.create_document_info_collection()

        self.mock_db.create_table.assert_called()
        self.assertTrue(lancedb_vdb._document_info_table_initialized)

    def test_add_document_info(self):
        """Test add_document_info method."""
        mock_table = Mock()
        self.mock_db.table_names.return_value = ["document_info"]
        self.mock_db.open_table.return_value = mock_table

        lancedb_vdb = LanceDBVDB(db_uri=self.db_uri, collection_name="test")
        lancedb_vdb._document_info_table_initialized = True

        lancedb_vdb.add_document_info(
            info_type="document",
            collection_name="test_collection",
            document_name="test.pdf",
            info_value={"pages": 10},
        )

        mock_table.add.assert_called_once()

    def test_get_document_info_found(self):
        """Test get_document_info when info is found."""
        mock_table = Mock()
        mock_search = Mock()
        mock_where = Mock()
        mock_limit = Mock()

        mock_df = pd.DataFrame({
            "info_value": [json.dumps({"pages": 10})]
        })
        mock_limit.to_pandas.return_value = mock_df
        mock_where.limit.return_value = mock_limit
        mock_search.where.return_value = mock_where
        mock_table.search.return_value = mock_search

        self.mock_db.table_names.return_value = ["document_info"]
        self.mock_db.open_table.return_value = mock_table

        lancedb_vdb = LanceDBVDB(db_uri=self.db_uri, collection_name="test")

        result = lancedb_vdb.get_document_info(
            info_type="document",
            collection_name="test_collection",
            document_name="test.pdf",
        )

        self.assertEqual(result, {"pages": 10})

    def test_get_document_info_not_found(self):
        """Test get_document_info when info is not found."""
        self.mock_db.table_names.return_value = []

        lancedb_vdb = LanceDBVDB(db_uri=self.db_uri, collection_name="test")

        result = lancedb_vdb.get_document_info(
            info_type="document",
            collection_name="test_collection",
            document_name="test.pdf",
        )

        self.assertEqual(result, {})

    def test_get_catalog_metadata(self):
        """Test get_catalog_metadata method."""
        lancedb_vdb = LanceDBVDB(db_uri=self.db_uri, collection_name="test")
        lancedb_vdb.get_document_info = Mock(return_value={"description": "Test"})

        result = lancedb_vdb.get_catalog_metadata("test_collection")

        self.assertEqual(result, {"description": "Test"})
        lancedb_vdb.get_document_info.assert_called_once_with(
            info_type="catalog",
            collection_name="test_collection",
            document_name="NA",
        )

    def test_update_catalog_metadata(self):
        """Test update_catalog_metadata method."""
        lancedb_vdb = LanceDBVDB(db_uri=self.db_uri, collection_name="test")
        lancedb_vdb.get_catalog_metadata = Mock(return_value={"existing": "value"})
        lancedb_vdb.add_document_info = Mock()

        lancedb_vdb.update_catalog_metadata(
            collection_name="test_collection",
            updates={"new_field": "new_value"},
        )

        lancedb_vdb.add_document_info.assert_called_once()
        call_args = lancedb_vdb.add_document_info.call_args[1]
        self.assertIn("existing", call_args["info_value"])
        self.assertIn("new_field", call_args["info_value"])
        self.assertIn("last_updated", call_args["info_value"])

    def test_get_document_catalog_metadata(self):
        """Test get_document_catalog_metadata method."""
        lancedb_vdb = LanceDBVDB(db_uri=self.db_uri, collection_name="test")
        lancedb_vdb.get_document_info = Mock(
            return_value={"description": "Test doc", "tags": ["tag1"]}
        )

        result = lancedb_vdb.get_document_catalog_metadata(
            collection_name="test_collection",
            document_name="test.pdf",
        )

        self.assertEqual(result["description"], "Test doc")
        self.assertEqual(result["tags"], ["tag1"])

    def test_update_document_catalog_metadata(self):
        """Test update_document_catalog_metadata method."""
        lancedb_vdb = LanceDBVDB(db_uri=self.db_uri, collection_name="test")
        lancedb_vdb.get_document_info = Mock(
            return_value={"description": "Old", "tags": []}
        )
        lancedb_vdb.add_document_info = Mock()

        lancedb_vdb.update_document_catalog_metadata(
            collection_name="test_collection",
            document_name="test.pdf",
            updates={"description": "New", "tags": ["new_tag"]},
        )

        lancedb_vdb.add_document_info.assert_called_once()
        call_args = lancedb_vdb.add_document_info.call_args[1]
        self.assertEqual(call_args["info_value"]["description"], "New")
        self.assertEqual(call_args["info_value"]["tags"], ["new_tag"])

    def test_add_collection_name_to_retreived_docs(self):
        """Test _add_collection_name_to_retreived_docs static method."""
        docs = [
            Document(page_content="doc1", metadata={"source": "file1.pdf"}),
            Document(page_content="doc2", metadata={"source": "file2.pdf"}),
        ]

        result = LanceDBVDB._add_collection_name_to_retreived_docs(
            docs, "test_collection"
        )

        for doc in result:
            self.assertEqual(doc.metadata["collection_name"], "test_collection")

        self.assertEqual(result[0].metadata["source"], "file1.pdf")
        self.assertEqual(result[1].metadata["source"], "file2.pdf")

    def test_add_collection_name_parses_json_source(self):
        """Test _add_collection_name_to_retreived_docs parses JSON source."""
        source_json = json.dumps({"source_name": "file1.pdf", "path": "/path/to"})
        docs = [
            Document(page_content="doc1", metadata={"source": source_json}),
        ]

        result = LanceDBVDB._add_collection_name_to_retreived_docs(
            docs, "test_collection"
        )

        self.assertIsInstance(result[0].metadata["source"], dict)
        self.assertEqual(result[0].metadata["source"]["source_name"], "file1.pdf")


class TestLanceDBVDBAsync(unittest.TestCase):
    """Async test cases for LanceDBVDB class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_uri = self.temp_dir
        mock_lancedb.reset_mock()
        self.mock_db = Mock()
        mock_lancedb.connect.return_value = self.mock_db

    def tearDown(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_check_health_healthy(self):
        """Test check_health method returns healthy status."""
        self.mock_db.table_names.return_value = ["table1", "table2"]

        lancedb_vdb = LanceDBVDB(db_uri=self.db_uri, collection_name="test")
        result = await lancedb_vdb.check_health()

        self.assertEqual(result["status"], "healthy")
        self.assertEqual(result["service"], "LanceDB")
        self.assertEqual(result["tables"], 2)
        self.assertIn("latency_ms", result)

    @pytest.mark.asyncio
    async def test_check_health_no_uri(self):
        """Test check_health method when no URI provided."""
        lancedb_vdb = LanceDBVDB(db_uri=self.db_uri, collection_name="test")
        lancedb_vdb._db_uri = ""

        result = await lancedb_vdb.check_health()

        self.assertEqual(result["status"], "skipped")

    @pytest.mark.asyncio
    async def test_check_health_error(self):
        """Test check_health method when error occurs."""
        self.mock_db.table_names.side_effect = Exception("Connection failed")

        lancedb_vdb = LanceDBVDB(db_uri=self.db_uri, collection_name="test")
        result = await lancedb_vdb.check_health()

        self.assertEqual(result["status"], "error")
        self.assertIn("Connection failed", result["error"])


if __name__ == "__main__":
    unittest.main()

