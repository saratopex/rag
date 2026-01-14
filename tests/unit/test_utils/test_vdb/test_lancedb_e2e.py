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

"""End-to-end tests for LanceDB VDB - tests real database operations without mocks."""

import json
import shutil
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import pytest

try:
    import lancedb
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False

from nvidia_rag.utils.vdb.lancedb.lancedb_vdb import LanceDBVDB


@pytest.mark.skipif(not LANCEDB_AVAILABLE, reason="lancedb not installed")
class TestLanceDBE2E(unittest.TestCase):
    """End-to-end tests using a real LanceDB database."""

    def setUp(self):
        """Set up test fixtures with a real temp directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_uri = self.temp_dir
        self.collection_name = "test_e2e_collection"
        self.dense_dim = 384  # Common embedding dimension

        # Create a mock embedding model that returns consistent embeddings
        self.mock_embedding_model = Mock()
        # Return deterministic embeddings based on text content
        def embed_documents(texts):
            embeddings = []
            for text in texts:
                # Create a simple hash-based embedding for reproducibility
                np.random.seed(hash(text) % (2**32))
                embeddings.append(np.random.randn(self.dense_dim).tolist())
            return embeddings

        def embed_query(text):
            np.random.seed(hash(text) % (2**32))
            return np.random.randn(self.dense_dim).tolist()

        self.mock_embedding_model.embed_documents = embed_documents
        self.mock_embedding_model.embed_query = embed_query

    def tearDown(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_collection_and_verify_exists(self):
        """Test creating a collection and verifying it exists."""
        vdb = LanceDBVDB(
            db_uri=self.db_uri,
            collection_name=self.collection_name,
            embedding_model=self.mock_embedding_model,
            dense_dim=self.dense_dim,
        )

        # Initially collection should not exist
        self.assertFalse(vdb.check_collection_exists(self.collection_name))

        # Create collection
        vdb.create_collection(self.collection_name, dimension=self.dense_dim)

        # Now it should exist
        self.assertTrue(vdb.check_collection_exists(self.collection_name))

    def test_ingest_documents_and_retrieve(self):
        """Test full ingestion and retrieval flow."""
        vdb = LanceDBVDB(
            db_uri=self.db_uri,
            collection_name=self.collection_name,
            embedding_model=self.mock_embedding_model,
            dense_dim=self.dense_dim,
        )

        # Create collection
        vdb.create_collection(self.collection_name, dimension=self.dense_dim)

        # Create sample documents with embeddings
        sample_docs = [
            {
                "text": "NVIDIA GPUs are used for deep learning and AI workloads.",
                "vector": np.random.randn(self.dense_dim).tolist(),
                "source": json.dumps({"source_name": "nvidia_gpu.pdf"}),
                "content_metadata": json.dumps({"page_number": 1}),
            },
            {
                "text": "LanceDB is a vector database optimized for AI applications.",
                "vector": np.random.randn(self.dense_dim).tolist(),
                "source": json.dumps({"source_name": "lancedb_intro.pdf"}),
                "content_metadata": json.dumps({"page_number": 1}),
            },
            {
                "text": "RAG combines retrieval with generation for better responses.",
                "vector": np.random.randn(self.dense_dim).tolist(),
                "source": json.dumps({"source_name": "rag_overview.pdf"}),
                "content_metadata": json.dumps({"page_number": 1}),
            },
        ]

        # Insert documents directly into LanceDB
        db = lancedb.connect(self.db_uri)
        table = db.open_table(self.collection_name)
        table.add(sample_docs)

        # Verify documents were added
        self.assertEqual(table.count_rows(), 3)

        # Test retrieval - query the table directly to verify data
        df = table.to_pandas()
        self.assertEqual(len(df), 3)

        # Verify document structure - check that sources are stored correctly
        sources = df["source"].tolist()
        self.assertIn("nvidia_gpu.pdf", str(sources))
        self.assertIn("lancedb_intro.pdf", str(sources))
        self.assertIn("rag_overview.pdf", str(sources))

        # Test vector search
        query_vector = np.random.randn(self.dense_dim).tolist()
        results = table.search(query_vector).limit(2).to_list()
        self.assertEqual(len(results), 2)
        self.assertIn("text", results[0])

    def test_delete_documents_by_source(self):
        """Test deleting documents by source."""
        vdb = LanceDBVDB(
            db_uri=self.db_uri,
            collection_name=self.collection_name,
            embedding_model=self.mock_embedding_model,
            dense_dim=self.dense_dim,
        )

        # Create collection and add documents
        vdb.create_collection(self.collection_name, dimension=self.dense_dim)

        sample_docs = [
            {
                "text": "Document from source A",
                "vector": np.random.randn(self.dense_dim).tolist(),
                "source": json.dumps({"source_name": "source_a.pdf"}),
                "content_metadata": json.dumps({}),
            },
            {
                "text": "Another document from source A",
                "vector": np.random.randn(self.dense_dim).tolist(),
                "source": json.dumps({"source_name": "source_a.pdf"}),
                "content_metadata": json.dumps({}),
            },
            {
                "text": "Document from source B",
                "vector": np.random.randn(self.dense_dim).tolist(),
                "source": json.dumps({"source_name": "source_b.pdf"}),
                "content_metadata": json.dumps({}),
            },
        ]

        db = lancedb.connect(self.db_uri)
        table = db.open_table(self.collection_name)
        table.add(sample_docs)

        # Verify initial count
        self.assertEqual(table.count_rows(), 3)

        # Delete documents from source_a.pdf
        result = vdb.delete_documents(self.collection_name, ["source_a.pdf"])
        self.assertTrue(result)

        # Re-open table to get fresh count after delete
        table = db.open_table(self.collection_name)
        remaining_count = table.count_rows()

        # Should have only 1 document remaining (from source_b.pdf)
        self.assertEqual(remaining_count, 1)

        # Verify the remaining document is from source_b
        df = table.to_pandas()
        self.assertIn("source_b.pdf", str(df["source"].tolist()))

    def test_delete_collection(self):
        """Test deleting a collection."""
        vdb = LanceDBVDB(
            db_uri=self.db_uri,
            collection_name=self.collection_name,
            embedding_model=self.mock_embedding_model,
            dense_dim=self.dense_dim,
        )

        # Create collection
        vdb.create_collection(self.collection_name, dimension=self.dense_dim)
        self.assertTrue(vdb.check_collection_exists(self.collection_name))

        # Delete collection
        result = vdb.delete_collections([self.collection_name])
        self.assertEqual(result["total_success"], 1)
        self.assertEqual(result["total_failed"], 0)

        # Verify it's gone
        self.assertFalse(vdb.check_collection_exists(self.collection_name))

    def test_metadata_schema_persistence(self):
        """Test that metadata schemas are persisted and retrieved correctly."""
        vdb = LanceDBVDB(
            db_uri=self.db_uri,
            collection_name=self.collection_name,
            embedding_model=self.mock_embedding_model,
            dense_dim=self.dense_dim,
        )

        # Create metadata schema collection
        vdb.create_metadata_schema_collection()

        # Add a schema
        test_schema = [
            {"name": "author", "type": "string"},
            {"name": "date", "type": "string"},
            {"name": "page_count", "type": "int"},
        ]
        vdb.add_metadata_schema(self.collection_name, test_schema)

        # Retrieve and verify
        retrieved_schema = vdb.get_metadata_schema(self.collection_name)
        self.assertEqual(len(retrieved_schema), 3)
        self.assertEqual(retrieved_schema[0]["name"], "author")
        self.assertEqual(retrieved_schema[1]["name"], "date")
        self.assertEqual(retrieved_schema[2]["name"], "page_count")

    def test_document_info_persistence(self):
        """Test that document info is persisted and retrieved correctly."""
        vdb = LanceDBVDB(
            db_uri=self.db_uri,
            collection_name=self.collection_name,
            embedding_model=self.mock_embedding_model,
            dense_dim=self.dense_dim,
        )

        # Create document info collection
        vdb.create_document_info_collection()

        # Add document info
        doc_info = {
            "total_pages": 10,
            "has_images": True,
            "file_size": 1024000,
        }
        vdb.add_document_info(
            info_type="document",
            collection_name=self.collection_name,
            document_name="test_doc.pdf",
            info_value=doc_info,
        )

        # Retrieve and verify
        retrieved_info = vdb.get_document_info(
            info_type="document",
            collection_name=self.collection_name,
            document_name="test_doc.pdf",
        )
        self.assertEqual(retrieved_info["total_pages"], 10)
        self.assertTrue(retrieved_info["has_images"])
        self.assertEqual(retrieved_info["file_size"], 1024000)

    def test_get_collection_lists_all_tables(self):
        """Test that get_collection returns info about all tables."""
        vdb = LanceDBVDB(
            db_uri=self.db_uri,
            collection_name="collection1",
            embedding_model=self.mock_embedding_model,
            dense_dim=self.dense_dim,
        )

        # Create multiple collections
        vdb.create_collection("collection1", dimension=self.dense_dim)
        vdb.create_collection("collection2", dimension=self.dense_dim)
        vdb.create_collection("collection3", dimension=self.dense_dim)

        # Get all collections
        collections = vdb.get_collection()

        # Should have at least 3 collections (might have system tables too)
        collection_names = [c["collection_name"] for c in collections]
        self.assertIn("collection1", collection_names)
        self.assertIn("collection2", collection_names)
        self.assertIn("collection3", collection_names)

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test that health check returns healthy status."""
        vdb = LanceDBVDB(
            db_uri=self.db_uri,
            collection_name=self.collection_name,
            embedding_model=self.mock_embedding_model,
            dense_dim=self.dense_dim,
        )

        health = await vdb.check_health()
        self.assertEqual(health["status"], "healthy")
        self.assertEqual(health["service"], "LanceDB")
        self.assertIn("latency_ms", health)


@pytest.mark.skipif(not LANCEDB_AVAILABLE, reason="lancedb not installed")
class TestLanceDBVectorSearch(unittest.TestCase):
    """Test vector similarity search functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_uri = self.temp_dir
        self.collection_name = "vector_search_test"
        self.dense_dim = 128

        # Create a deterministic embedding model for testing
        self.mock_embedding_model = Mock()

        def embed_query(text):
            # Return embeddings that are similar for related content
            if "GPU" in text or "NVIDIA" in text:
                base = np.ones(self.dense_dim) * 0.5
            elif "database" in text or "LanceDB" in text:
                base = np.ones(self.dense_dim) * -0.5
            else:
                base = np.zeros(self.dense_dim)
            # Add small noise
            return (base + np.random.randn(self.dense_dim) * 0.1).tolist()

        self.mock_embedding_model.embed_query = embed_query

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_similarity_search_returns_relevant_docs(self):
        """Test that similarity search returns the most relevant documents."""
        vdb = LanceDBVDB(
            db_uri=self.db_uri,
            collection_name=self.collection_name,
            embedding_model=self.mock_embedding_model,
            dense_dim=self.dense_dim,
        )

        # Create collection
        vdb.create_collection(self.collection_name, dimension=self.dense_dim)

        # Create documents with embeddings that cluster by topic
        gpu_embedding = (np.ones(self.dense_dim) * 0.5).tolist()
        db_embedding = (np.ones(self.dense_dim) * -0.5).tolist()
        neutral_embedding = np.zeros(self.dense_dim).tolist()

        sample_docs = [
            {
                "text": "NVIDIA GPUs accelerate deep learning training.",
                "vector": gpu_embedding,
                "source": json.dumps({"source_name": "gpu_doc.pdf"}),
                "content_metadata": json.dumps({}),
            },
            {
                "text": "LanceDB is a vector database for AI.",
                "vector": db_embedding,
                "source": json.dumps({"source_name": "db_doc.pdf"}),
                "content_metadata": json.dumps({}),
            },
            {
                "text": "Weather today is sunny.",
                "vector": neutral_embedding,
                "source": json.dumps({"source_name": "weather.pdf"}),
                "content_metadata": json.dumps({}),
            },
        ]

        db = lancedb.connect(self.db_uri)
        table = db.open_table(self.collection_name)
        table.add(sample_docs)

        # Search for GPU-related content
        query_embedding = (np.ones(self.dense_dim) * 0.5).tolist()
        results = table.search(query_embedding).limit(1).to_list()

        # Should return the GPU document as most similar
        self.assertEqual(len(results), 1)
        self.assertIn("NVIDIA", results[0]["text"])


if __name__ == "__main__":
    unittest.main()

