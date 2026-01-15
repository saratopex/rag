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

"""
This module contains the implementation of the LanceDBVDB class,
which provides LanceDB vector database operations for RAG applications.
Extends VDBRagIngest for both nv-ingest operations and RAG-specific functionality.

LanceDB is an embedded/serverless vector database that stores data in Lance format
and supports both local file system and cloud storage (S3, GCS, Azure Blob).

NV-Ingest Client VDB Operations:
1. create_index: Create a table in LanceDB
2. write_to_index: Write records to the LanceDB table
3. retrieval: Retrieve documents from LanceDB based on queries
4. reindex: Reindex documents in LanceDB
5. run: Run the process of ingestion of records to the LanceDB table

Collection Management:
6. create_collection: Create a new collection with specified dimensions and type
7. check_collection_exists: Check if the specified collection exists
8. get_collection: Retrieve all collections with their metadata schemas
9. delete_collections: Delete multiple collections and their associated metadata

Document Management:
10. get_documents: Retrieve all unique documents from the specified collection
11. delete_documents: Remove documents matching the specified source values

Metadata Schema Management:
12. create_metadata_schema_collection: Initialize the metadata schema storage table
13. add_metadata_schema: Store metadata schema configuration for the collection
14. get_metadata_schema: Retrieve the metadata schema for the specified collection

Document Info Management:
15. create_document_info_collection: Initialize the document info storage table
16. add_document_info: Store document info for a collection or document
17. get_document_info: Retrieve document info for a specified collection/document

Retrieval Operations:
18. retrieval_langchain: Perform semantic search and return top-k relevant documents
19. get_langchain_vectorstore: Get the vectorstore for a collection
20. _add_collection_name_to_retreived_docs: Add the collection name to the retrieved documents
"""

import json
import logging
import os
import time
from concurrent.futures import Future
from typing import Any

import pyarrow as pa
import requests
from langchain_core.documents import Document
from langchain_core.runnables import RunnableAssign, RunnableLambda
from opentelemetry import context as otel_context

from nvidia_rag.rag_server.main import APIError
from nvidia_rag.rag_server.response_generator import ErrorCodeMapping
from nvidia_rag.utils.configuration import AppConfig
from nvidia_rag.utils.vdb import (
    DEFAULT_DOCUMENT_INFO_COLLECTION,
    DEFAULT_METADATA_SCHEMA_COLLECTION,
    SYSTEM_COLLECTIONS,
)
from nvidia_rag.utils.vdb.vdb_ingest_base import VDBRagIngest

logger = logging.getLogger(__name__)


def _get_current_timestamp() -> str:
    """Get the current timestamp in ISO format."""
    from datetime import datetime, UTC
    return datetime.now(UTC).isoformat()


def _perform_document_info_aggregation(
    existing_info: dict[str, Any],
    new_info: dict[str, Any],
) -> dict[str, Any]:
    """
    Aggregate document info by merging existing and new info.
    For numeric values, sum them; for lists, extend them; otherwise replace.
    """
    result = existing_info.copy()
    for key, value in new_info.items():
        if key in result:
            if isinstance(value, (int, float)) and isinstance(result[key], (int, float)):
                result[key] = result[key] + value
            elif isinstance(value, list) and isinstance(result[key], list):
                result[key] = result[key] + value
            else:
                result[key] = value
        else:
            result[key] = value
    return result


# LanceDB table names for system collections
LANCEDB_METADATA_SCHEMA_TABLE = DEFAULT_METADATA_SCHEMA_COLLECTION
LANCEDB_DOCUMENT_INFO_TABLE = DEFAULT_DOCUMENT_INFO_COLLECTION


class LanceDBVDB(VDBRagIngest):
    """
    LanceDB vector database implementation for RAG applications.

    LanceDB is an embedded/serverless vector database that can work with:
    - Local file system (file:///path/to/db or just /path/to/db)
    - S3 (s3://bucket/path)
    - Google Cloud Storage (gs://bucket/path)
    - Azure Blob Storage (az://container/path)

    Inherits from VDBRagIngest which provides the abstract interface for both
    RAG retrieval and nv_ingest ingestion operations.
    """

    def __init__(
        self,
        db_uri: str,
        collection_name: str = "",
        embedding_model: Any = None,
        config: AppConfig | None = None,
        # Hybrid search configurations
        hybrid: bool = False,
        # Dimension of dense embeddings
        dense_dim: int = 2048,
        # Custom metadata configurations (optional)
        meta_dataframe: str | None = None,
        meta_source_field: str | None = None,
        meta_fields: list[str] | None = None,
    ):
        """
        Initialize LanceDBVDB instance.

        Args:
            db_uri: URI for LanceDB storage. Can be:
                - Local path: "/path/to/db" or "file:///path/to/db"
                - S3: "s3://bucket/path"
                - GCS: "gs://bucket/path"
                - Azure: "az://container/path"
            collection_name: Name of the collection/table
            embedding_model: Embedding model instance for retrieval
            config: AppConfig instance (optional, creates default if None)
            hybrid: Enable hybrid search (not yet supported in LanceDB langchain)
            dense_dim: Dimension of dense embeddings
            meta_dataframe: Path to CSV file containing custom metadata
            meta_source_field: Field name for source identification in metadata
            meta_fields: List of metadata field names to include
        """
        try:
            import lancedb
        except ImportError as e:
            raise ImportError(
                "lancedb is required for LanceDBVDB. "
                "Install with: pip install lancedb"
            ) from e

        self.config = config or AppConfig()
        self._db_uri = db_uri
        self._collection_name = collection_name
        self._embedding_model = embedding_model
        self.hybrid = hybrid
        self.dense_dim = dense_dim

        # Metadata fields for NV-Ingest Client
        self.meta_dataframe = meta_dataframe
        self.meta_source_field = meta_source_field
        self.meta_fields = meta_fields
        self.csv_file_path = meta_dataframe

        # Track if system tables have been initialized
        self._metadata_schema_table_initialized = False
        self._document_info_table_initialized = False

        # Connect to LanceDB
        try:
            self._db = lancedb.connect(self._db_uri)
            logger.info(f"Connected to LanceDB at {self._db_uri}")
        except Exception as e:
            logger.error(f"Failed to connect to LanceDB at {self._db_uri}: {e}")
            raise APIError(
                f"Vector database (LanceDB) is unavailable at {self._db_uri}. "
                f"Please verify the path exists and is accessible. Error: {str(e)}",
                ErrorCodeMapping.SERVICE_UNAVAILABLE,
            ) from e

    def close(self):
        """Close the LanceDB connection (no-op for LanceDB as it's embedded)."""
        pass

    def __enter__(self):
        """Enter the runtime context (for use as context manager)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context."""
        self.close()

    @property
    def collection_name(self) -> str:
        """Get the collection name."""
        return self._collection_name

    @collection_name.setter
    def collection_name(self, collection_name: str) -> None:
        """Set the collection name."""
        self._collection_name = collection_name

    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists in LanceDB."""
        return table_name in self._db.table_names()

    def _get_table(self, table_name: str):
        """Get a LanceDB table by name."""
        if not self._table_exists(table_name):
            return None
        return self._db.open_table(table_name)

    def _create_vector_table_schema(self, dimension: int) -> pa.Schema:
        """Create PyArrow schema for vector table."""
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dimension)),
            pa.field("source", pa.string()),
            pa.field("content_metadata", pa.string()),
        ])

    # -------------------------------------------------------------------------
    # NV-Ingest VDB Interface Methods
    # -------------------------------------------------------------------------
    def create_index(self, **kwargs) -> None:
        """Create a LanceDB table/index if it doesn't exist."""
        table_name = kwargs.get("collection_name", self._collection_name)
        dimension = kwargs.get("dimension", self.dense_dim)

        if self._table_exists(table_name):
            logger.info(f"LanceDB table {table_name} already exists")
            return

        schema = self._create_vector_table_schema(dimension)
        self._db.create_table(table_name, schema=schema)
        logger.info(f"Created LanceDB table: {table_name}")

    def write_to_index(self, records: list, **kwargs) -> None:
        """
        Write records to the LanceDB table in batches.

        Requires nv_ingest_client to be installed for record cleanup.
        Install with: pip install nvidia-rag[ingest]
        """
        try:
            from nv_ingest_client.util.milvus import cleanup_records, pandas_file_reader
        except ImportError as e:
            raise ImportError(
                "nv_ingest_client is required for write_to_index operation. "
                "Install with: pip install nvidia-rag[ingest]"
            ) from e

        table_name = kwargs.get("collection_name", self._collection_name)

        # Load meta_dataframe lazily if not already loaded
        meta_dataframe = self.meta_dataframe
        if meta_dataframe is None and self.csv_file_path is not None:
            meta_dataframe = pandas_file_reader(self.csv_file_path)

        # Clean up and flatten records
        cleaned_records = cleanup_records(
            records=records,
            meta_dataframe=meta_dataframe,
            meta_source_field=self.meta_source_field,
            meta_fields=self.meta_fields,
        )

        # Prepare data for LanceDB
        data = []
        for i, record in enumerate(cleaned_records):
            source = record.get("source", {})
            if isinstance(source, dict):
                source_str = json.dumps(source)
            else:
                source_str = str(source)

            content_metadata = record.get("content_metadata", {})
            if isinstance(content_metadata, dict):
                content_metadata_str = json.dumps(content_metadata)
            else:
                content_metadata_str = str(content_metadata)

            data.append({
                "id": f"{table_name}_{i}_{time.time_ns()}",
                "text": record.get("text", ""),
                "vector": record.get("vector", []),
                "source": source_str,
                "content_metadata": content_metadata_str,
            })

        if not data:
            logger.warning("No records to write to LanceDB")
            return

        # Get or create table
        table = self._get_table(table_name)
        if table is None:
            schema = self._create_vector_table_schema(len(data[0]["vector"]))
            table = self._db.create_table(table_name, schema=schema)

        # Write in batches
        batch_size = 1000
        total_records = len(data)
        uploaded_count = 0

        logger.info(f"Starting LanceDB ingestion for {total_records} records...")

        for i in range(0, total_records, batch_size):
            batch = data[i:i + batch_size]
            table.add(batch)
            uploaded_count += len(batch)

            if uploaded_count % (5 * batch_size) == 0 or uploaded_count == total_records:
                logger.info(
                    f"Ingested {uploaded_count}/{total_records} records into "
                    f"LanceDB table {table_name}"
                )

        logger.info(
            f"LanceDB ingestion completed. Total records: {uploaded_count}"
        )

    def retrieval(self, queries: list, **kwargs) -> list[dict[str, Any]]:
        """Retrieve documents from LanceDB based on queries."""
        raise NotImplementedError(
            "Use retrieval_langchain for RAG retrieval operations"
        )

    def reindex(self, records: list, **kwargs) -> None:
        """Reindex documents in LanceDB."""
        raise NotImplementedError("reindex is not yet implemented for LanceDBVDB")

    def run(self, records: list) -> None:
        """Run the process of ingestion of records to the LanceDB table."""
        self.create_index(collection_name=self._collection_name)
        self.write_to_index(records, collection_name=self._collection_name)

    def run_async(self, records: list | Future) -> list:
        """Run ingestion from either a list of records or a Future."""
        logger.info(f"Creating LanceDB table: {self._collection_name}")
        self.create_index(collection_name=self._collection_name)

        if isinstance(records, Future):
            records = records.result()

        logger.info(f"Writing to LanceDB table: {self._collection_name}")
        self.write_to_index(records, collection_name=self._collection_name)

        return records

    # -------------------------------------------------------------------------
    # VDBRag Collection Management
    # -------------------------------------------------------------------------
    async def check_health(self) -> dict[str, Any]:
        """Check LanceDB health status."""
        status = {
            "service": "LanceDB",
            "url": self._db_uri,
            "status": "unknown",
            "error": None,
        }

        if not self._db_uri:
            status["status"] = "skipped"
            status["error"] = "No URI provided"
            return status

        try:
            start_time = time.time()
            tables = self._db.table_names()
            status["status"] = "healthy"
            status["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            status["tables"] = len(tables)
        except Exception as e:
            status["status"] = "error"
            status["error"] = str(e)

        return status

    def create_collection(
        self,
        collection_name: str,
        dimension: int = 2048,
        collection_type: str = "text",
    ) -> None:
        """Create a new collection/table in LanceDB."""
        if self._table_exists(collection_name):
            logger.info(f"LanceDB table {collection_name} already exists")
            return

        schema = self._create_vector_table_schema(dimension)
        self._db.create_table(collection_name, schema=schema)
        logger.info(f"Created LanceDB table: {collection_name}")

    def check_collection_exists(self, collection_name: str) -> bool:
        """Check if a collection/table exists in LanceDB."""
        return self._table_exists(collection_name)

    def get_collection(self) -> list[dict[str, Any]]:
        """Get the list of collections/tables in LanceDB."""
        self.create_metadata_schema_collection()
        self.create_document_info_collection()

        tables = self._db.table_names()
        collection_info = []

        for table_name in tables:
            if table_name in SYSTEM_COLLECTIONS:
                continue

            table = self._db.open_table(table_name)
            num_entities = table.count_rows()

            metadata_schema = self.get_metadata_schema(table_name)
            catalog_data = self.get_document_info(
                info_type="catalog",
                collection_name=table_name,
                document_name="NA",
            )
            metrics_data = self.get_document_info(
                info_type="collection",
                collection_name=table_name,
                document_name="NA",
            )

            collection_info.append({
                "collection_name": table_name,
                "num_entities": num_entities,
                "metadata_schema": metadata_schema,
                "collection_info": {**metrics_data, **catalog_data},
            })

        return collection_info

    def delete_collections(self, collection_names: list[str]) -> dict[str, Any]:
        """Delete collections/tables from LanceDB."""
        deleted_collections = []
        failed_collections = []

        for collection_name in collection_names:
            try:
                if self._table_exists(collection_name):
                    self._db.drop_table(collection_name)
                    deleted_collections.append(collection_name)
                    logger.info(f"Deleted LanceDB table: {collection_name}")
                else:
                    failed_collections.append({
                        "collection_name": collection_name,
                        "error_message": f"Table {collection_name} not found.",
                    })
                    logger.warning(f"Table {collection_name} not found.")
            except Exception as e:
                failed_collections.append({
                    "collection_name": collection_name,
                    "error_message": str(e),
                })
                logger.exception(f"Failed to delete table {collection_name}")

        # Delete metadata schema and document info for deleted collections
        for collection_name in deleted_collections:
            try:
                self._delete_from_system_table(
                    LANCEDB_METADATA_SCHEMA_TABLE,
                    f"collection_name = '{collection_name}'"
                )
            except Exception as e:
                logger.warning(
                    f"Error deleting metadata schema for {collection_name}: {e}"
                )
            try:
                self._delete_from_system_table(
                    LANCEDB_DOCUMENT_INFO_TABLE,
                    f"collection_name = '{collection_name}'"
                )
            except Exception as e:
                logger.warning(
                    f"Error deleting document info for {collection_name}: {e}"
                )

        return {
            "message": "Collection deletion process completed.",
            "successful": deleted_collections,
            "failed": failed_collections,
            "total_success": len(deleted_collections),
            "total_failed": len(failed_collections),
        }

    def _delete_from_system_table(self, table_name: str, where_clause: str) -> None:
        """Delete rows from a system table matching the where clause."""
        if not self._table_exists(table_name):
            return
        table = self._db.open_table(table_name)
        table.delete(where_clause)

    # -------------------------------------------------------------------------
    # VDBRag Document Management
    # -------------------------------------------------------------------------
    def get_documents(self, collection_name: str) -> list[dict[str, Any]]:
        """Get the list of unique documents in a collection."""
        if not self._table_exists(collection_name):
            return []

        table = self._db.open_table(collection_name)
        metadata_schema = self.get_metadata_schema(collection_name)

        # Query all rows to get unique sources
        df = table.to_pandas()
        if df.empty:
            return []

        # Parse source field and get unique documents
        documents_list = []
        seen_sources = set()

        for _, row in df.iterrows():
            source_str = row.get("source", "")
            try:
                source = json.loads(source_str) if source_str else {}
            except (json.JSONDecodeError, TypeError):
                source = {"source_name": source_str}

            source_name = source.get("source_name", source_str)
            doc_name = os.path.basename(source_name) if source_name else ""

            if doc_name and doc_name not in seen_sources:
                seen_sources.add(doc_name)

                # Parse content metadata
                content_metadata_str = row.get("content_metadata", "{}")
                try:
                    content_metadata = json.loads(content_metadata_str)
                except (json.JSONDecodeError, TypeError):
                    content_metadata = {}

                # Extract metadata based on schema
                metadata_dict = {}
                for metadata_item in metadata_schema:
                    meta_name = metadata_item.get("name")
                    metadata_dict[meta_name] = content_metadata.get(meta_name)

                documents_list.append({
                    "document_name": doc_name,
                    "metadata": metadata_dict,
                    "document_info": self.get_document_info(
                        info_type="document",
                        collection_name=collection_name,
                        document_name=doc_name,
                    ),
                })

        return documents_list

    def delete_documents(
        self,
        collection_name: str,
        source_values: list[str],
        result_dict: dict[str, list[str]] | None = None,
    ) -> bool:
        """Delete documents from a collection by source values."""
        if result_dict is not None:
            result_dict["deleted"] = []
            result_dict["not_found"] = []

        if not self._table_exists(collection_name):
            if result_dict is not None:
                result_dict["not_found"] = [
                    os.path.basename(s) for s in source_values
                ]
            return False

        table = self._db.open_table(collection_name)

        for source_value in source_values:
            doc_name = os.path.basename(source_value)
            try:
                # Delete by matching source_name in the JSON source field
                where_clause = f"source LIKE '%{source_value}%'"
                initial_count = table.count_rows()
                table.delete(where_clause)
                final_count = table.count_rows()
                deleted_count = initial_count - final_count

                if result_dict is not None:
                    if deleted_count > 0:
                        result_dict["deleted"].append(doc_name)
                    else:
                        result_dict["not_found"].append(doc_name)

                # Also delete document info
                self._delete_from_system_table(
                    LANCEDB_DOCUMENT_INFO_TABLE,
                    f"info_type = 'document' AND collection_name = '{collection_name}' "
                    f"AND document_name = '{doc_name}'"
                )
            except Exception as e:
                logger.warning(f"Failed to delete document {source_value}: {e}")
                if result_dict is not None:
                    result_dict["not_found"].append(doc_name)

        return True

    # -------------------------------------------------------------------------
    # Metadata Schema Management
    # -------------------------------------------------------------------------
    def create_metadata_schema_collection(self) -> None:
        """Create the metadata schema system table."""
        if self._metadata_schema_table_initialized:
            return

        if not self._table_exists(LANCEDB_METADATA_SCHEMA_TABLE):
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("collection_name", pa.string()),
                pa.field("metadata_schema", pa.string()),
            ])
            self._db.create_table(LANCEDB_METADATA_SCHEMA_TABLE, schema=schema)
            logger.info(f"Created metadata schema table: {LANCEDB_METADATA_SCHEMA_TABLE}")

        self._metadata_schema_table_initialized = True

    def add_metadata_schema(
        self,
        collection_name: str,
        metadata_schema: list[dict[str, Any]],
    ) -> None:
        """Add metadata schema for a collection."""
        self.create_metadata_schema_collection()

        table = self._db.open_table(LANCEDB_METADATA_SCHEMA_TABLE)

        # Delete existing schema for this collection
        try:
            table.delete(f"collection_name = '{collection_name}'")
        except Exception:
            pass

        # Add new schema
        data = [{
            "id": f"schema_{collection_name}_{time.time_ns()}",
            "collection_name": collection_name,
            "metadata_schema": json.dumps(metadata_schema),
        }]
        table.add(data)
        logger.info(f"Added metadata schema for collection {collection_name}")

    def get_metadata_schema(self, collection_name: str) -> list[dict[str, Any]]:
        """Get the metadata schema for a collection."""
        if not self._table_exists(LANCEDB_METADATA_SCHEMA_TABLE):
            return []

        table = self._db.open_table(LANCEDB_METADATA_SCHEMA_TABLE)
        try:
            results = table.search().where(
                f"collection_name = '{collection_name}'"
            ).limit(1).to_pandas()

            if not results.empty:
                schema_str = results.iloc[0]["metadata_schema"]
                return json.loads(schema_str)
        except Exception as e:
            logger.debug(f"Error getting metadata schema for {collection_name}: {e}")

        return []

    # -------------------------------------------------------------------------
    # Document Info Management
    # -------------------------------------------------------------------------
    def create_document_info_collection(self) -> None:
        """Create the document info system table."""
        if self._document_info_table_initialized:
            return

        if not self._table_exists(LANCEDB_DOCUMENT_INFO_TABLE):
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("info_type", pa.string()),
                pa.field("collection_name", pa.string()),
                pa.field("document_name", pa.string()),
                pa.field("info_value", pa.string()),
            ])
            self._db.create_table(LANCEDB_DOCUMENT_INFO_TABLE, schema=schema)
            logger.info(f"Created document info table: {LANCEDB_DOCUMENT_INFO_TABLE}")

        self._document_info_table_initialized = True

    def _get_aggregated_document_info(
        self,
        collection_name: str,
        info_value: dict[str, Any],
    ) -> dict[str, Any]:
        """Get aggregated document info for a collection."""
        try:
            existing_info = self.get_document_info(
                info_type="collection",
                collection_name=collection_name,
                document_name="NA",
            )
        except Exception as e:
            logger.error(f"Error getting aggregated document info: {e}")
            return info_value

        return _perform_document_info_aggregation(existing_info, info_value)

    def add_document_info(
        self,
        info_type: str,
        collection_name: str,
        document_name: str,
        info_value: dict[str, Any],
    ) -> None:
        """Add document info to the system table."""
        self.create_document_info_collection()

        # Aggregate if this is collection-level info
        if info_type == "collection":
            info_value = self._get_aggregated_document_info(
                collection_name=collection_name,
                info_value=info_value,
            )

        table = self._db.open_table(LANCEDB_DOCUMENT_INFO_TABLE)

        # Delete existing entry
        try:
            table.delete(
                f"info_type = '{info_type}' AND collection_name = '{collection_name}' "
                f"AND document_name = '{document_name}'"
            )
        except Exception:
            pass

        # Add new entry
        data = [{
            "id": f"info_{info_type}_{collection_name}_{document_name}_{time.time_ns()}",
            "info_type": info_type,
            "collection_name": collection_name,
            "document_name": document_name,
            "info_value": json.dumps(info_value),
        }]
        table.add(data)
        logger.debug(
            f"Added document info: {info_type}, {collection_name}, {document_name}"
        )

    def get_document_info(
        self,
        info_type: str,
        collection_name: str,
        document_name: str,
    ) -> dict[str, Any]:
        """Get document info from the system table."""
        if not self._table_exists(LANCEDB_DOCUMENT_INFO_TABLE):
            return {}

        table = self._db.open_table(LANCEDB_DOCUMENT_INFO_TABLE)
        try:
            results = table.search().where(
                f"info_type = '{info_type}' AND collection_name = '{collection_name}' "
                f"AND document_name = '{document_name}'"
            ).limit(1).to_pandas()

            if not results.empty:
                info_str = results.iloc[0]["info_value"]
                return json.loads(info_str)
        except Exception as e:
            logger.debug(
                f"No document info found for {info_type}, {collection_name}, "
                f"{document_name}: {e}"
            )

        return {}

    def get_catalog_metadata(self, collection_name: str) -> dict[str, Any]:
        """Get catalog metadata for a collection."""
        return self.get_document_info(
            info_type="catalog",
            collection_name=collection_name,
            document_name="NA",
        )

    def update_catalog_metadata(
        self,
        collection_name: str,
        updates: dict[str, Any],
    ) -> None:
        """Update catalog metadata for a collection."""
        existing = self.get_catalog_metadata(collection_name)
        merged = {**existing, **updates}
        merged["last_updated"] = _get_current_timestamp()

        self.add_document_info(
            info_type="catalog",
            collection_name=collection_name,
            document_name="NA",
            info_value=merged,
        )

    def get_document_catalog_metadata(
        self,
        collection_name: str,
        document_name: str,
    ) -> dict[str, Any]:
        """Get catalog metadata for a document."""
        doc_info = self.get_document_info(
            info_type="document",
            collection_name=collection_name,
            document_name=document_name,
        )
        return {
            "description": doc_info.get("description", ""),
            "tags": doc_info.get("tags", []),
        }

    def update_document_catalog_metadata(
        self,
        collection_name: str,
        document_name: str,
        updates: dict[str, Any],
    ) -> None:
        """Update catalog metadata for a document."""
        existing = self.get_document_info(
            info_type="document",
            collection_name=collection_name,
            document_name=document_name,
        )

        for key in ["description", "tags"]:
            if key in updates:
                existing[key] = updates[key]

        self.add_document_info(
            info_type="document",
            collection_name=collection_name,
            document_name=document_name,
            info_value=existing,
        )

    # -------------------------------------------------------------------------
    # Retrieval Operations
    # -------------------------------------------------------------------------
    def retrieval_langchain(
        self,
        query: str,
        collection_name: str,
        vectorstore=None,
        top_k: int = 10,
        filter_expr: str = "",
        otel_ctx: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve documents from a collection using LangChain."""
        if vectorstore is None:
            vectorstore = self.get_langchain_vectorstore(collection_name)

        token = otel_context.attach(otel_ctx) if otel_ctx is not None else None

        try:
            start_time = time.time()

            retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

            retriever_lambda = RunnableLambda(
                lambda x: retriever.invoke(x)
            )
            retriever_chain = {"context": retriever_lambda} | RunnableAssign(
                {"context": lambda input: input["context"]}
            )
            retriever_docs = retriever_chain.invoke(
                query, config={"run_name": "retriever"}
            )
            docs = retriever_docs.get("context", [])

            end_time = time.time()
            latency = end_time - start_time
            logger.info(f"LanceDB Retrieval latency: {latency:.4f} seconds")

            return self._add_collection_name_to_retreived_docs(docs, collection_name)
        except (requests.exceptions.ConnectionError, ConnectionError, OSError) as e:
            embedding_url = (
                self._embedding_model._client.base_url
                if hasattr(self._embedding_model, "_client")
                else "configured endpoint"
            )
            error_msg = (
                f"Embedding NIM unavailable at {embedding_url}. "
                "Please verify the service is running and accessible."
            )
            logger.error(f"Connection error in retrieval_langchain: {e}")
            raise APIError(error_msg, ErrorCodeMapping.SERVICE_UNAVAILABLE) from e
        finally:
            if token is not None:
                otel_context.detach(token)

    def get_langchain_vectorstore(self, collection_name: str):
        """Get the LangChain vectorstore for a collection."""
        try:
            from langchain_community.vectorstores import LanceDB as LangchainLanceDB
        except ImportError:
            try:
                from langchain_lancedb import LanceDB as LangchainLanceDB
            except ImportError as e:
                raise ImportError(
                    "langchain-lancedb or langchain-community with lancedb support "
                    "is required. Install with: pip install langchain-lancedb"
                ) from e

        if not self._table_exists(collection_name):
            raise ValueError(f"Table {collection_name} does not exist in LanceDB")

        table = self._db.open_table(collection_name)

        vectorstore = LangchainLanceDB(
            connection=table,
            embedding=self._embedding_model,
            text_key="text",
            vector_key="vector",
        )

        return vectorstore

    @staticmethod
    def _add_collection_name_to_retreived_docs(
        docs: list[Document],
        collection_name: str,
    ) -> list[Document]:
        """Add the collection name to retrieved documents."""
        for doc in docs:
            # Parse source and content_metadata from JSON strings if needed
            source = doc.metadata.get("source", "")
            if isinstance(source, str) and source.startswith("{"):
                try:
                    doc.metadata["source"] = json.loads(source)
                except json.JSONDecodeError:
                    pass

            content_metadata = doc.metadata.get("content_metadata", "")
            if isinstance(content_metadata, str) and content_metadata.startswith("{"):
                try:
                    doc.metadata["content_metadata"] = json.loads(content_metadata)
                except json.JSONDecodeError:
                    pass

            doc.metadata["collection_name"] = collection_name

        return docs

