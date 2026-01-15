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
Base class for VDB operations that combines ingest and RAG functionality.
"""

from nvidia_rag.utils.vdb.vdb_base import VDBRag


class VDBRagIngest(VDBRag):
    """
    VDBRagIngest combines VDBRag with ingest-specific operations.
    This is a base class for VDB implementations that support both
    nv-ingest operations and RAG-specific functionality.
    """
    pass

