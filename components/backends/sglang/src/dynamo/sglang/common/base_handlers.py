# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any, Optional

import sglang as sgl
from sglang.srt.server_args import ServerArgs


class BaseWorkerHandler(ABC):
    """
    Abstract base class for sglang request handlers. We use this to implement native sglang endpoints for
    workers
    """

    @abstractmethod
    def __init__(
        self,
        engine: sgl.Engine,
        server_args: ServerArgs,
        component,
        decode_client: Optional[Any] = None,
    ):
        self.engine = engine
        self.server_args = server_args
        self.component = component

    @abstractmethod
    async def generate(self, request):
        """Generate tokens from the engine"""
        ...

    async def flush_cache(self, request: dict):
        """Flush KV cache for each worker"""
        _ = request
        await self.engine.tokenizer_manager.flush_cache()
        yield True

    async def start_expert_distribution_record(self, request: dict):
        """
        Start recording expert distribution.
        """
        _ = request
        await self.engine.tokenizer_manager.start_expert_distribution_record()
        yield True

    async def stop_expert_distribution_record(self, request: dict):
        """
        Stop recording expert distribution.
        """
        _ = request
        await self.engine.tokenizer_manager.stop_expert_distribution_record()
        yield True

    async def dump_expert_distribution_record(self, request: dict):
        """
        Dumps the expert distribution record to the directory specified in the environment variable `SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR`.
        """
        _ = request
        await self.engine.tokenizer_manager.dump_expert_distribution_record()
        yield True

    async def get_model_info(self, request: dict):
        """Get model information including configuration and capabilities."""
        _ = request
        tokenizer = self.engine.tokenizer_manager.tokenizer
        model_info = {
            "model_path": self.server_args.model_path,
            "served_model_name": getattr(self.server_args, 'served_model_name', self.server_args.model_path),
            "vocab_size": getattr(tokenizer, 'vocab_size', None),
        }
        yield model_info

    async def get_server_info(self, request: dict):
        """Get server configuration and runtime information."""
        _ = request
        server_info = {
            "host": getattr(self.server_args, 'host', '0.0.0.0'),
            "port": getattr(self.server_args, 'port', 30000),
            "model_path": self.server_args.model_path,
            "version": getattr(sgl, '__version__', 'unknown'),
        }
        yield server_info

    async def health(self, request: dict):
        """Basic health check endpoint."""
        _ = request
        yield {"status": "healthy"}

    async def health_generate(self, request: dict):
        """Health check with actual generation test."""
        test_prompt = request.get("prompt", "Hello")
        max_tokens = request.get("max_tokens", 5)
        result = await self.engine.async_generate(
            input_ids=test_prompt,
            sampling_params={"max_new_tokens": max_tokens, "temperature": 0.0},
            stream=False
        )
        async for output in result:
            data = output.data() if hasattr(output, 'data') else output
            yield data
            break

    async def update_weights_from_disk(self, request: dict):
        """Update model weights from disk."""
        model_path = request.get("model_path")
        await self.engine.update_weights(model_path)
        yield True

    async def encode(self, request: dict):
        """Generate embeddings using an embedding model."""
        input_text = request.get("input")
        result = await self.engine.async_generate(
            input_ids=input_text,
            sampling_params={"max_new_tokens": 0},
            stream=False,
        )
        async for output in result:
            data = output.data() if hasattr(output, 'data') else output
            yield data
            break

    async def rerank(self, request: dict):
        """Rerank documents using a cross-encoder rerank model."""
        result = await self.engine.async_generate(
            input_ids=request,
            sampling_params={"max_new_tokens": 1, "temperature": 0.0},
            stream=False,
        )
        async for output in result:
            data = output.data() if hasattr(output, 'data') else output
            yield data
            break

    async def classify(self, request: dict):
        """Classify text using a reward/classification model."""
        input_text = request.get("input")
        result = await self.engine.async_generate(
            input_ids=input_text,
            sampling_params={"max_new_tokens": 1, "temperature": 0.0},
            stream=False,
        )
        async for output in result:
            data = output.data() if hasattr(output, 'data') else output
            yield data
            break
