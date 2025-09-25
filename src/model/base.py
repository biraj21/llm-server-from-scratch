import asyncio
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Generic, TypeVar
from uuid import uuid4

from src.env import env
from src.logger import setup_logger

ReqT = TypeVar("ReqT")
"""
Type of request object.
"""

RespT = TypeVar("RespT")
"""
Type of response object.
"""


class InferenceState(Enum):
    PENDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class InferenceRequest(Generic[ReqT]):
    request: ReqT
    ready_flag: asyncio.Event = field(default_factory=asyncio.Event)
    id: str = field(default_factory=lambda: uuid4().hex)
    state: InferenceState = InferenceState.PENDING


logger = setup_logger(__name__)


@dataclass
class InferenceRecord(Generic[ReqT, RespT]):
    request: ReqT
    """
    The request object for this inference.
    """

    ready_flag: asyncio.Event = field(default_factory=asyncio.Event)
    """
    To be set when the response is ready to be consumed.
    """

    response: RespT | None = None
    """
    The response object for this inference.
    Must be consumed only after the ready_flag is set.
    """

    error: Exception | None = None
    """
    If an error occurred during processing, this will be set.
    """


class HFModel(Generic[ReqT, RespT]):
    def __init__(self, model_id: str, batch_size: int = 1):
        self.model_id = model_id

        # batching, queueing and record keeping
        self.batch_size = batch_size
        self.request_queue: asyncio.Queue[InferenceRequest[ReqT]] = asyncio.Queue(maxsize=self.batch_size)
        self._inference_records: dict[str, InferenceRecord[ReqT, RespT]] = {}

        # set the HF_TOKEN env variable for transformers to pick up
        os.environ["HF_TOKEN"] = env.HF_TOKEN

        # storing the event loop for async operations from threads
        self._event_loop = asyncio.get_event_loop()

    async def _collect_batch(self, batch_wait_ms: int = 100) -> list[InferenceRequest[ReqT]]:
        batch: list[InferenceRequest[ReqT]] = []
        batch_wait_sec = batch_wait_ms / 1000

        # wait to get one item from the queue first
        req = await self.request_queue.get()
        batch.append(req)

        # now try to collect more within the timeout
        end_time = asyncio.get_event_loop().time() + batch_wait_sec
        while len(batch) < self.batch_size:
            remaining_time = end_time - asyncio.get_event_loop().time()
            if remaining_time <= 0:
                break

            try:
                req = await asyncio.wait_for(self.request_queue.get(), timeout=remaining_time)
                if req.state == InferenceState.PENDING:
                    batch.append(req)
                else:
                    logger.error(f"_collect_batch(): unexpectedly got {req.state} request! skipping...")
                    continue

            except asyncio.TimeoutError:
                break

        return batch

    async def _process_request(self, req: ReqT) -> InferenceRecord | None:
        pending_req = InferenceRequest(request=req)
        inference_record = InferenceRecord(request=req)
        self._inference_records[pending_req.id] = inference_record

        try:
            await asyncio.wait_for(self.request_queue.put(pending_req), timeout=0.05)
        except asyncio.QueueFull:
            logger.error("Request queue is full, cannot process the request at this time.")
            self._inference_records.pop(pending_req.id, None)
            return None

        try:
            # await response to be available
            await asyncio.wait_for(inference_record.ready_flag.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            logger.error("Request timed out while waiting for response.")
            pending_req.state = InferenceState.CANCELLED
            return None

        return inference_record
