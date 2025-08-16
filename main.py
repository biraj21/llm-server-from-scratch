import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from threading import Thread
from typing import Dict, List, Literal, cast
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from torch import Tensor
from transformers import AsyncTextIteratorStreamer, AutoModelForCausalLM, AutoTokenizer

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceRequest(BaseModel):
    text: str
    max_output_tokens: int = 200
    stream: bool = False


@dataclass
class PendingInferenceRequest:
    request: InferenceRequest
    id: str = field(default_factory=lambda: uuid4().hex)


@dataclass
class InferenceRecord:
    request: InferenceRequest
    ready_flag: asyncio.Event = field(default_factory=asyncio.Event)
    response: AsyncTextIteratorStreamer | None = None


# ensure HF_TOKEN is set
if not os.getenv("HF_TOKEN"):
    raise ValueError("HF_TOKEN env var needs to be set")


class HFModel:
    def __init__(self, model_name: Literal["google/gemma-3-270m-it"], batch_size: int = 1):
        """
        Initializes the HFModel with the specified model name and batch size.
        Loads the tokenizer and model from Hugging Face.
        """

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.batch_size = batch_size

        self.request_queue = asyncio.Queue[PendingInferenceRequest](maxsize=self.batch_size)

        self._inference_worker_task = asyncio.create_task(self._inference_worker())

        self._inference_records: Dict[str, InferenceRecord] = {}

    async def generate(self, req: InferenceRequest) -> AsyncTextIteratorStreamer | str | None:
        pending_req = PendingInferenceRequest(request=req)
        inference_record = InferenceRecord(request=req)
        self._inference_records[pending_req.id] = inference_record

        try:
            self.request_queue.put_nowait(pending_req)
        except asyncio.QueueFull:
            logger.error("Request queue is full, cannot process the request at this time.")
            self._inference_records.pop(pending_req.id, None)
            return None

        # await inference generation
        await inference_record.ready_flag.wait()
        if inference_record.response is None or req.stream:
            return inference_record.response

        chunks = [chunk async for chunk in inference_record.response]
        return "".join(chunks)

    async def _collect_batch(self, batch_wait_ms: int = 100) -> List[PendingInferenceRequest]:
        batch: List[PendingInferenceRequest] = []
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
                batch.append(req)
            except asyncio.TimeoutError:
                break

        return batch

    async def _inference_worker(self):
        while True:
            num_items = 0
            try:
                batch = await self._collect_batch()
                num_items = len(batch)
                responses = self._generate_stream(batch)
                for req, resp in zip(batch, responses):
                    inference_record = self._inference_records.get(req.id)
                    if not inference_record:
                        logger.error(f"Record for request {req.id} not found.")
                        continue

                    inference_record.response = resp
                    inference_record.ready_flag.set()
            except Exception as e:
                logger.error(f"Error during inference: {e}")
            finally:
                for i in range(num_items):
                    self.request_queue.task_done()

    def _generate_stream(self, requests: List[PendingInferenceRequest]) -> List[AsyncTextIteratorStreamer]:
        if len(requests) != 1:
            raise NotImplementedError("AsyncTextIteratorStreamer only supports batch size 1 for now")

        max_new_tokens = max(pr.request.max_output_tokens for pr in requests)
        inputs = self.tokenizer(
            [pr.request.text for pr in requests],
            padding=True,  # Pad to same length
            return_tensors="pt",
        ).to(self.model.device)

        streamer = AsyncTextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        thread = Thread(
            target=self.model.generate,
            kwargs={
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "streamer": streamer,
                "max_new_tokens": max_new_tokens,
            },
        )
        thread.start()
        return [streamer]

    def _generate_non_stream(self, req: InferenceRequest) -> str:
        inputs = self.tokenizer(req.text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=req.max_output_tokens)
        assert isinstance(outputs, Tensor)

        logger.info(f"birajlog Input shape: {inputs['input_ids'].shape}")
        logger.info(f"birajlog Output shape: {outputs.shape}")

        # skip input tokens
        input_length = inputs["input_ids"].shape[1]
        outputs = outputs[0][input_length:]

        response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        assert isinstance(response, str)
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Load the ML model
        model = HFModel("google/gemma-3-270m-it")

        # run test inferences
        test_inputs = ["hey, how are you?", "what's the capital for india?", "what's 2 + 2?"]
        for i, text in enumerate(test_inputs):
            req = InferenceRequest(text=text, max_output_tokens=25, stream=False)
            logger.info(f"test input {i}: {req.text}")
            response = await model.generate(req)
            logger.info(f"response: {response}")
            logger.info("-" * 80)

        logger.info("💪 model loaded successfully")

        app.state.model = model
    except Exception as e:
        logger.error(f"error loading model:{e}")
        raise e

    yield


app = FastAPI(lifespan=lifespan)


@app.post("/generate")
async def run_inference(inference_req: InferenceRequest):
    model = cast(HFModel, app.state.model)  # for type completion
    response = await model.generate(inference_req)
    if isinstance(response, AsyncTextIteratorStreamer):

        async def _stream_with_sse():
            async for chunk in response:
                yield f"data: {chunk}\n\n"

        return StreamingResponse(_stream_with_sse(), media_type="text/event-stream")

    return response
