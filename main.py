import asyncio
import logging
import os
from collections.abc import AsyncIterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from threading import Thread
from typing import Dict, List, Literal, cast
from uuid import uuid4

import torch
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    response: AsyncIterable[str] | None = None


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

        # set up model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        eos_token_id = self.model.generation_config.eos_token_id if self.model.generation_config else []
        if not isinstance(eos_token_id, list):
            eos_token_id = [eos_token_id]

        self.eos_token_ids = set(eos_token_id)  # convert to set for faster lookup

        # queueing and record keeping
        self.batch_size = batch_size
        self.request_queue = asyncio.Queue[PendingInferenceRequest](maxsize=self.batch_size)
        self._inference_worker_task = asyncio.create_task(self._inference_worker())
        self._inference_records: Dict[str, InferenceRecord] = {}

        # storing the event loop for async operations from threads
        self._event_loop = asyncio.get_event_loop()

    async def generate(self, req: InferenceRequest) -> AsyncIterable[str] | str | None:
        pending_req = PendingInferenceRequest(request=req)
        inference_record = InferenceRecord(request=req)
        self._inference_records[pending_req.id] = inference_record

        try:
            self.request_queue.put_nowait(pending_req)
        except asyncio.QueueFull:
            logger.error("Request queue is full, cannot process the request at this time.")
            self._inference_records.pop(pending_req.id, None)
            return None

        # await response to be available
        await inference_record.ready_flag.wait()

        # handle failure (None) or streaming response
        if inference_record.response is None or req.stream:
            return inference_record.response

        # non-streaming collect the chunks and return it as a single string
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

    @staticmethod
    async def _create_stream(queue: asyncio.Queue[str | None]) -> AsyncIterable[str]:
        """
        Utility function to create an async iterable stream of tokens
        from a asyncio queue.
        """

        while True:
            token = await queue.get()
            if token is None:
                break

            yield token

    def _generate_stream(self, requests: List[PendingInferenceRequest]) -> List[AsyncIterable]:
        max_new_tokens = max(pr.request.max_output_tokens for pr in requests)
        inputs = self.tokenizer(
            [pr.request.text for pr in requests],
            padding=True,  # pad all to same length
            return_tensors="pt",
        ).to(self.model.device)

        token_queues: List[asyncio.Queue[str | None]] = [asyncio.Queue() for _ in requests]
        completed_requests = [False for _ in requests]

        def auto_regressive_loop():
            input_ids = inputs["input_ids"]  # [batch_size, max_seq_length]
            # attention_mask = inputs["attention_mask"]  # same shape as input_ids

            # we don't need gradients for inference
            with torch.no_grad():
                for i in range(max_new_tokens):
                    # one forward pass
                    # passing attention_mask caused it to get stuck in a loop idk why
                    outputs = self.model(input_ids)

                    # outputs.logits has shape [batch_size, max_seq_length, vocab_size]
                    # for each request, we only need the logits for the latest generation,
                    generated_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]

                    # greedy sampling (getting the token with the highest probability)
                    generated_tokens = torch.argmax(generated_token_logits, dim=-1).reshape(-1, 1)  # [batch_size, 1]

                    # decode the token
                    for req_idx, token_id in enumerate(generated_tokens):
                        # skip if this request has already completed
                        if completed_requests[req_idx]:
                            continue

                        token_id = token_id.squeeze().item()  # convert to python int
                        if token_id in self.eos_token_ids:
                            # signal end of generation
                            asyncio.run_coroutine_threadsafe(token_queues[req_idx].put(None), self._event_loop)
                            completed_requests[req_idx] = True
                            logger.info(f"Request {requests[req_idx].id} completed at step {i + 1}.")
                            continue

                        decoded_token = self.tokenizer.decode(token_id, skip_special_tokens=False)
                        asyncio.run_coroutine_threadsafe(
                            token_queues[req_idx].put(decoded_token),
                            self._event_loop,
                        )

                    input_ids = torch.cat([input_ids, generated_tokens], dim=-1)  # [batch_size, max_seq_length + 1]

            # signal end of generation
            for req_idx, queue in enumerate(token_queues):
                if completed_requests[req_idx]:
                    continue

                asyncio.run_coroutine_threadsafe(queue.put(None), self._event_loop)
                completed_requests[req_idx] = True

        # start the auto-regressive loop in a separate thread
        logger.info("Starting auto-regressive loop for inference generation.")
        Thread(target=auto_regressive_loop, daemon=True).start()

        return [self._create_stream(queue) for queue in token_queues]

    def _generate_non_stream(self, req: InferenceRequest) -> str:
        inputs = self.tokenizer(req.text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=req.max_output_tokens)
        assert isinstance(outputs, torch.Tensor)

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
        model = HFModel("google/gemma-3-270m-it", batch_size=8)

        # run test inferences
        test_inputs = ["Hey, how are you?", "What's the capital of India?", "What's 2 + 2?"]
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
    if response is None:
        return JSONResponse(
            content={"error": "Request queue is full, please try again later."},
            status_code=503,
        )

    if isinstance(response, AsyncIterable):

        async def _stream_with_sse():
            async for chunk in response:
                yield f"data: {chunk}\n\n"

        return StreamingResponse(_stream_with_sse(), media_type="text/event-stream")

    return JSONResponse(content={"response": response})
