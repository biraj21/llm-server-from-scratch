import asyncio
from collections.abc import AsyncIterable
from threading import Thread
from typing import List, Literal

import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.logger import setup_logger

from .base import HFModel, InferenceRecord, PendingInferenceRequest


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    messages: List[Message] = []
    max_output_tokens: int = 200
    stream: bool = False


logger = setup_logger(__name__)


class HFModelLLM(HFModel[LLMRequest, AsyncIterable[str]]):
    def __init__(self, model_id: Literal["google/gemma-3-270m-it"], batch_size: int = 1):
        """
        Initializes an LLM from HuggingFace for text generation.
        """

        super().__init__(model_id=model_id, batch_size=batch_size)

        # set up model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        logger.info(f"Using device: {self.model.device} and dtype: {self.model.dtype}")

        eos_token_id = self.model.generation_config.eos_token_id if self.model.generation_config else []
        if not isinstance(eos_token_id, list):
            eos_token_id = [eos_token_id]

        self.eos_token_ids = set(eos_token_id)  # convert to set for faster lookup

        # batching, queueing and record keeping
        self._inference_worker_task = asyncio.create_task(self._inference_worker())

    async def generate(self, req: LLMRequest) -> Exception | AsyncIterable[str] | str | None:
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

        # return error if any
        if inference_record.error is not None:
            return inference_record.error

        # handle failure (None) or streaming response
        if inference_record.response is None or req.stream:
            return inference_record.response

        # non-streaming collect the chunks and return it as a single string
        chunks = [chunk async for chunk in inference_record.response]
        return "".join(chunks)

    async def _inference_worker(self):
        while True:
            num_items = 0
            try:
                batch = await self._collect_batch()
                num_items = len(batch)

                # Create an event for this batch
                batch_complete_event = asyncio.Event()

                responses = self._generate_stream(batch, batch_complete_event)
                for req, resp in zip(batch, responses):
                    inference_record = self._inference_records.get(req.id)
                    if not inference_record:
                        logger.error(f"Record for request {req.id} not found.")
                        continue

                    inference_record.response = resp
                    inference_record.ready_flag.set()

                await batch_complete_event.wait()
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

    def _auto_regressive_loop(
        self,
        requests: List[PendingInferenceRequest[LLMRequest]],
        token_queues: List[asyncio.Queue[str | None]],
        completed_requests: List[bool],
        batch_complete_event: asyncio.Event,
    ):
        logger.info(f"Starting auto-regressive loop for inference generation with {len(requests)} requests.")
        try:
            max_new_tokens = max(pr.request.max_output_tokens for pr in requests)

            # prepare the inputs
            formatted_chats: List[str] = []
            for r in requests:
                formatted = self.tokenizer.apply_chat_template(
                    r.request.messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if not isinstance(formatted, str):
                    # TODO: set error in inference record
                    raise ValueError("Formatted chat is not a string")

                formatted_chats.append(formatted)

            inputs = self.tokenizer(
                formatted_chats,
                padding=True,  # pad all to same length
                return_tensors="pt",
            ).to(self.model.device)

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
        except Exception as e:
            logger.error(f"Error during auto-regressive loop: {e}")
        finally:
            # signal end of generation
            for req_idx, queue in enumerate(token_queues):
                if completed_requests[req_idx]:
                    continue

                asyncio.run_coroutine_threadsafe(queue.put(None), self._event_loop)
                completed_requests[req_idx] = True

            logger.info("Auto-regressive loop completed for all requests. Ready to accept new batch.")

            async def set_batch_complete_event():
                batch_complete_event.set()

            asyncio.run_coroutine_threadsafe(set_batch_complete_event(), self._event_loop)

    def _generate_stream(
        self, requests: List[PendingInferenceRequest[LLMRequest]], batch_complete_event: asyncio.Event
    ) -> List[AsyncIterable]:
        token_queues: List[asyncio.Queue[str | None]] = [asyncio.Queue() for _ in requests]
        completed_requests = [False for _ in requests]

        # start the auto-regressive loop in a separate thread
        Thread(
            target=self._auto_regressive_loop,
            kwargs={
                "requests": requests,
                "token_queues": token_queues,
                "completed_requests": completed_requests,
                "batch_complete_event": batch_complete_event,
            },
            daemon=True,
        ).start()

        return [self._create_stream(queue) for queue in token_queues]

    def _generate_non_stream(self, req: LLMRequest) -> str:
        inputs = self.tokenizer.apply_chat_template(
            req.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(self.model.device)
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
