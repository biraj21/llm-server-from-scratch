import asyncio
from collections.abc import AsyncIterable
from threading import Thread
from typing import Callable, Literal

import torch
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.logger import setup_logger

from .base import HFModel, InferenceRequest, InferenceState


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    messages: list[Message] = []
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
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
        inference = await self._process_request(req)
        if inference is None:
            return None

        # return error if any
        if inference.error is not None:
            return inference.error

        # handle failure (None) or streaming response
        if inference.response is None or req.stream:
            return inference.response

        # non-streaming collect the chunks and return it as a single string
        chunks = [chunk async for chunk in inference.response]
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
            # wrapping in wait_for to avoid waiting forever in case of some bug in the code
            token = await asyncio.wait_for(queue.get(), timeout=10)
            if token is None:
                break

            yield token

    def _auto_regressive_loop(
        self,
        requests: list[InferenceRequest[LLMRequest]],
        token_queues: list[asyncio.Queue[str | None]],
        set_batch_complete: Callable[[], None],
    ):
        logger.info(f"Starting auto-regressive loop for inference generation with {len(requests)} requests.")
        try:
            max_new_tokens = max(pr.request.max_output_tokens for pr in requests)

            # prepare the inputs
            formatted_chats: list[str] = []
            for r in requests:
                r.state = InferenceState.PROCESSING
                formatted = self.tokenizer.apply_chat_template(
                    r.request.messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if not isinstance(formatted, str):
                    inference = self._inference_records.get(r.id)
                    if not inference:
                        logger.error(
                            f"_auto_regressive_loop(): apply_chat_template error: Record for request {r.id} not found."
                        )
                        continue

                    inference.error = ValueError("Formatted chat is not a string")

                formatted_chats.append(formatted)

            inputs = self.tokenizer(
                formatted_chats,
                padding=True,  # pad all to same length
                return_tensors="pt",
            ).to(self.model.device)

            temperatures = torch.tensor(
                [r.request.temperature for r in requests],
                device=self.model.device,
            )  # [batch_size]

            greedy_mask = temperatures == 0.0  # argmax
            non_greedy_mask = ~greedy_mask  # random sampling with scaled logits

            input_ids = inputs["input_ids"]  # [batch_size, max_seq_length]
            # attention_mask = inputs["attention_mask"]  # same shape as input_ids

            past_key_values: torch.Tensor | None = None

            # we don't need gradients for inference
            with torch.no_grad():
                for i in range(max_new_tokens):
                    # one forward pass
                    # passing attention_mask caused it to get stuck in a loop idk why
                    outputs = self.model(
                        input_ids=input_ids
                        if past_key_values is None
                        else input_ids[:, -1:],  # only last token after first iteration
                        past_key_values=past_key_values,
                    )

                    past_key_values = outputs.past_key_values

                    # outputs.logits has shape [batch_size, max_seq_length, vocab_size]
                    # for each request, we only need the logits for the latest generation,
                    generated_token_logits: torch.Tensor = outputs.logits[:, -1, :]  # [batch_size, vocab_size]

                    # handle zero temperature (greedy sampling)
                    argmaxed_tokens = sampled_tokens = None
                    if greedy_mask.any():
                        # greedy sampling - getting the token with the highest probability
                        argmaxed_tokens = torch.argmax(generated_token_logits[greedy_mask], dim=-1).view(-1)

                    # handle non-zero temperature (random sampling with scaled logits)
                    if non_greedy_mask.any():
                        sampling_logits = generated_token_logits[non_greedy_mask]  # [non-zero temp inputs, vocab_size]
                        sampling_temps = temperatures[non_greedy_mask].view(-1, 1)  # [non-zero temp inputs, 1]

                        # apply temperature scaling
                        scaled_logits = sampling_logits / sampling_temps  # [non-zero temp inputs, vocab_size]

                        # use logits to generate probabilities of each token across vocab
                        probs = torch.softmax(scaled_logits, dim=-1)  # [non-zero temp inputs, vocab_size]

                        # sample indices (tokens) using these probabilities
                        sampled_tokens = torch.multinomial(probs, num_samples=1).view(-1)  # [non-zero temp inputs]

                    # a tensor of shape [batch_size] that will combine both
                    # argmax (for zero temp) and sampled (for non-zero temp) tokens
                    generated_token_ids = torch.zeros(len(requests), dtype=torch.long, device=self.model.device)

                    if argmaxed_tokens is not None:
                        generated_token_ids[greedy_mask] = argmaxed_tokens

                    if sampled_tokens is not None:
                        generated_token_ids[non_greedy_mask] = sampled_tokens

                    # decode the token
                    for token_id, req, token_queue in zip(generated_token_ids, requests, token_queues):
                        # skip if this request has already completed
                        if req.state == InferenceState.PENDING:
                            logger.warning(f"Request {req.id} is still in PENDING state during generation.")
                            req.state = InferenceState.FAILED
                            asyncio.run_coroutine_threadsafe(token_queue.put(None), self._event_loop)
                            continue

                        if req.state != InferenceState.PROCESSING:
                            continue

                        token_id = token_id.squeeze().item()  # convert to python int
                        if token_id in self.eos_token_ids:
                            # signal end of generation
                            asyncio.run_coroutine_threadsafe(token_queue.put(None), self._event_loop)
                            req.state = InferenceState.COMPLETED
                            logger.info(f"Request {req.id} completed at step {i + 1}.")
                            continue

                        decoded_token = self.tokenizer.decode(token_id, skip_special_tokens=False)
                        asyncio.run_coroutine_threadsafe(
                            token_queue.put(decoded_token),
                            self._event_loop,
                        )

                    input_ids = torch.cat(
                        [input_ids, generated_token_ids.view(-1, 1)], dim=-1
                    )  # [batch_size, max_seq_length + 1]
        except Exception as e:
            logger.error(f"Error during auto-regressive loop: {e}")

            # signal end of generation
            for req, token_queue in zip(requests, token_queues):
                if req.state == InferenceState.PROCESSING:
                    asyncio.run_coroutine_threadsafe(token_queue.put(None), self._event_loop)
                    req.state = InferenceState.FAILED
        finally:
            # signal end of generation
            for req, token_queue in zip(requests, token_queues):
                if req.state == InferenceState.PROCESSING:
                    asyncio.run_coroutine_threadsafe(token_queue.put(None), self._event_loop)
                    req.state = InferenceState.FAILED

            logger.info("Auto-regressive loop completed for all requests. Ready to accept new batch.")
            set_batch_complete()

    def _generate_stream(
        self, requests: list[InferenceRequest[LLMRequest]], batch_complete_event: asyncio.Event
    ) -> list[AsyncIterable[str]]:
        token_queues: list[asyncio.Queue[str | None]] = [asyncio.Queue() for _ in requests]

        def set_batch_complete_event():
            async def _set_async():
                batch_complete_event.set()

            asyncio.run_coroutine_threadsafe(_set_async(), self._event_loop)

        # start the auto-regressive loop in a separate thread
        Thread(
            target=self._auto_regressive_loop,
            kwargs={
                "requests": requests,
                "token_queues": token_queues,
                "set_batch_complete": set_batch_complete_event,
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
