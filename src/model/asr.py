import asyncio
import threading
from typing import List, Literal

import torch
from pydantic import BaseModel
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from src.logger import setup_logger
from src.utils import bytes_to_tensor

from .base import HFModel, InferenceRecord, PendingInferenceRequest

logger = setup_logger(__name__)


class ASRRequest(BaseModel):
    audio: bytes
    language: Literal["en"] = "en"


class HFModelASR(HFModel[ASRRequest, str]):
    def __init__(self, model_id: Literal["openai/whisper-large-v3-turbo"], batch_size: int = 1):
        """
        Initializes an ASR model from HuggingFace for transcription.
        """

        super().__init__(model_id=model_id, batch_size=batch_size)

        # set up model
        torch_dtype = torch.float16
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="eager",
        )
        self.model.eval()
        logger.info(f"Using device: {self.model.device} and dtype: {self.model.dtype}")
        self.processor = AutoProcessor.from_pretrained(model_id)

        self._inference_worker_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self._inference_worker_thread.start()

    async def transcribe(self, req: ASRRequest) -> Exception | str | None:
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

        return inference_record.response

    def _inference_worker(self):
        while True:
            batch = []
            try:
                batch = asyncio.run_coroutine_threadsafe(
                    self._collect_batch(),
                    loop=self._event_loop,
                ).result()

                self._generate(batch)
            except Exception as e:
                logger.error(f"Error during inference: {e}")
                for req in batch:
                    inference_record = self._inference_records.get(req.id)
                    if not inference_record:
                        logger.error(f"Record for request {req.id} not found.")
                        continue

                    inference_record.error = e

                    async def set_ready_flag(record: InferenceRecord):
                        record.ready_flag.set()

                    asyncio.run_coroutine_threadsafe(
                        set_ready_flag(inference_record),
                        loop=self._event_loop,
                    )
            finally:
                for i in range(len(batch)):
                    self.request_queue.task_done()

    def _generate(self, batch: List[PendingInferenceRequest[ASRRequest]]):
        with torch.no_grad():
            with sdpa_kernel(SDPBackend.MATH):
                # process batch of audio
                batch_features = []
                for req in batch:
                    audio, sample_rate = bytes_to_tensor(req.request.audio)
                    logger.info(f"birajlog request audio: {audio.shape}, {sample_rate}")
                    features = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt").input_features
                    batch_features.append(features)

                # stack into batch tensor
                batch_input = torch.cat(batch_features, dim=0).to(self.model.device, dtype=self.model.dtype)

                # forward pass
                generated_ids = self.model.generate(
                    batch_input,
                    do_sample=False,
                    num_beams=1,
                    language="en",
                    task="transcribe",
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )

                # decode batch results
                batch_transcriptions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

            # set results in inference records
            for req, transcription in zip(batch, batch_transcriptions):
                inference_record = self._inference_records.get(req.id)
                if not inference_record:
                    logger.error(f"Record for request {req.id} not found.")
                    continue

                inference_record.response = transcription

                async def set_ready_flag(record: InferenceRecord):
                    record.ready_flag.set()

                asyncio.run_coroutine_threadsafe(
                    set_ready_flag(inference_record),
                    loop=self._event_loop,
                )
