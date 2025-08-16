import asyncio
import logging
import os
from contextlib import asynccontextmanager
from threading import Thread
from typing import Literal, cast

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


# ensure HF_TOKEN is set
if not os.getenv("HF_TOKEN"):
    raise ValueError("HF_TOKEN env var needs to be set")


class HFModel:
    def __init__(self, model_name: Literal["google/gemma-3-270m"]):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    async def generate(self, req: InferenceRequest):
        if req.stream:
            return self._generate_stream(req)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._generate_non_stream, req)

    def _generate_stream(self, req: InferenceRequest):
        inputs = self.tokenizer(req.text, return_tensors="pt").to(self.model.device)
        text_streamer = AsyncTextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        thread = Thread(
            target=self.model.generate,
            kwargs={
                "input_ids": inputs["input_ids"],
                "streamer": text_streamer,
                "max_new_tokens": req.max_output_tokens,
            },
        )
        thread.start()
        return text_streamer

    def _generate_non_stream(self, req: InferenceRequest):
        inputs = self.tokenizer(req.text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=req.max_output_tokens)
        assert isinstance(outputs, Tensor)

        logger.info(f"birajlog Input shape: {inputs['input_ids'].shape}")
        logger.info(f"birajlog Output shape: {outputs.shape}")

        # skip input tokens
        input_length = inputs["input_ids"].shape[1]
        outputs = outputs[0][input_length:]

        response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Load the ML model
        model = HFModel("google/gemma-3-270m")

        # run test inferences
        for text in ["hey, how are you?", "what's the capital for india?", "what's 2 + 2?"]:
            req = InferenceRequest(text=text, max_output_tokens=25, stream=False)
            logger.info(f"test input: {req.text}")
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
