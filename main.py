import asyncio
import logging
import os
from contextlib import asynccontextmanager
from threading import Thread

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AsyncTextIteratorStreamer, AutoModelForCausalLM, AutoTokenizer
from torch import Tensor

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceRequest(BaseModel):
    text: str
    max_output_tokens: int = 200
    stream: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    model_name = "google/gemma-3-270m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    async def generate(ireq: InferenceRequest):
        inputs = tokenizer(ireq.text, return_tensors="pt").to(model.device)

        if ireq.stream:
            text_streamer = AsyncTextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            thread = Thread(
                target=model.generate,
                kwargs={
                    "input_ids": inputs["input_ids"],
                    "streamer": text_streamer,
                    "max_new_tokens": ireq.max_output_tokens,
                },
            )
            thread.start()
            return text_streamer

        def _generate_non_stream():
            outputs = model.generate(**inputs, max_new_tokens=ireq.max_output_tokens)
            assert isinstance(outputs, Tensor)
            logger.info(f"birajlog Input shape: {inputs['input_ids'].shape}")
            logger.info(f"birajlog Output shape: {outputs.shape}")
            input_length = inputs["input_ids"].shape[1]
            outputs = outputs[0][input_length:]
            response = tokenizer.decode(outputs, skip_special_tokens=True)
            return response

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _generate_non_stream)

    app.state.generate = generate


    # run test inferences
    for text in ["hey, how are you?", "what's the capital for india?", "what's 2 + 2?"]:
        ireq = InferenceRequest(text=text, max_output_tokens=25, stream=False)
        logger.info(f"test input: {ireq.text}")
        response = await generate(ireq)
        logger.info(f"response: {response}")

    logger.info("💪 model loaded successfully")

    yield


app = FastAPI(lifespan=lifespan)

# ensure HF_TOKEN is set
if not os.getenv("HF_TOKEN"):
    raise ValueError("HF_TOKEN env var needs to be set")


@app.post("/generate")
async def run_inference(ireq: InferenceRequest):
    response = await app.state.generate(ireq)
    if isinstance(response, AsyncTextIteratorStreamer):
        return StreamingResponse(response)

    return response
