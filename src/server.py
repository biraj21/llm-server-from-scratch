import asyncio
from collections.abc import AsyncIterable
from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from src.logger import setup_logger
from src.model.asr import ASRRequest, HFModelASR
from src.model.llm import HFModelLLM, LLMRequest

logger = setup_logger(__name__)


async def _load_and_warm_up_llm():
    # load the LLM
    llm = HFModelLLM("google/gemma-3-270m-it", batch_size=8)

    # run test inferences
    test_inputs = [
        "Hey, how are you?",
        "What's the capital of India?",
        "What's 2 + 2?",
        "Write a haiku about AI.",
        "Tell me a joke.",
        "Explain quantum computing in 2 sentences.",
        "What is the meaning of life?",
        "Tell me about Batman",
    ]
    inferences_tasks = []
    for i, text in enumerate(test_inputs):
        req = LLMRequest(text=text, max_output_tokens=25, stream=False)
        task = asyncio.create_task(llm.generate(req))
        inferences_tasks.append(task)

    results = await asyncio.gather(*inferences_tasks)
    for i, (input_text, response) in enumerate(zip(test_inputs, results)):
        logger.info(f"test input {i}: {input_text}")
        logger.info(f"response: {response}")
        logger.info("-" * 80)

    logger.info("💪 LLM loaded successfully")


async def _load_and_warm_up_asr():
    # load the ASR model
    asr_model = HFModelASR("openai/whisper-large-v3-turbo", batch_size=4)

    with open("samples/jerry_maguire_love.wav", "rb") as f:
        audio_bytes = f.read()
        req = ASRRequest(audio=audio_bytes)
        transcription = await asr_model.transcribe(req)
        logger.info(f"test transcription: {transcription}")

    logger.info("💪 ASR model loaded successfully")

    return asr_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.llm = await _load_and_warm_up_llm()
        app.state.asr = await _load_and_warm_up_asr()
    except Exception as e:
        logger.error(f"error loading model:{e}")
        raise e

    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/generate")
async def generate_text(inference_req: LLMRequest):
    model = cast(HFModelLLM, app.state.llm)  # for type completion
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


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile):
    model = cast(HFModelASR, app.state.asr)  # for type completion
    audio_bytes = await file.read()
    req = ASRRequest(audio=audio_bytes)
    response = await model.transcribe(req)
    if response is None:
        return JSONResponse(
            content={"error": "Request queue is full, please try again later."},
            status_code=503,
        )

    return JSONResponse(content={"transcription": response})
