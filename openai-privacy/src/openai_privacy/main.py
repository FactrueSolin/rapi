import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import torch
import uvicorn
from openai_privacy.opf import OPF, RedactionResult
from pydantic import BaseModel, Field

from fastapi import FastAPI
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class RedactRequest(BaseModel):
    text: str = Field(
        ...,
        description="Input text to analyze for PII",
        examples=["John Smith lives at 123 Main St and his email is john@example.com"],
    )


class ReplacementPair(BaseModel):
    original: str = Field(..., description="The original text that was detected")
    replacement: str = Field(..., description="The placeholder that replaces the original text")


class RedactResponse(BaseModel):
    pairs: list[ReplacementPair] = Field(
        ...,
        description="Array of original text and replacement placeholder pairs",
    )
    redacted_text: str = Field(
        ...,
        description="The full text with all PII replaced by placeholders",
    )


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the privacy model is loaded")
    device: str = Field(..., description="Device used for inference (cuda or cpu)")


opf_instance: OPF | None = None
model_device: str = "cpu"
executor: ThreadPoolExecutor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global opf_instance, model_device, executor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_device = device

    model_path = os.environ.get("OPF_CHECKPOINT")
    if model_path is None:
        logger.warning(
            "OPF_CHECKPOINT environment variable not set. "
            "Set it to a local model path to enable redaction. "
            "The model will be auto-downloaded from HuggingFace on first use if not set."
        )

    try:
        logger.info("Loading OPF model on device: %s", device)
        kwargs = {"device": device, "output_text_only": False}
        if model_path:
            kwargs["model"] = model_path
        opf_instance = OPF(**kwargs)
        logger.info("OPF model loaded successfully on %s", device)
    except Exception as e:
        logger.error("Failed to load OPF model: %s", e)
        opf_instance = None

    executor = ThreadPoolExecutor(max_workers=4)
    logger.info("Thread pool executor initialized with 4 workers")

    yield

    logger.info("Shutting down OPF service")
    if executor:
        executor.shutdown(wait=False)
    opf_instance = None


app = FastAPI(
    title="OpenAI Privacy Filter API",
    description="FastAPI service for PII detection and redaction using OpenAI's privacy-filter model",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the service is running and the model is loaded",
    tags=["system"],
)
async def health() -> HealthResponse:
    if opf_instance is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "device": model_device,
            },
        )
    return HealthResponse(status="healthy", model_loaded=True, device=model_device)


@app.post(
    "/redact",
    response_model=RedactResponse,
    summary="Redact PII from text",
    description="Analyze input text for personally identifiable information and return original-replacement pairs",
    tags=["privacy"],
)
async def redact(request: RedactRequest) -> RedactResponse:
    if opf_instance is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not loaded"},
        )

    loop = asyncio.get_event_loop()
    result: RedactionResult = await loop.run_in_executor(
        executor,
        opf_instance.redact,
        request.text,
    )

    pairs = [
        ReplacementPair(original=span.text, replacement=span.placeholder)
        for span in result.detected_spans
    ]

    return RedactResponse(pairs=pairs, redacted_text=result.redacted_text)


def main():
    uvicorn.run(
        "openai_privacy.main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    main()
