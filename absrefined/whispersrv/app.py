from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException
from typing import List, Optional, Dict, Any
import logging
import time
import sys
import os
import mlx_whisper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("whisper-server")


import tempfile
import os
from pydantic import BaseModel

app = FastAPI(
    title="Local Whisper API",
    description="A local implementation of OpenAI's audio transcription API",
)


class TranscriptionWords(BaseModel):
    end: float
    start: float
    word: str


class TranscriptionResponse(BaseModel):
    text: str
    task: str = "transcribe"
    language: str = None
    duration: float = None
    words: List[TranscriptionWords] = None
    segments: List[Dict[str, Any]] = None


@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    temperature: float = Form(0.0),
    timestamp_granularities: Optional[List[str]] = Form(None),
):
    logger.info(f"Received transcription request for file: {file.filename}")
    logger.debug(
        f"Parameters: language={language}, prompt={'present' if prompt else 'none'}, temperature={temperature}"
    )

    start_time = time.time()
    temp_path = None

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, suffix="." + file.filename.split(".")[-1]
        ) as temp:
            temp.write(await file.read())
            temp_path = temp.name

        logger.debug(f"Saved uploaded file to temporary path: {temp_path}")

        # Perform transcription
        logger.info("Starting transcription...")
        result = mlx_whisper.transcribe(
            temp_path,
            word_timestamps=True,
            path_or_hf_repo="mlx-community/whisper-turbo",
        )

        logger.debug(
            f"Initial transcription successful with keys: {result.keys() if isinstance(result, dict) else 'not a dict'}"
        )
        logger.info(
            f"Transcription completed in {time.time() - start_time:.2f} seconds"
        )

        logger.info(f"Result keys: {result.keys()}")

        segments = []
        words = []
        for segment in result["segments"]:
            out = {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
            }
            segments.append(out)
            if "words" in segment:
                words = segment["words"]

        # Prepare response
        response = {
            "text": result["text"] if isinstance(result, dict) else str(result),
            "task": "transcribe",
            "language": result.get("language", language)
            if isinstance(result, dict)
            else language,
            "duration": result.get("duration", 0) if isinstance(result, dict) else 0,
            "segments": segments,
            "words": words,
        }

        logger.debug(response)
        return response

    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}", exc_info=True)
        # Re-raise with more details for better client error messages
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            logger.debug(f"Removed temporary file: {temp_path}")


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting uvicorn server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
