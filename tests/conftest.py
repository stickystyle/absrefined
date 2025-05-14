import json
import os
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


# Mock the mlx_whisper module for tests
class MockMlxWhisper:
    @staticmethod
    def transcribe(audio_file, word_timestamps=False):
        return {
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "This is test audio.",
                    "words": [
                        {"word": "This", "start": 0.0, "end": 1.0, "probability": 0.9},
                        {"word": "is", "start": 1.0, "end": 2.0, "probability": 0.9},
                        {"word": "test", "start": 2.0, "end": 3.0, "probability": 0.9},
                        {
                            "word": "audio.",
                            "start": 3.0,
                            "end": 5.0,
                            "probability": 0.9,
                        },
                    ],
                },
                {
                    "start": 5.0,
                    "end": 10.0,
                    "text": "Chapter 1",
                    "words": [
                        {
                            "word": "Chapter",
                            "start": 5.0,
                            "end": 8.0,
                            "probability": 0.9,
                        },
                        {"word": "1", "start": 8.0, "end": 10.0, "probability": 0.9},
                    ],
                },
            ]
        }


# Add mock modules to sys.modules
sys.modules["mlx_whisper"] = MockMlxWhisper


@pytest.fixture
def mock_abs_response():
    """Mock response from Audiobookshelf API."""

    def _mock_response(
        status_code=200,
        json_data=None,
        content=None,
        raise_for_status=None,
        headers=None,
    ):
        mock_resp = MagicMock()
        # Set status code
        mock_resp.status_code = status_code

        # Set raise_for_status method
        mock_resp.raise_for_status = MagicMock()
        if raise_for_status:
            mock_resp.raise_for_status.side_effect = raise_for_status

        # Set json method
        if json_data is not None:
            mock_resp.json = MagicMock(return_value=json_data)

        # Set content attribute and iter_content method
        if content is not None:
            mock_resp.content = content
            # Make iter_content more realistic for chunking if content is bytes
            if isinstance(content, bytes):
                mock_resp.iter_content = MagicMock(
                    return_value=iter(
                        [content[i : i + 8192] for i in range(0, len(content), 8192)]
                    )
                )
            else:  # For non-bytes content, original behavior
                mock_resp.iter_content = MagicMock(return_value=iter([content]))

        # Set text attribute for error messages
        mock_resp.text = "Mock response text"

        # Set headers
        mock_resp.headers = headers if headers is not None else {}

        return mock_resp

    return _mock_response


@pytest.fixture
def abs_auth_response():
    """Mock auth response from Audiobookshelf API."""
    return {
        "user": {
            "id": "user-123",
            "type": "admin",
            "username": "testuser",
            "token": "test-token-123",
        }
    }


@pytest.fixture
def abs_item_response():
    """Mock item response from AudioBookShelf API."""
    return {
        "id": "lib-item-123",
        "mediaType": "book",
        "media": {
            "metadata": {"title": "Test Book", "author": "Test Author"},
            "chapters": [
                {
                    "id": "chapter-1",
                    "start": 291.18,
                    "end": 1845.2,
                    "title": "Chapter 1",
                },
                {
                    "id": "chapter-2",
                    "start": 1845.2,
                    "end": 2257.88,
                    "title": "Chapter 2",
                },
                {
                    "id": "chapter-3",
                    "start": 2257.88,
                    "end": 3483.66,
                    "title": "Chapter 3",
                },
            ],
        },
    }


@pytest.fixture
def mock_audio_file(tmp_path):
    """Create a mock audio file."""
    audio_path = tmp_path / "test_audio.mp3"
    # Create an empty file for testing
    audio_path.touch()
    return str(audio_path)


@pytest.fixture
def mock_transcript_data():
    """Mock transcription data (previously from whisper-mlx, now hardcoded)."""
    # chapter_segments = {}
    # # Load real chapter segment data from the chapter-segments directory
    # segments_dir = Path("chapter-segments")
    # if segments_dir.exists():
    #     for segment_file in segments_dir.glob("*.jsonl"):
    #         chapter_name = segment_file.stem

    #         with open(segment_file, "r", encoding="utf-8") as f:
    #             # Load each line as JSON
    #             segments = [json.loads(line) for line in f]
    #             chapter_segments[chapter_name] = segments
    # return chapter_segments

    # Return a hardcoded dictionary with a sample chapter's transcript segments
    # This structure should match what the tests in test_chapter_refiner.py expect.
    # The segments should have 'start', 'end', 'text', and 'words' (list of dicts with 'word', 'start', 'end').
    return {
        "Chapter 1": [
            {
                "start": 0.0,
                "end": 5.2,
                "text": "It was a dark and stormy night.",
                "words": [
                    {"word": "It", "start": 0.0, "end": 0.5, "probability": 0.9},
                    {"word": "was", "start": 0.6, "end": 1.0, "probability": 0.9},
                    {"word": "a", "start": 1.1, "end": 1.3, "probability": 0.9},
                    {"word": "dark", "start": 1.4, "end": 2.0, "probability": 0.9},
                    {"word": "and", "start": 2.1, "end": 2.5, "probability": 0.9},
                    {"word": "stormy", "start": 2.6, "end": 3.5, "probability": 0.9},
                    {"word": "night.", "start": 3.6, "end": 5.2, "probability": 0.9},
                ],
            },
            {
                "start": 6.0,
                "end": 10.5,
                "text": "Suddenly, a shot rang out!",
                "words": [
                    {"word": "Suddenly,", "start": 6.0, "end": 7.0, "probability": 0.9},
                    {"word": "a", "start": 7.1, "end": 7.3, "probability": 0.9},
                    {"word": "shot", "start": 7.4, "end": 8.0, "probability": 0.9},
                    {"word": "rang", "start": 8.1, "end": 8.7, "probability": 0.9},
                    {"word": "out!", "start": 8.8, "end": 10.5, "probability": 0.9},
                ],
            },
        ],
        "Chapter 2": [  # Add another chapter for variety if needed by other tests
            {
                "start": 0.0,
                "end": 3.0,
                "text": "The next morning...",
                "words": [
                    {"word": "The", "start": 0.0, "end": 0.5, "probability": 0.9},
                    {"word": "next", "start": 0.6, "end": 1.2, "probability": 0.9},
                    {
                        "word": "morning...",
                        "start": 1.3,
                        "end": 3.0,
                        "probability": 0.9,
                    },
                ],
            }
        ],
    }


@pytest.fixture
def mock_valid_chapters():
    """Mock valid chapters data from valid_chapters.txt."""
    valid_chapters = {}

    # Load real chapter data from valid_chapters.txt if it exists
    valid_file = Path("valid_chapters.txt")
    if valid_file.exists():
        with open(valid_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    # Parse format like "04:51.180 - Chapter 1"
                    parts = line.split(" - ", 1)
                    if len(parts) == 2:
                        timestamp, chapter_name = parts
                        valid_chapters[chapter_name] = timestamp

    return valid_chapters


@pytest.fixture
def mock_llm_response():
    """Mock response from OpenAI-compatible LLM API."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "The precise chapter start is at 291.18 seconds. This timestamp represents the beginning of 'Chapter 1', which is clearly marked in the audio.",
                },
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 40, "completion_tokens": 20, "total_tokens": 60},
    }
