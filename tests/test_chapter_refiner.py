import pytest
from unittest.mock import patch, MagicMock
import json
import os
from absrefined.refiner.chapter_refiner import ChapterRefiner


class TestChapterRefiner:
    """Tests for the ChapterRefiner class."""

    def test_init(self):
        """Test initialization of the ChapterRefiner."""
        refiner = ChapterRefiner("http://localhost:1234/v1")
        assert refiner.api_base == "http://localhost:1234/v1"
        assert refiner.model == "gpt-4o"
        assert refiner.window_size == 15
        assert refiner.verbose is False

        refiner = ChapterRefiner(
            "http://localhost:1234/v1/", model="gpt-4", window_size=30, verbose=True
        )
        assert refiner.api_base == "http://localhost:1234/v1"
        assert refiner.model == "gpt-4"
        assert refiner.window_size == 30
        assert refiner.verbose is True

    @patch("absrefined.refiner.chapter_refiner.requests.post")
    @patch("absrefined.refiner.chapter_refiner.ChapterRefiner.verify_api_key")
    def test_query_llm(
        self, mock_verify_api_key, mock_post, mock_llm_response, mock_abs_response
    ):
        """Test querying the LLM API."""
        # Mock the API key verification to return True
        mock_verify_api_key.return_value = True

        # Create a custom mock response that matches the expected format
        custom_response = {
            "id": mock_llm_response["id"],
            "object": mock_llm_response["object"],
            "created": mock_llm_response["created"],
            "model": mock_llm_response["model"],
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "291.18",  # Just return the number directly
                    },
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "usage": mock_llm_response["usage"],
        }

        # Setup the mock response
        mock_post.return_value = mock_abs_response(json_data=custom_response)

        # Create refiner
        refiner = ChapterRefiner("http://localhost:1234/v1")

        # Set the api_key directly
        refiner.api_key = "test-api-key"

        # Test query_llm
        system_prompt = "You are a helpful assistant."
        user_prompt = "Find the exact timestamp where Chapter 1 begins."
        response = refiner.query_llm(system_prompt, user_prompt)

        # Verify results
        assert response == "291.18"

        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:1234/v1/chat/completions"
        assert call_args[1]["headers"] == {
            "Content-Type": "application/json",
            "Authorization": "Bearer test-api-key",
        }
        request_json = call_args[1]["json"]
        assert request_json["model"] == "gpt-4.1"  # Check for the correct model
        assert request_json["messages"][0]["content"] == system_prompt
        assert request_json["messages"][1]["content"] == user_prompt
        assert (
            request_json["temperature"] < 1.0
        )  # Check that temperature is low for deterministic results

    @patch("absrefined.refiner.chapter_refiner.ChapterRefiner.query_llm")
    def test_detect_chapter_start(self, mock_query_llm):
        """Test detecting the precise start of a chapter."""
        # Setup the mock query response
        mock_query_llm.return_value = "291.18"

        # Create sample transcript
        transcript = [
            {
                "start": 285.24,
                "end": 287.7,
                "text": "It begins with a house.",
                "words": [
                    {"word": "It", "start": 285.24, "end": 285.58, "probability": 0.98},
                    {
                        "word": "begins",
                        "start": 285.58,
                        "end": 286.0,
                        "probability": 0.99,
                    },
                    {
                        "word": "with",
                        "start": 286.0,
                        "end": 286.98,
                        "probability": 0.97,
                    },
                    {"word": "a", "start": 286.98, "end": 287.14, "probability": 0.94},
                    {
                        "word": "house.",
                        "start": 287.14,
                        "end": 287.7,
                        "probability": 0.92,
                    },
                ],
            },
            {
                "start": 291.18,
                "end": 292.54,
                "text": "Chapter 1",
                "words": [
                    {
                        "word": "Chapter",
                        "start": 291.18,
                        "end": 291.86,
                        "probability": 0.73,
                    },
                    {"word": "1", "start": 291.86, "end": 292.54, "probability": 0.67},
                ],
            },
            {
                "start": 293.28,
                "end": 298.24,
                "text": "The house stood on a slight rise just on the edge of the village.",
                "words": [
                    {
                        "word": "The",
                        "start": 293.28,
                        "end": 293.96,
                        "probability": 0.57,
                    },
                    {
                        "word": "house",
                        "start": 293.96,
                        "end": 294.38,
                        "probability": 0.88,
                    },
                    # ... more words ...
                ],
            },
        ]

        # Create refiner with window_size
        refiner = ChapterRefiner("http://localhost:1234/v1", window_size=15)

        # Test detect_chapter_start with original timestamp
        orig_timestamp = 290.0
        result = refiner.detect_chapter_start(transcript, "Chapter 1", orig_timestamp)

        # Verify results
        assert result is not None
        assert result["timestamp"] == 291.18

        # Verify the LLM query was called
        assert mock_query_llm.called

        # Test with window size filtering
        # Create a refiner with a small window
        refiner_small_window = ChapterRefiner("http://localhost:1234/v1", window_size=1)

        # Reset the mock
        mock_query_llm.reset_mock()

        # Call with a timestamp far from any relevant segments
        far_timestamp = 350.0
        result_no_segments = refiner_small_window.detect_chapter_start(
            transcript, "Chapter 1", far_timestamp
        )

        # Should return None since no segments are in the window
        assert result_no_segments is None

        # Verify LLM was not called
        assert not mock_query_llm.called

    def test_query_llm_no_api_key(self, caplog):
        """Test querying LLM without an API key."""
        refiner = ChapterRefiner("http://localhost:1234/v1")
        # ... existing code ...

    # This test uses the mock_transcript_data fixture provided by conftest.py
    def test_query_llm(self, mock_transcript_data):
        """Test querying the LLM API successfully (simplified)."""

        # Target for the OpenAI class within the chapter_refiner module
        openai_client_target = "absrefined.refiner.chapter_refiner.OpenAI"

        with patch(openai_client_target) as mock_openai_class:
            # Mock the instance and its method
            mock_client_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[
                0
            ].message.content = "10.5"  # Mock LLM response content
            mock_client_instance.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_client_instance

            # Initialize ChapterRefiner (this should now use the mocked OpenAI class)
            refiner = ChapterRefiner(
                api_base="http://mock-api.com/v1", llm_api_key="test_key"
            )

            # Get an available chapter name and its segments from the fixture
            if not mock_transcript_data:
                pytest.skip(
                    "No transcript data found in chapter-segments, skipping test."
                )

            # Use the first available chapter name for the test
            chapter_name = next(iter(mock_transcript_data))
            transcript_segments = mock_transcript_data[chapter_name]
            orig_timestamp = 5.0  # Use a dummy timestamp for the test context

            # Call the method that uses the mocked client
            result = refiner.detect_chapter_start(
                transcript_segments, chapter_name, orig_timestamp
            )

            # Basic assertions to ensure the mocked path was taken
            assert result is not None, "Result should not be None if mock worked"
            assert result["timestamp"] == 10.5, "Timestamp should match mocked response"

            # Verify the mock was used as expected
            mock_openai_class.assert_called_once_with(api_key="test_key")
            mock_client_instance.chat.completions.create.assert_called_once()
            self.logger.info("test_query_llm passed with simplified mock.")

    # Test case for when the LLM response is just a number
    @patch("absrefined.refiner.chapter_refiner.OpenAI")
    def test_detect_chapter_start_llm_returns_number(
        self, mock_openai, mock_transcript_data
    ):
        """Test chapter detection when LLM returns only a number."""
        refiner = ChapterRefiner(
            api_base="http://mock-api.com/v1", llm_api_key="test_key"
        )
        # ... existing code ...
