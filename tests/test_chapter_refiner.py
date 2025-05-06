import pytest
from unittest.mock import patch, MagicMock
import json
import os
from absrefined.refiner.chapter_refiner import ChapterRefiner
from openai import APIError


# Mock config for the refiner
MOCK_REFINER_OPENAI_API_URL = (
    "http://localhost:7890/v1"  # Different from transcriber for clarity
)
MOCK_REFINER_OPENAI_API_KEY = "test_refiner_openai_api_key_789"
MOCK_REFINER_MODEL_NAME = "gpt-refiner-test"
MOCK_REFINER_CONFIG = {
    "refiner": {
        "openai_api_url": MOCK_REFINER_OPENAI_API_URL,
        "openai_api_key": MOCK_REFINER_OPENAI_API_KEY,
        "model_name": MOCK_REFINER_MODEL_NAME,
    },
    "logging": {"level": "DEBUG"},  # Basic logging config
}


class TestChapterRefiner:
    """Tests for the ChapterRefiner class."""

    def test_init(self):
        """Test initialization of the ChapterRefiner."""
        # Test with default config
        with patch(
            "absrefined.refiner.chapter_refiner.OpenAI"
        ) as mock_openai_constructor:
            refiner = ChapterRefiner(config=MOCK_REFINER_CONFIG)
            assert refiner.api_base_url == MOCK_REFINER_OPENAI_API_URL
            assert refiner.api_key == MOCK_REFINER_OPENAI_API_KEY
            assert refiner.default_model_name == MOCK_REFINER_MODEL_NAME
            # window_size and verbose are no longer attributes of ChapterRefiner init
            mock_openai_constructor.assert_called_once_with(
                api_key=MOCK_REFINER_OPENAI_API_KEY,
                base_url=MOCK_REFINER_OPENAI_API_URL,
            )

    @patch("absrefined.refiner.chapter_refiner.ChapterRefiner.query_llm")
    def test_detect_chapter_start(self, mock_query_llm_method):
        """Test detecting the precise start of a chapter."""
        # Setup the mock query_llm method on an instance
        # query_llm now returns a tuple (content, usage_data)
        # LLM should return a time relative to the chunk, and within search_window_for_test
        mock_llm_response_time = "6.5"
        mock_query_llm_method.return_value = (
            mock_llm_response_time,
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

        # Create sample transcript (assuming this structure is still valid for prompt construction)
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

        # Create refiner
        refiner = ChapterRefiner(config=MOCK_REFINER_CONFIG)

        # Test detect_chapter_start with original timestamp
        # search_window_seconds is now a required parameter for refine_chapter_start_time
        # Let's assume the transcript covers a 30-second window for this test
        search_window_for_test = 30.0
        # target_time_seconds is relative to the chunk start (0 for this example)
        target_time_within_chunk = (
            5.0  # Example: original chapter mark was 5s into this 30s chunk
        )

        # The method is now refine_chapter_start_time
        # It returns refined_time_offset_in_chunk (float) or (None, None)
        refined_time_offset, usage = refiner.refine_chapter_start_time(
            transcript_segments=transcript,
            chapter_title="Chapter 1",
            target_time_seconds=target_time_within_chunk,
            search_window_seconds=search_window_for_test,
        )

        # Verify results
        assert refined_time_offset is not None
        assert refined_time_offset == float(mock_llm_response_time)

        # Verify the LLM query was called (query_llm is a method of the instance)
        mock_query_llm_method.assert_called_once()
        # We can add more specific checks on args passed to mock_query_llm_method if needed

        # Test with window size filtering logic (now part of refine_chapter_start_time itself if transcript is empty)
        # The old test created a refiner with a small window.
        # Now, if transcript_segments is empty, refine_chapter_start_time returns None early.
        mock_query_llm_method.reset_mock()

        # Call with empty transcript segments
        result_no_segments, usage_no_segments = refiner.refine_chapter_start_time(
            transcript_segments=[],
            chapter_title="Chapter 1",
            target_time_seconds=target_time_within_chunk,
            search_window_seconds=search_window_for_test,
        )

        # Should return None since no segments are provided
        assert result_no_segments is None
        assert usage_no_segments is None

        # Verify LLM was not called
        assert not mock_query_llm_method.called

    def test_query_llm_no_api_key(self, caplog):
        """Test ChapterRefiner init without an API key in config."""
        bad_config = {
            "refiner": {
                "openai_api_url": MOCK_REFINER_OPENAI_API_URL,
                # "openai_api_key": MOCK_REFINER_OPENAI_API_KEY, # Key is missing
                "model_name": MOCK_REFINER_MODEL_NAME,
            }
        }
        with pytest.raises(KeyError) as excinfo:
            ChapterRefiner(config=bad_config)
        assert "Missing openai_api_key" in str(excinfo.value)

    @patch("absrefined.refiner.chapter_refiner.OpenAI")
    def test_refine_chapter_start_time_with_llm_mock(
        self, mock_openai_constructor, mock_transcript_data
    ):
        """Test refine_chapter_start_time with mocked OpenAI client making a successful call."""
        mock_llm_client_instance = MagicMock()
        mock_chat_completion_response = MagicMock()
        mock_chat_completion_response.choices = [MagicMock()]
        mock_chat_completion_response.choices[
            0
        ].message.content = "10.5"  # Mock LLM response content
        mock_chat_completion_response.usage.prompt_tokens = 100
        mock_chat_completion_response.usage.completion_tokens = 5
        mock_chat_completion_response.usage.total_tokens = 105
        mock_llm_client_instance.chat.completions.create.return_value = (
            mock_chat_completion_response
        )
        mock_openai_constructor.return_value = mock_llm_client_instance

        refiner = ChapterRefiner(config=MOCK_REFINER_CONFIG)

        if not mock_transcript_data:
            pytest.skip("No transcript data found in chapter-segments, skipping test.")

        chapter_name = next(iter(mock_transcript_data))
        transcript_segments = mock_transcript_data[chapter_name]
        target_time_seconds = 5.0
        search_window_seconds = 60.0  # Assume a 60s window for these segments

        refined_time, usage_data = refiner.refine_chapter_start_time(
            transcript_segments,
            chapter_name,
            target_time_seconds,
            search_window_seconds,
        )

        assert refined_time == 10.5
        assert usage_data["prompt_tokens"] == 100
        mock_openai_constructor.assert_called_once_with(
            api_key=MOCK_REFINER_CONFIG["refiner"]["openai_api_key"],
            base_url=MOCK_REFINER_CONFIG["refiner"]["openai_api_url"],
        )
        mock_llm_client_instance.chat.completions.create.assert_called_once()

    @patch("absrefined.refiner.chapter_refiner.OpenAI")
    def test_detect_chapter_start_llm_returns_number(
        self, mock_openai_constructor, mock_transcript_data
    ):
        """Test chapter detection when LLM returns only a number."""
        # Setup mock OpenAI client
        mock_llm_client_instance = MagicMock()
        mock_chat_completion_response = MagicMock()
        mock_chat_completion_response.choices = [MagicMock()]
        mock_chat_completion_response.choices[
            0
        ].message.content = " 12.345 "  # LLM returns number with spaces
        mock_chat_completion_response.usage.prompt_tokens = 10
        mock_chat_completion_response.usage.completion_tokens = 2
        mock_chat_completion_response.usage.total_tokens = 12
        mock_llm_client_instance.chat.completions.create.return_value = (
            mock_chat_completion_response
        )
        mock_openai_constructor.return_value = mock_llm_client_instance

        refiner = ChapterRefiner(config=MOCK_REFINER_CONFIG)

        if not mock_transcript_data:
            pytest.skip("No transcript data found, skipping test.")

        chapter_name = next(iter(mock_transcript_data))
        transcript_segments = mock_transcript_data[chapter_name]
        # Assume transcript segments are within a 0-30s window for this test
        target_time_s = 10.0
        search_window_s = 30.0

        refined_timestamp, usage = refiner.refine_chapter_start_time(
            transcript_segments,
            chapter_name,
            target_time_seconds=target_time_s,
            search_window_seconds=search_window_s,
        )

        assert refined_timestamp == 12.345
        assert usage["total_tokens"] == 12
        mock_llm_client_instance.chat.completions.create.assert_called_once()

    @patch("absrefined.refiner.chapter_refiner.OpenAI")
    def test_init_missing_api_url(self, mock_openai_constructor):
        """Test initialization fails when API URL is missing from config."""
        bad_config = {
            "refiner": {
                # "openai_api_url": MOCK_REFINER_OPENAI_API_URL,  # URL is missing
                "openai_api_key": MOCK_REFINER_OPENAI_API_KEY,
                "model_name": MOCK_REFINER_MODEL_NAME,
            }
        }
        with pytest.raises(KeyError) as excinfo:
            ChapterRefiner(config=bad_config)
        assert "Missing openai_api_url" in str(excinfo.value)

    @patch("absrefined.refiner.chapter_refiner.OpenAI")
    def test_refine_chapter_start_time_with_model_override(
        self, mock_openai_constructor, mock_transcript_data
    ):
        """Test refine_chapter_start_time with a model name override."""
        mock_llm_client_instance = MagicMock()
        mock_chat_completion_response = MagicMock()
        mock_chat_completion_response.choices = [MagicMock()]
        mock_chat_completion_response.choices[0].message.content = "15.75"
        mock_chat_completion_response.usage.prompt_tokens = 100
        mock_chat_completion_response.usage.completion_tokens = 5
        mock_chat_completion_response.usage.total_tokens = 105
        mock_llm_client_instance.chat.completions.create.return_value = (
            mock_chat_completion_response
        )
        mock_openai_constructor.return_value = mock_llm_client_instance

        refiner = ChapterRefiner(config=MOCK_REFINER_CONFIG)

        if not mock_transcript_data:
            pytest.skip("No transcript data found in chapter-segments, skipping test.")

        chapter_name = next(iter(mock_transcript_data))
        transcript_segments = mock_transcript_data[chapter_name]
        target_time_seconds = 5.0
        search_window_seconds = 60.0
        override_model = "gpt-4-turbo"

        refined_time, usage_data = refiner.refine_chapter_start_time(
            transcript_segments,
            chapter_name,
            target_time_seconds,
            search_window_seconds,
            model_name_override=override_model,
        )

        assert refined_time == 15.75
        assert usage_data["total_tokens"] == 105

        # Verify the override model was used
        create_call_kwargs = (
            mock_llm_client_instance.chat.completions.create.call_args.kwargs
        )
        assert create_call_kwargs["model"] == override_model

    @patch("absrefined.refiner.chapter_refiner.OpenAI")
    def test_llm_returns_non_numeric_response(
        self, mock_openai_constructor, mock_transcript_data
    ):
        """Test handling when LLM returns a non-numeric response."""
        mock_llm_client_instance = MagicMock()
        mock_chat_completion_response = MagicMock()
        mock_chat_completion_response.choices = [MagicMock()]
        mock_chat_completion_response.choices[
            0
        ].message.content = "I cannot determine the precise timestamp."
        mock_chat_completion_response.usage.prompt_tokens = 100
        mock_chat_completion_response.usage.completion_tokens = 8
        mock_chat_completion_response.usage.total_tokens = 108
        mock_llm_client_instance.chat.completions.create.return_value = (
            mock_chat_completion_response
        )
        mock_openai_constructor.return_value = mock_llm_client_instance

        refiner = ChapterRefiner(config=MOCK_REFINER_CONFIG)

        if not mock_transcript_data:
            pytest.skip("No transcript data found, skipping test.")

        chapter_name = next(iter(mock_transcript_data))
        transcript_segments = mock_transcript_data[chapter_name]
        target_time_seconds = 5.0
        search_window_seconds = 60.0

        refined_time, usage_data = refiner.refine_chapter_start_time(
            transcript_segments,
            chapter_name,
            target_time_seconds,
            search_window_seconds,
        )

        # Should return None when response can't be parsed as a float
        assert refined_time is None
        assert usage_data is None
        mock_llm_client_instance.chat.completions.create.assert_called_once()

    @patch("absrefined.refiner.chapter_refiner.OpenAI")
    def test_llm_returns_out_of_bounds_timestamp(
        self, mock_openai_constructor, mock_transcript_data
    ):
        """Test handling when LLM returns a timestamp outside the valid range."""
        mock_llm_client_instance = MagicMock()
        mock_chat_completion_response = MagicMock()
        mock_chat_completion_response.choices = [MagicMock()]

        # Return timestamp outside valid chunk window (negative)
        mock_chat_completion_response.choices[0].message.content = "-5.75"
        mock_chat_completion_response.usage.prompt_tokens = 100
        mock_chat_completion_response.usage.completion_tokens = 5
        mock_chat_completion_response.usage.total_tokens = 105
        mock_llm_client_instance.chat.completions.create.return_value = (
            mock_chat_completion_response
        )
        mock_openai_constructor.return_value = mock_llm_client_instance

        refiner = ChapterRefiner(config=MOCK_REFINER_CONFIG)

        if not mock_transcript_data:
            pytest.skip("No transcript data found, skipping test.")

        chapter_name = next(iter(mock_transcript_data))
        transcript_segments = mock_transcript_data[chapter_name]
        target_time_seconds = 5.0
        search_window_seconds = 20.0  # 20-second window

        # Test with negative timestamp (beyond small tolerance)
        refined_time, usage_data = refiner.refine_chapter_start_time(
            transcript_segments,
            chapter_name,
            target_time_seconds,
            search_window_seconds,
        )
        assert refined_time is None
        assert usage_data is None

        # Test with timestamp beyond window (too large)
        mock_chat_completion_response.choices[
            0
        ].message.content = "30.75"  # Beyond 20-second window + tolerance
        refined_time, usage_data = refiner.refine_chapter_start_time(
            transcript_segments,
            chapter_name,
            target_time_seconds,
            search_window_seconds,
        )
        assert refined_time is None
        assert usage_data is None

        # Test with timestamp at boundary (within tolerance)
        mock_chat_completion_response.choices[
            0
        ].message.content = "20.3"  # Just within tolerance of 0.5s
        refined_time, usage_data = refiner.refine_chapter_start_time(
            transcript_segments,
            chapter_name,
            target_time_seconds,
            search_window_seconds,
        )
        assert refined_time is not None
        assert refined_time == 20.3

        # Test with slightly negative timestamp (within tolerance)
        mock_chat_completion_response.choices[
            0
        ].message.content = "-0.3"  # Within tolerance of -0.5s
        refined_time, usage_data = refiner.refine_chapter_start_time(
            transcript_segments,
            chapter_name,
            target_time_seconds,
            search_window_seconds,
        )
        assert refined_time is not None
        assert refined_time == 0.0  # Should be clamped to 0

    @patch("absrefined.refiner.chapter_refiner.OpenAI")
    def test_query_llm_direct(self, mock_openai_constructor):
        """Test query_llm method directly."""
        mock_llm_client_instance = MagicMock()
        mock_chat_completion_response = MagicMock()
        mock_chat_completion_response.choices = [MagicMock()]
        mock_chat_completion_response.choices[0].message.content = "Test response"
        mock_chat_completion_response.usage.prompt_tokens = 50
        mock_chat_completion_response.usage.completion_tokens = 2
        mock_chat_completion_response.usage.total_tokens = 52
        mock_llm_client_instance.chat.completions.create.return_value = (
            mock_chat_completion_response
        )
        mock_openai_constructor.return_value = mock_llm_client_instance

        refiner = ChapterRefiner(config=MOCK_REFINER_CONFIG)

        # Test query_llm directly
        system_prompt = "You are a helpful assistant."
        user_prompt = "Tell me something."
        model_name = "test-model"
        max_tokens = 30

        content, usage = refiner.query_llm(
            system_prompt, user_prompt, model_name, max_tokens
        )

        assert content == "Test response"
        assert usage["prompt_tokens"] == 50
        assert usage["completion_tokens"] == 2
        assert usage["total_tokens"] == 52

        # Verify parameters sent to create
        create_call_kwargs = (
            mock_llm_client_instance.chat.completions.create.call_args.kwargs
        )
        assert create_call_kwargs["model"] == model_name
        assert create_call_kwargs["max_tokens"] == max_tokens
        assert create_call_kwargs["temperature"] == 0.1
        assert len(create_call_kwargs["messages"]) == 2
        assert create_call_kwargs["messages"][0]["role"] == "system"
        assert create_call_kwargs["messages"][0]["content"] == system_prompt
        assert create_call_kwargs["messages"][1]["role"] == "user"
        assert create_call_kwargs["messages"][1]["content"] == user_prompt

    @patch("absrefined.refiner.chapter_refiner.OpenAI")
    def test_llm_response_missing_choices(self, mock_openai_constructor):
        """Test handling when LLM response is missing choices."""
        mock_llm_client_instance = MagicMock()
        mock_chat_completion_response = MagicMock()
        # Missing choices
        mock_chat_completion_response.choices = []
        mock_chat_completion_response.usage.prompt_tokens = 100
        mock_chat_completion_response.usage.completion_tokens = 0
        mock_chat_completion_response.usage.total_tokens = 100
        mock_llm_client_instance.chat.completions.create.return_value = (
            mock_chat_completion_response
        )
        mock_openai_constructor.return_value = mock_llm_client_instance

        refiner = ChapterRefiner(config=MOCK_REFINER_CONFIG)

        system_prompt = "You are a helpful assistant."
        user_prompt = "Tell me something."
        model_name = "test-model"

        content, usage = refiner.query_llm(system_prompt, user_prompt, model_name)

        assert content is None
        assert usage is None

    @patch("absrefined.refiner.chapter_refiner.OpenAI")
    def test_llm_api_errors(self, mock_openai_constructor):
        """Test handling of various API errors in query_llm."""
        mock_llm_client_instance = MagicMock()

        # Create mock request for APIError
        mock_request = MagicMock()

        # Test authentication error (401)
        auth_error = APIError(
            message="Authentication error",
            request=mock_request,
            body={
                "error": {
                    "message": "Invalid API key",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
        )
        # Add status_code attribute which is checked in the error handler
        auth_error.status_code = 401
        mock_llm_client_instance.chat.completions.create.side_effect = auth_error
        mock_openai_constructor.return_value = mock_llm_client_instance

        refiner = ChapterRefiner(config=MOCK_REFINER_CONFIG)
        content, usage = refiner.query_llm("System prompt", "User prompt", "test-model")

        assert content is None
        assert usage is None

        # Test rate limit error (429)
        rate_limit_error = APIError(
            message="Rate limit exceeded",
            request=mock_request,
            body={
                "error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}
            },
        )
        rate_limit_error.status_code = 429
        mock_llm_client_instance.chat.completions.create.side_effect = rate_limit_error

        content, usage = refiner.query_llm("System prompt", "User prompt", "test-model")
        assert content is None
        assert usage is None

        # Test model not found error (404)
        model_error = APIError(
            message="Model not found",
            request=mock_request,
            body={
                "error": {"message": "Model not found", "type": "invalid_request_error"}
            },
        )
        model_error.status_code = 404
        mock_llm_client_instance.chat.completions.create.side_effect = model_error

        content, usage = refiner.query_llm("System prompt", "User prompt", "test-model")
        assert content is None
        assert usage is None

        # Test generic exception
        mock_llm_client_instance.chat.completions.create.side_effect = Exception(
            "Something went wrong"
        )

        content, usage = refiner.query_llm("System prompt", "User prompt", "test-model")
        assert content is None
        assert usage is None

    @patch("absrefined.refiner.chapter_refiner.OpenAI")
    def test_init_generic_exception(self, mock_openai_constructor):
        """Test handling a generic exception during client initialization."""
        mock_openai_constructor.side_effect = Exception("Connection error")

        with pytest.raises(Exception) as excinfo:
            ChapterRefiner(config=MOCK_REFINER_CONFIG)

        assert "Connection error" in str(excinfo.value)

    @patch("absrefined.refiner.chapter_refiner.ChapterRefiner.query_llm")
    def test_refine_chapter_start_time_query_llm_returns_none(self, mock_query_llm):
        """Test when query_llm returns None."""
        mock_query_llm.return_value = (None, None)

        refiner = ChapterRefiner(config=MOCK_REFINER_CONFIG)

        transcript_segments = [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "Chapter 1",
                "words": [
                    {"word": "Chapter", "start": 0.0, "end": 2.0, "probability": 0.9},
                    {"word": "1", "start": 2.0, "end": 5.0, "probability": 0.9},
                ],
            }
        ]

        refined_time, usage = refiner.refine_chapter_start_time(
            transcript_segments=transcript_segments,
            chapter_title="Chapter 1",
            target_time_seconds=1.0,
            search_window_seconds=10.0,
        )

        assert refined_time is None
        assert usage is None
        mock_query_llm.assert_called_once()

    @patch("absrefined.refiner.chapter_refiner.OpenAI")
    def test_query_llm_missing_usage_data(self, mock_openai_constructor):
        """Test handling missing usage data in LLM response."""
        mock_llm_client_instance = MagicMock()
        mock_chat_completion_response = MagicMock()
        mock_chat_completion_response.choices = [MagicMock()]
        mock_chat_completion_response.choices[0].message.content = "15.0"

        # Set usage to None instead of deleting it
        # This matches line 206 in chapter_refiner.py: "self.logger.warning("LLM response missing usage data.")"
        mock_chat_completion_response.usage = None

        mock_llm_client_instance.chat.completions.create.return_value = (
            mock_chat_completion_response
        )
        mock_openai_constructor.return_value = mock_llm_client_instance

        refiner = ChapterRefiner(config=MOCK_REFINER_CONFIG)

        content, usage = refiner.query_llm(
            "You are a helpful assistant.", "What is 5+10?", "test-model"
        )

        assert content == "15.0"
        assert usage is None
