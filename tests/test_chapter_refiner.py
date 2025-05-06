import pytest
from unittest.mock import patch, MagicMock
import json
import os
from absrefined.refiner.chapter_refiner import ChapterRefiner


# Mock config for the refiner
MOCK_REFINER_OPENAI_API_URL = "http://localhost:7890/v1" # Different from transcriber for clarity
MOCK_REFINER_OPENAI_API_KEY = "test_refiner_openai_api_key_789"
MOCK_REFINER_MODEL_NAME = "gpt-refiner-test"
MOCK_REFINER_CONFIG = {
    "refiner": {
        "openai_api_url": MOCK_REFINER_OPENAI_API_URL,
        "openai_api_key": MOCK_REFINER_OPENAI_API_KEY,
        "model_name": MOCK_REFINER_MODEL_NAME
    },
    "logging": {"level": "DEBUG"} # Basic logging config
}


class TestChapterRefiner:
    """Tests for the ChapterRefiner class."""

    def test_init(self):
        """Test initialization of the ChapterRefiner."""
        # Test with default config
        with patch("absrefined.refiner.chapter_refiner.OpenAI") as mock_openai_constructor:
            refiner = ChapterRefiner(config=MOCK_REFINER_CONFIG)
            assert refiner.api_base_url == MOCK_REFINER_OPENAI_API_URL
            assert refiner.api_key == MOCK_REFINER_OPENAI_API_KEY
            assert refiner.default_model_name == MOCK_REFINER_MODEL_NAME
            # window_size and verbose are no longer attributes of ChapterRefiner init
            mock_openai_constructor.assert_called_once_with(
                api_key=MOCK_REFINER_OPENAI_API_KEY, 
                base_url=MOCK_REFINER_OPENAI_API_URL
            )

    @patch("absrefined.refiner.chapter_refiner.ChapterRefiner.query_llm")
    def test_detect_chapter_start(self, mock_query_llm_method):
        """Test detecting the precise start of a chapter."""
        # Setup the mock query_llm method on an instance
        # query_llm now returns a tuple (content, usage_data)
        # LLM should return a time relative to the chunk, and within search_window_for_test
        mock_llm_response_time = "6.5"
        mock_query_llm_method.return_value = (mock_llm_response_time, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})

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
        target_time_within_chunk = 5.0 # Example: original chapter mark was 5s into this 30s chunk
        
        # The method is now refine_chapter_start_time
        # It returns refined_time_offset_in_chunk (float) or (None, None)
        refined_time_offset, usage = refiner.refine_chapter_start_time(
            transcript_segments=transcript, 
            chapter_title="Chapter 1", 
            target_time_seconds=target_time_within_chunk,
            search_window_seconds=search_window_for_test 
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
            search_window_seconds=search_window_for_test
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
                "model_name": MOCK_REFINER_MODEL_NAME
            }
        }
        with pytest.raises(KeyError) as excinfo:
            ChapterRefiner(config=bad_config)
        assert "Missing openai_api_key" in str(excinfo.value)

    @patch("absrefined.refiner.chapter_refiner.OpenAI")
    def test_refine_chapter_start_time_with_llm_mock(self, mock_openai_constructor, mock_transcript_data):
        """Test refine_chapter_start_time with mocked OpenAI client making a successful call."""
        mock_llm_client_instance = MagicMock()
        mock_chat_completion_response = MagicMock()
        mock_chat_completion_response.choices = [MagicMock()]
        mock_chat_completion_response.choices[0].message.content = "10.5"  # Mock LLM response content
        mock_chat_completion_response.usage.prompt_tokens = 100
        mock_chat_completion_response.usage.completion_tokens = 5
        mock_chat_completion_response.usage.total_tokens = 105
        mock_llm_client_instance.chat.completions.create.return_value = mock_chat_completion_response
        mock_openai_constructor.return_value = mock_llm_client_instance

        refiner = ChapterRefiner(config=MOCK_REFINER_CONFIG)

        if not mock_transcript_data:
            pytest.skip("No transcript data found in chapter-segments, skipping test.")

        chapter_name = next(iter(mock_transcript_data))
        transcript_segments = mock_transcript_data[chapter_name]
        target_time_seconds = 5.0
        search_window_seconds = 60.0 # Assume a 60s window for these segments

        refined_time, usage_data = refiner.refine_chapter_start_time(
            transcript_segments, chapter_name, target_time_seconds, search_window_seconds
        )

        assert refined_time == 10.5
        assert usage_data["prompt_tokens"] == 100
        mock_openai_constructor.assert_called_once_with(
            api_key=MOCK_REFINER_CONFIG["refiner"]["openai_api_key"],
            base_url=MOCK_REFINER_CONFIG["refiner"]["openai_api_url"]
        )
        mock_llm_client_instance.chat.completions.create.assert_called_once()
        # logger is not an attribute of TestChapterRefiner, so self.logger.info will fail.
        # If logging is desired here, import logging and use logging.info()
        # For now, removing the self.logger.info call.
        # self.logger.info("test_refine_chapter_start_time_with_llm_mock passed with simplified mock.")

    @patch("absrefined.refiner.chapter_refiner.OpenAI")
    def test_detect_chapter_start_llm_returns_number(
        self, mock_openai_constructor, mock_transcript_data # Changed mock_openai to mock_openai_constructor
    ):
        """Test chapter detection when LLM returns only a number."""
        # Setup mock OpenAI client
        mock_llm_client_instance = MagicMock()
        mock_chat_completion_response = MagicMock()
        mock_chat_completion_response.choices = [MagicMock()]
        mock_chat_completion_response.choices[0].message.content = " 12.345 " # LLM returns number with spaces
        mock_chat_completion_response.usage.prompt_tokens = 10
        mock_chat_completion_response.usage.completion_tokens = 2
        mock_chat_completion_response.usage.total_tokens = 12
        mock_llm_client_instance.chat.completions.create.return_value = mock_chat_completion_response
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
            search_window_seconds=search_window_s
        )

        assert refined_timestamp == 12.345
        assert usage["total_tokens"] == 12
        mock_llm_client_instance.chat.completions.create.assert_called_once()
