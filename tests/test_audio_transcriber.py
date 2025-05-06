import pytest
from unittest.mock import patch, MagicMock, ANY, mock_open
import json
import os
from pathlib import Path
from absrefined.transcriber.audio_transcriber import AudioTranscriber
from openai import (
    OpenAI,
    APIConnectionError,
    AuthenticationError,
    RateLimitError,
)  # Import OpenAI exceptions


# Mock config for the transcriber
MOCK_OPENAI_API_KEY = "test_openai_api_key_456"
MOCK_OPENAI_API_URL = "http://localhost:1234/v1"
MOCK_WHISPER_MODEL = "whisper-test-1"
MOCK_TRANSCRIBER_CONFIG = {
    "refiner": {
        "openai_api_key": MOCK_OPENAI_API_KEY,
        "openai_api_url": MOCK_OPENAI_API_URL,
        "whisper_model_name": MOCK_WHISPER_MODEL,
    },
    "logging": {
        "level": "DEBUG",
        "debug_files": False,  # Default to False, can be overridden in specific tests if needed
    },
}

# Original raw data for crafting mock objects
# Word data should not have 'probability' if it's not consistently there in OpenAI's Word object
# The transcriber uses getattr(word_raw, 'probability', None)
RAW_WORD_DATA_LIST = [
    {"word": "This", "start": 0.0, "end": 1.0, "probability": 0.9},
    {"word": "is", "start": 1.0, "end": 2.0, "probability": 0.9},
    {"word": "test", "start": 2.0, "end": 3.0, "probability": 0.9},
    {"word": "audio.", "start": 3.0, "end": 5.0, "probability": 0.9},
    {"word": "Chapter", "start": 5.0, "end": 8.0, "probability": 0.9},
    {"word": "1", "start": 8.0, "end": 10.0, "probability": 0.9},
]

RAW_SEGMENT_DATA_LIST = [
    {
        "id": 0,
        "seek": 0,
        "start": 0.0,
        "end": 5.0,
        "text": "This is test audio.",
        "tokens": [50364, 639, 318, 1332, 1115, 13, 50614],
        "temperature": 0.0,
        "avg_logprob": -0.3,
        "compression_ratio": 1.0,
        "no_speech_prob": 0.1,
        "words": [  # Words specific to this segment
            {"word": "This", "start": 0.0, "end": 1.0, "probability": 0.9},
            {"word": "is", "start": 1.0, "end": 2.0, "probability": 0.9},
            {"word": "test", "start": 2.0, "end": 3.0, "probability": 0.9},
            {"word": "audio.", "start": 3.0, "end": 5.0, "probability": 0.9},
        ],
    },
    {
        "id": 1,
        "seek": 500,
        "start": 5.0,
        "end": 10.0,
        "text": "Chapter 1",
        "tokens": [50614, 1383, 257, 13, 50864],
        "temperature": 0.0,
        "avg_logprob": -0.4,
        "compression_ratio": 1.1,
        "no_speech_prob": 0.05,
        "words": [  # Words specific to this segment
            {"word": "Chapter", "start": 5.0, "end": 8.0, "probability": 0.9},
            {"word": "1", "start": 8.0, "end": 10.0, "probability": 0.9},
        ],
    },
]

FULL_TRANSCRIPT_TEXT = "This is test audio. Chapter 1"


# Helper to create a mock Word object
def create_mock_word_object(data_dict):
    mock = MagicMock()
    for key, value in data_dict.items():
        setattr(mock, key, value)
    # Ensure model_dump returns a plain dict for JSON serialization
    mock.model_dump = MagicMock(return_value=data_dict.copy())
    return mock


# Helper to create a mock Segment object
def create_mock_segment_object(data_dict):
    mock = MagicMock()
    # Store a copy of the original dict for model_dump
    original_dict_for_dump = data_dict.copy()
    original_dict_for_dump["words"] = [
        w.copy() for w in data_dict["words"]
    ]  # Ensure words are also dicts in dump

    for key, value in data_dict.items():
        if key == "words":
            setattr(mock, key, [create_mock_word_object(w_data) for w_data in value])
        else:
            setattr(mock, key, value)

    mock.model_dump = MagicMock(return_value=original_dict_for_dump)
    return mock


# Fixture to mock the OpenAI client and its methods
@pytest.fixture
def mock_openai_client():
    with patch("absrefined.transcriber.audio_transcriber.OpenAI") as mock_constructor:
        mock_instance = MagicMock()

        # Mock the response object from client.audio.transcriptions.create()
        mock_transcription_response = MagicMock()

        # Populate with Pydantic-like objects
        mock_transcription_response.segments = [
            create_mock_segment_object(s_data) for s_data in RAW_SEGMENT_DATA_LIST
        ]
        mock_transcription_response.words = [
            create_mock_word_object(w_data) for w_data in RAW_WORD_DATA_LIST
        ]
        mock_transcription_response.text = FULL_TRANSCRIPT_TEXT

        mock_instance.audio.transcriptions.create.return_value = (
            mock_transcription_response
        )
        mock_constructor.return_value = mock_instance
        yield mock_instance


# Fixture for a dummy audio file path
@pytest.fixture
def mock_audio_file(tmp_path):
    audio_path = tmp_path / "test.mp3"
    audio_path.touch()  # Create an empty file for path existence checks
    return str(audio_path)


class TestAudioTranscriber:
    """Tests for the AudioTranscriber class using OpenAI API."""

    def test_init(self):
        """Test initialization of the AudioTranscriber with API key."""
        with patch(
            "absrefined.transcriber.audio_transcriber.OpenAI"
        ) as mock_constructor:
            mock_client_instance = MagicMock()
            mock_constructor.return_value = mock_client_instance

            transcriber = AudioTranscriber(config=MOCK_TRANSCRIBER_CONFIG)
            assert transcriber.api_key == MOCK_OPENAI_API_KEY
            assert transcriber.base_url == MOCK_OPENAI_API_URL
            assert transcriber.whisper_model == MOCK_WHISPER_MODEL
            assert transcriber.client == mock_client_instance
            mock_constructor.assert_called_once_with(
                api_key=MOCK_OPENAI_API_KEY, base_url=MOCK_OPENAI_API_URL
            )

            # Test with debug_files = True
            mock_constructor.reset_mock()
            debug_config = MOCK_TRANSCRIBER_CONFIG.copy()
            debug_config["logging"] = {"debug_files": True, "level": "DEBUG"}
            transcriber_debug = AudioTranscriber(config=debug_config)
            assert transcriber_debug.debug_preserve_files is True

    @patch("builtins.open", new_callable=mock_open, read_data=b"dummy_audio_data")
    def test_transcribe_audio_success_no_offset(
        self, mock_file_open, mock_openai_client, mock_audio_file, tmp_path
    ):
        """Test successful transcription using OpenAI API with no time offset."""
        transcriber = AudioTranscriber(config=MOCK_TRANSCRIBER_CONFIG)
        output_path = tmp_path / "output.jsonl"

        segments = transcriber.transcribe_audio(
            mock_audio_file, str(output_path), segment_start_time=0
        )

        # Verify the structure of the adjusted segments
        assert len(segments) == 2
        # Check first segment (should be identical to mock as offset is 0)
        assert segments[0]["start"] == 0.0
        assert segments[0]["end"] == 5.0
        assert segments[0]["text"] == "This is test audio."
        assert len(segments[0]["words"]) == 4
        assert segments[0]["words"][0]["start"] == 0.0
        assert segments[0]["words"][0]["end"] == 1.0

        # Check second segment
        assert segments[1]["start"] == 5.0
        assert segments[1]["end"] == 10.0
        assert segments[1]["text"] == "Chapter 1"
        assert len(segments[1]["words"]) == 2
        assert segments[1]["words"][0]["start"] == 5.0
        assert segments[1]["words"][0]["end"] == 8.0

        # Verify OpenAI API call
        # Check that the input file was opened in binary read mode among the calls
        mock_file_open.assert_any_call(mock_audio_file, "rb")
        # Ensure the output file was also opened for writing, if write_to_file is True (default)
        if transcriber.debug_preserve_files or (
            output_path and True
        ):  # True is default for write_to_file
            mock_file_open.assert_any_call(str(output_path), "w", encoding="utf-8")

        mock_openai_client.audio.transcriptions.create.assert_called_once_with(
            model=MOCK_WHISPER_MODEL,  # Check against the model from config
            file=ANY,
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )

        # Verify output file writing attempt
        # assert output_path.exists() # mock_open doesn't create real files
        # with open(output_path, "r", encoding="utf-8") as f:
        #     lines = f.readlines()
        #     assert len(lines) == 2
        #     parsed_line_1 = json.loads(lines[0])
        #     assert parsed_line_1["start"] == 0.0 # Re-verify file contents match segments
        #     assert parsed_line_1["words"][0]["start"] == 0.0

    @patch("builtins.open", new_callable=mock_open, read_data=b"dummy_audio_data")
    def test_transcribe_audio_success_with_offset(
        self, mock_file_open, mock_openai_client, mock_audio_file, tmp_path
    ):
        """Test successful transcription using OpenAI API with a time offset."""
        transcriber = AudioTranscriber(config=MOCK_TRANSCRIBER_CONFIG)
        output_path = tmp_path / "output_offset.jsonl"
        offset = 100.5  # Example offset

        segments = transcriber.transcribe_audio(
            mock_audio_file, str(output_path), segment_start_time=offset
        )

        # Verify the structure and adjusted timestamps
        assert len(segments) == 2
        # Check first segment adjusted timestamps
        assert segments[0]["start"] == pytest.approx(0.0 + offset)
        assert segments[0]["end"] == pytest.approx(5.0 + offset)
        assert segments[0]["text"] == "This is test audio."
        assert len(segments[0]["words"]) == 4
        assert segments[0]["words"][0]["start"] == pytest.approx(0.0 + offset)
        assert segments[0]["words"][0]["end"] == pytest.approx(1.0 + offset)

        # Check second segment adjusted timestamps
        assert segments[1]["start"] == pytest.approx(5.0 + offset)
        assert segments[1]["end"] == pytest.approx(10.0 + offset)
        assert segments[1]["text"] == "Chapter 1"
        assert len(segments[1]["words"]) == 2
        assert segments[1]["words"][0]["start"] == pytest.approx(5.0 + offset)
        assert segments[1]["words"][0]["end"] == pytest.approx(8.0 + offset)

        # Verify OpenAI API call (same as before, checking for whisper-1)
        # Check that the input file was opened in binary read mode among the calls
        mock_file_open.assert_any_call(mock_audio_file, "rb")
        # Ensure the output file was also opened for writing, if write_to_file is True (default)
        if transcriber.debug_preserve_files or (
            output_path and True
        ):  # True is default for write_to_file
            mock_file_open.assert_any_call(str(output_path), "w", encoding="utf-8")

        mock_openai_client.audio.transcriptions.create.assert_called_once_with(
            model=MOCK_WHISPER_MODEL,  # Check against the model from config
            file=ANY,
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )

        # Verify output file writing attempt with offset
        # assert output_path.exists() # mock_open doesn't create real files
        # with open(output_path, "r", encoding="utf-8") as f:
        #     lines = f.readlines()
        #     assert len(lines) == 2
        #     parsed_line_1 = json.loads(lines[0])
        #     assert parsed_line_1["start"] == pytest.approx(0.0 + offset) # Verify offset in file
        #     assert parsed_line_1["words"][0]["start"] == pytest.approx(0.0 + offset)

    @patch("builtins.open", new_callable=mock_open, read_data=b"dummy_audio_data")
    def test_transcribe_audio_api_error(
        self, mock_file_open, mock_openai_client, mock_audio_file, tmp_path
    ):
        """Test transcription handles OpenAI API errors."""
        # Create a minimal mock response object with a mock request attribute
        mock_response = MagicMock()
        mock_response.request = MagicMock()

        # Test different API errors
        for error_type in [
            AuthenticationError("Auth error", response=mock_response, body=None),
            RateLimitError("Rate limit error", response=mock_response, body=None),
            APIConnectionError(request=MagicMock()),
        ]:  # APIConnectionError has different signature
            mock_openai_client.reset_mock()  # Reset mock for each error type
            # Configure the side effect on the specific method
            mock_openai_client.audio.transcriptions.create.side_effect = error_type

            transcriber = AudioTranscriber(config=MOCK_TRANSCRIBER_CONFIG)
            output_path = tmp_path / f"output_error_{type(error_type).__name__}.jsonl"

            with pytest.raises(
                type(error_type)
            ):  # Expect the specific error to be raised
                transcriber.transcribe_audio(mock_audio_file, str(output_path))

            # Verify API call was attempted
            mock_openai_client.audio.transcriptions.create.assert_called_once()
            # Ensure no output file was written on error
            assert not output_path.exists()

    def test_transcribe_audio_no_output_file_needed(
        self, mock_openai_client, mock_audio_file
    ):
        """Test transcription without writing to a file."""
        # Don't need tmp_path if not writing file
        transcriber = AudioTranscriber(config=MOCK_TRANSCRIBER_CONFIG)

        # Mock open specifically for this test to ensure it's NOT called for output
        with patch(
            "builtins.open", new_callable=mock_open, read_data=b"dummy_audio_data"
        ) as mock_file_open:
            segments = transcriber.transcribe_audio(
                mock_audio_file, output_file=None, write_to_file=False
            )

            # Verify segments are returned correctly
            assert len(segments) == 2
            assert segments[0]["start"] == 0.0  # Check basic segment structure

            # Verify API call was made
            mock_openai_client.audio.transcriptions.create.assert_called_once()

            # Verify builtins.open was only called for reading the input audio file
            # It should be called once with 'rb' mode. Any other call would be for writing output.
            found_read_call = False
            for call_args, call_kwargs in mock_file_open.call_args_list:
                if call_args[0] == mock_audio_file and call_args[1] == "rb":
                    found_read_call = True
                elif (
                    call_args[1] != "rb"
                ):  # If any call is not 'rb', it's likely the output write
                    pytest.fail(
                        f"builtins.open called unexpectedly for writing: {call_args}"
                    )
            assert found_read_call

    def test_transcribe_audio_empty_response(
        self, mock_openai_client, mock_audio_file, tmp_path
    ):
        """Test transcription when the API returns no segments."""
        # Configure mock to return an empty segments list
        mock_response = MagicMock()
        mock_response.segments = []
        mock_response.words = []  # Also ensure words are empty for this test case
        mock_response.text = ""  # Explicitly set text for empty response
        mock_openai_client.audio.transcriptions.create.return_value = mock_response

        # Config for this specific test to enable debug file writing
        debug_config = MOCK_TRANSCRIBER_CONFIG.copy()
        debug_config["logging"] = {"debug_files": True, "level": "DEBUG"}
        transcriber = AudioTranscriber(config=debug_config)

        output_path = tmp_path / "empty_output.jsonl"
        debug_output_path = (
            tmp_path / f"{Path(mock_audio_file).stem}_transcript_DEBUG.jsonl"
        )

        with patch("builtins.open", new_callable=mock_open) as mock_file_write:
            segments = transcriber.transcribe_audio(
                mock_audio_file, str(output_path), write_to_file=True
            )
            assert segments == []

            # Verify that the debug output file was written (due to debug_config)
            mock_file_write.assert_any_call(
                str(debug_output_path), "w", encoding="utf-8"
            )

            # Verify that the main output file (output_path) was NOT written to,
            # as the transcriber returns early if API provides no segments/words.
            main_output_file_written = False
            for call_args_tuple in mock_file_write.call_args_list:
                # call_args_tuple is like (('/path/to/file', 'w'), {'encoding': 'utf-8'})
                # or just (('/path/to/file', 'rb'),) for read
                if call_args_tuple[0][0] == str(output_path):
                    main_output_file_written = True
                    break
            assert not main_output_file_written, (
                f"Main output file {output_path} should not have been written when API returns no segments/words."
            )
