import pytest
from unittest.mock import patch, MagicMock, ANY, mock_open
import json
import os
from pathlib import Path
from absrefined.transcriber.audio_transcriber import AudioTranscriber
from openai import OpenAI, APIConnectionError, AuthenticationError, RateLimitError # Import OpenAI exceptions


# Mock data simulating OpenAI verbose_json response
MOCK_OPENAI_RESPONSE_DATA = {
    "task": "transcribe",
    "language": "english",
    "duration": 10.0,
    "text": "This is test audio. Chapter 1",
    "words": [
        {"word": "This", "start": 0.0, "end": 1.0},
        {"word": "is", "start": 1.0, "end": 2.0},
        {"word": "test", "start": 2.0, "end": 3.0},
        {"word": "audio.", "start": 3.0, "end": 5.0},
        {"word": "Chapter", "start": 5.0, "end": 8.0},
        {"word": "1", "start": 8.0, "end": 10.0},
    ],
    "segments": [
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
            "words": [
                {"word": "This", "start": 0.0, "end": 1.0, "probability": 0.9}, # Keep probability for structure similarity if needed downstream, though OpenAI response may vary
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
            "words": [
                {"word": "Chapter", "start": 5.0, "end": 8.0, "probability": 0.9},
                {"word": "1", "start": 8.0, "end": 10.0, "probability": 0.9},
            ],
        },
    ],
}


# Fixture to mock the OpenAI client and its methods
@pytest.fixture
def mock_openai_client():
    with patch("absrefined.transcriber.audio_transcriber.OpenAI") as mock_constructor:
        mock_instance = MagicMock()
        # Mock the response object structure directly if needed
        mock_response = MagicMock()
        mock_response.segments = MOCK_OPENAI_RESPONSE_DATA["segments"] # Simulate the response structure
        mock_instance.audio.transcriptions.create.return_value = mock_response
        mock_constructor.return_value = mock_instance
        yield mock_instance


# Fixture for a dummy audio file path
@pytest.fixture
def mock_audio_file(tmp_path):
    audio_path = tmp_path / "test.mp3"
    audio_path.touch() # Create an empty file for path existence checks
    return str(audio_path)


class TestAudioTranscriber:
    """Tests for the AudioTranscriber class using OpenAI API."""

    def test_init(self):
        """Test initialization of the AudioTranscriber with API key."""
        api_key = "test_api_key"
        with patch("absrefined.transcriber.audio_transcriber.OpenAI") as mock_constructor:
             mock_client = MagicMock()
             mock_constructor.return_value = mock_client

             # Test default verbose
             transcriber = AudioTranscriber(api_key=api_key)
             assert transcriber.verbose is False
             assert transcriber.api_key == api_key
             assert transcriber.client == mock_client
             mock_constructor.assert_called_once_with(api_key=api_key)

             # Reset mock for next assertion
             mock_constructor.reset_mock()

             # Test verbose=True
             transcriber_verbose = AudioTranscriber(api_key=api_key, verbose=True)
             assert transcriber_verbose.verbose is True
             assert transcriber_verbose.client == mock_client
             mock_constructor.assert_called_once_with(api_key=api_key)


    @patch("builtins.open", new_callable=mock_open, read_data=b"dummy_audio_data")
    def test_transcribe_audio_success_no_offset(self, mock_file_open, mock_openai_client, mock_audio_file, tmp_path):
        """Test successful transcription using OpenAI API with no time offset."""
        api_key = "test_key"
        transcriber = AudioTranscriber(api_key=api_key, verbose=False)
        output_path = tmp_path / "output.jsonl"

        segments = transcriber.transcribe_audio(mock_audio_file, str(output_path), segment_start_time=0)

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
        # Ensure the output file was also opened for writing
        mock_file_open.assert_any_call(str(output_path), "w", encoding="utf-8")

        mock_openai_client.audio.transcriptions.create.assert_called_once_with(
            model="whisper-1",
            file=ANY,
            response_format="verbose_json",
            timestamp_granularities=["word"]
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
    def test_transcribe_audio_success_with_offset(self, mock_file_open, mock_openai_client, mock_audio_file, tmp_path):
        """Test successful transcription using OpenAI API with a time offset."""
        api_key = "test_key"
        transcriber = AudioTranscriber(api_key=api_key, verbose=False)
        output_path = tmp_path / "output_offset.jsonl"
        offset = 100.5 # Example offset

        segments = transcriber.transcribe_audio(mock_audio_file, str(output_path), segment_start_time=offset)

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
        # Ensure the output file was also opened for writing
        mock_file_open.assert_any_call(str(output_path), "w", encoding="utf-8")

        mock_openai_client.audio.transcriptions.create.assert_called_once_with(
            model="whisper-1",
            file=ANY,
            response_format="verbose_json",
            timestamp_granularities=["word"]
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
    def test_transcribe_audio_api_error(self, mock_file_open, mock_openai_client, mock_audio_file, tmp_path):
        """Test transcription handles OpenAI API errors."""
        api_key = "test_key"
        # Create a minimal mock response object with a mock request attribute
        mock_response = MagicMock()
        mock_response.request = MagicMock()

        # Test different API errors
        for error_type in [AuthenticationError("Auth error", response=mock_response, body=None),
                           RateLimitError("Rate limit error", response=mock_response, body=None),
                           APIConnectionError(request=MagicMock())]: # APIConnectionError has different signature
            mock_openai_client.reset_mock() # Reset mock for each error type
            # Configure the side effect on the specific method
            mock_openai_client.audio.transcriptions.create.side_effect = error_type

            transcriber = AudioTranscriber(api_key=api_key)
            output_path = tmp_path / f"output_error_{type(error_type).__name__}.jsonl"

            with pytest.raises(type(error_type)): # Expect the specific error to be raised
                 transcriber.transcribe_audio(mock_audio_file, str(output_path))

            # Verify API call was attempted
            mock_openai_client.audio.transcriptions.create.assert_called_once()
            # Ensure no output file was written on error
            assert not output_path.exists()


    def test_transcribe_audio_no_output_file_needed(self, mock_openai_client, mock_audio_file):
        """Test transcription without writing to a file."""
        # Don't need tmp_path if not writing file
        api_key = "test_key"
        transcriber = AudioTranscriber(api_key=api_key)

        # Mock open specifically for this test to ensure it's NOT called for output
        with patch("builtins.open", new_callable=mock_open, read_data=b"dummy_audio_data") as mock_file_open:
            segments = transcriber.transcribe_audio(mock_audio_file, output_file=None, write_to_file=False)

            # Verify segments are returned correctly
            assert len(segments) == 2
            assert segments[0]['start'] == 0.0 # Check basic segment structure

            # Verify API call was made
            mock_openai_client.audio.transcriptions.create.assert_called_once()

            # Verify builtins.open was only called for reading the input audio file
            # It should be called once with 'rb' mode. Any other call would be for writing output.
            found_read_call = False
            for call_args, call_kwargs in mock_file_open.call_args_list:
                 if call_args[0] == mock_audio_file and call_args[1] == 'rb':
                     found_read_call = True
                 elif call_args[1] != 'rb': # If any call is not 'rb', it's likely the output write
                     pytest.fail(f"builtins.open called unexpectedly for writing: {call_args}")
            assert found_read_call


    def test_transcribe_audio_empty_response(self, mock_openai_client, mock_audio_file, tmp_path):
        """Test transcription when the API returns no segments."""
        api_key = "test_key"
        # Configure mock to return an empty segments list
        mock_response = MagicMock()
        mock_response.segments = []
        mock_openai_client.audio.transcriptions.create.return_value = mock_response

        transcriber = AudioTranscriber(api_key=api_key)
        output_path = tmp_path / "output_empty.jsonl"

        # Mock open to check calls
        with patch("builtins.open", new_callable=mock_open, read_data=b"dummy_audio_data") as mock_file_open:
             segments = transcriber.transcribe_audio(mock_audio_file, str(output_path), write_to_file=True)

             assert segments == []
             mock_openai_client.audio.transcriptions.create.assert_called_once()

             # Should NOT write an empty file if segments are empty
             assert not output_path.exists()

             # Verify open was only called for the input file
             mock_file_open.assert_called_once_with(mock_audio_file, "rb")


    # --- Tests for read_transcription remain the same as they don't depend on the transcription method ---

    def test_read_transcription(self, tmp_path):
        """Test reading a valid transcription file."""
        transcriber = AudioTranscriber(api_key="dummy_key") # API key not needed for reading
        transcription_path = tmp_path / "test_read.jsonl"
        mock_segments = [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "Segment 1",
                "words": [
                    {"word": "Segment", "start": 0.0, "end": 2.5},
                    {"word": "1", "start": 2.5, "end": 5.0},
                ],
            },
            {"start": 5.0, "end": 10.0, "text": "Segment 2", "words": []}, # Ensure words key exists even if empty
            {
                "start": 10.0,
                "end": 15.0,
                "text": "Segment 3",
                "words": [],
            },
        ]

        with open(transcription_path, "w", encoding="utf-8") as f:
            for segment in mock_segments:
                json.dump(segment, f)
                f.write("\n")

        read_segments = transcriber.read_transcription(str(transcription_path))
        assert read_segments == mock_segments

    def test_read_transcription_non_existent(self):
        """Test reading a non-existent transcription file."""
        transcriber = AudioTranscriber(api_key="dummy_key")
        read_segments = transcriber.read_transcription("non_existent_file.jsonl")
        assert read_segments == []

    def test_read_transcription_invalid_json(self, tmp_path):
        """Test reading a transcription file with invalid JSON."""
        transcriber = AudioTranscriber(api_key="dummy_key")
        transcription_path = tmp_path / "invalid.jsonl"
        with open(transcription_path, "w", encoding="utf-8") as f:
            f.write("this is not json\n")
            f.write('{"start": 0.0, "end": 1.0, "text": "valid", "words":[]}\n') # Added words

        # Based on current code, it returns [] on any exception during reading loop
        read_segments = transcriber.read_transcription(str(transcription_path))
        assert read_segments == [] # Expecting empty list on parse error
