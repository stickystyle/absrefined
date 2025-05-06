import pytest
from unittest.mock import patch, MagicMock, mock_open
import json
import os
import requests
from absrefined.client.abs_client import AudiobookshelfClient


# Mock config for the client
MOCK_SERVER_URL = "http://test-server.com"
MOCK_API_KEY = "test_api_key_123"
MOCK_CONFIG = {
    "audiobookshelf": {"host": MOCK_SERVER_URL, "api_key": MOCK_API_KEY, "timeout": 15},
    "logging": {"level": "DEBUG"},  # Add a basic logging config
}


class TestAudiobookshelfClient:
    """Tests for the AudiobookshelfClient class."""

    def test_init(self):
        """Test initialization of the AudiobookshelfClient."""
        client = AudiobookshelfClient(MOCK_CONFIG)
        assert client.server_url == MOCK_SERVER_URL
        assert client.api_key == MOCK_API_KEY
        assert client.request_timeout == 15
        # user_id is not set during init with API key
        # verbose is handled by logging config

    def test_init_missing_host(self):
        """Test initialization with missing host."""
        config_without_host = {"audiobookshelf": {"api_key": MOCK_API_KEY}}
        with pytest.raises(KeyError, match="Audiobookshelf host not found"):
            AudiobookshelfClient(config_without_host)

    def test_init_missing_api_key(self):
        """Test initialization with missing API key."""
        config_without_api_key = {"audiobookshelf": {"host": MOCK_SERVER_URL}}
        with pytest.raises(KeyError, match="Audiobookshelf API key not found"):
            AudiobookshelfClient(config_without_api_key)

    def test_init_empty_config(self):
        """Test initialization with empty config."""
        with pytest.raises(KeyError, match="Audiobookshelf host not found"):
            AudiobookshelfClient({})

    def test_init_default_timeout(self):
        """Test that default timeout is used when not specified."""
        config_without_timeout = {
            "audiobookshelf": {"host": MOCK_SERVER_URL, "api_key": MOCK_API_KEY}
        }
        client = AudiobookshelfClient(config_without_timeout)
        assert client.request_timeout == 30  # Default timeout

    def test_get_auth_headers(self):
        """Test the _get_auth_headers method."""
        client = AudiobookshelfClient(MOCK_CONFIG)
        headers = client._get_auth_headers()
        assert headers == {"Authorization": f"Bearer {MOCK_API_KEY}"}

    def test_get_auth_headers_missing_api_key(self):
        """Test _get_auth_headers with missing API key."""
        client = AudiobookshelfClient(MOCK_CONFIG)
        # Manually set api_key to None to simulate a potential issue
        client.api_key = None
        with pytest.raises(ValueError, match="API key not configured"):
            client._get_auth_headers()

    @patch("absrefined.client.abs_client.requests.get")
    def test_get_item_details(self, mock_get, mock_abs_response, abs_item_response):
        """Test getting item details."""
        # Setup the mock response
        mock_get.return_value = mock_abs_response(json_data=abs_item_response)

        # Create client
        client = AudiobookshelfClient(MOCK_CONFIG)

        # Test get_item_details
        item = client.get_item_details("lib-item-123")

        # Verify results
        assert item == abs_item_response

        # Verify API call
        mock_get.assert_called_once_with(
            f"{MOCK_SERVER_URL}/api/items/lib-item-123",
            headers={"Authorization": f"Bearer {MOCK_API_KEY}"},
            timeout=MOCK_CONFIG["audiobookshelf"]["timeout"],
        )

    @patch("absrefined.client.abs_client.requests.get")
    def test_get_item_details_http_error(self, mock_get, mock_abs_response):
        """Test get_item_details with HTTP error."""
        # Setup the mock to raise an HTTP error
        http_error = requests.exceptions.HTTPError("404 Client Error")
        mock_get.return_value = mock_abs_response(
            status_code=404, raise_for_status=http_error
        )

        client = AudiobookshelfClient(MOCK_CONFIG)
        result = client.get_item_details("non-existent-item")

        # Should return empty dict on error
        assert result == {}
        mock_get.assert_called_once()

    @patch("absrefined.client.abs_client.requests.get")
    def test_get_item_details_connection_error(self, mock_get):
        """Test get_item_details with connection error."""
        # Setup the mock to raise a connection error
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        client = AudiobookshelfClient(MOCK_CONFIG)
        result = client.get_item_details("lib-item-123")

        # Should return empty dict on error
        assert result == {}
        mock_get.assert_called_once()

    @patch("absrefined.client.abs_client.requests.get")
    def test_get_item_details_timeout_error(self, mock_get):
        """Test get_item_details with timeout error."""
        # Setup the mock to raise a timeout error
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

        client = AudiobookshelfClient(MOCK_CONFIG)
        result = client.get_item_details("lib-item-123")

        # Should return empty dict on error
        assert result == {}
        mock_get.assert_called_once()

    @patch("absrefined.client.abs_client.requests.get")
    def test_get_item_details_unexpected_error(self, mock_get):
        """Test get_item_details with unexpected error."""
        # Setup the mock to raise an unexpected error
        mock_get.side_effect = Exception("Unexpected error")

        client = AudiobookshelfClient(MOCK_CONFIG)
        result = client.get_item_details("lib-item-123")

        # Should return empty dict on error
        assert result == {}
        mock_get.assert_called_once()

    @patch("absrefined.client.abs_client.requests.get")
    def test_get_item_chapters(self, mock_get, mock_abs_response, abs_item_response):
        """Test getting item chapters."""
        # Setup the mock response
        mock_get.return_value = mock_abs_response(json_data=abs_item_response)

        # Create client
        client = AudiobookshelfClient(MOCK_CONFIG)

        # Test get_item_chapters
        # This internally calls get_item_details, so mock_get will be called.
        chapters = client.get_item_chapters("lib-item-123")

        # Verify results
        assert chapters == abs_item_response["media"]["chapters"]

        # Verify API call (to get_item_details)
        mock_get.assert_called_once_with(
            f"{MOCK_SERVER_URL}/api/items/lib-item-123",  # Called by get_item_details
            headers={"Authorization": f"Bearer {MOCK_API_KEY}"},
            timeout=MOCK_CONFIG["audiobookshelf"]["timeout"],
        )

    @patch("absrefined.client.abs_client.requests.get")
    def test_get_item_chapters_empty_response(self, mock_get, mock_abs_response):
        """Test getting item chapters with empty response."""
        # Setup the mock response with no chapters
        mock_get.return_value = mock_abs_response(json_data={})

        client = AudiobookshelfClient(MOCK_CONFIG)
        chapters = client.get_item_chapters("lib-item-123")

        # Should return empty list if no chapters
        assert chapters == []
        mock_get.assert_called_once()

    @patch("absrefined.client.abs_client.requests.get")
    def test_get_item_chapters_no_media(self, mock_get, mock_abs_response):
        """Test getting item chapters with no media key."""
        # Setup the mock response with no media key
        mock_get.return_value = mock_abs_response(json_data={"id": "lib-item-123"})

        client = AudiobookshelfClient(MOCK_CONFIG)
        chapters = client.get_item_chapters("lib-item-123")

        # Should return empty list if no media key
        assert chapters == []
        mock_get.assert_called_once()

    @patch("absrefined.client.abs_client.requests.get")
    def test_get_item_chapters_no_chapters(self, mock_get, mock_abs_response):
        """Test getting item chapters with no chapters key."""
        # Setup the mock response with media but no chapters
        mock_get.return_value = mock_abs_response(
            json_data={"id": "lib-item-123", "media": {}}
        )

        client = AudiobookshelfClient(MOCK_CONFIG)
        chapters = client.get_item_chapters("lib-item-123")

        # Should return empty list if no chapters key
        assert chapters == []
        mock_get.assert_called_once()

    @patch("absrefined.client.abs_client.requests.get")
    def test_download_audio_file(self, mock_get, mock_abs_response, tmp_path):
        """Test downloading an audio file."""
        # Setup the mock responses
        mock_audio_content = b"test audio data"

        # Mock item details response with audioFiles (ino needed for download URL)
        item_details_with_audio_info = {
            "id": "lib-item-123",
            "media": {
                "audioFiles": [
                    {
                        "index": 0,  # Or 1, depending on server
                        "ino": "audiofile-ino-12345",
                        "relPath": "test.mp3",
                        "duration": 200.0,
                    }
                ]
            },
        }

        # Mock responses: first for get_item_details, second for file download
        mock_get_item_details_response = mock_abs_response(
            json_data=item_details_with_audio_info
        )
        mock_download_stream_response = mock_abs_response(
            content=mock_audio_content,
            headers={"Content-Length": str(len(mock_audio_content))},
        )

        # Configure mock to return different responses for different URLs/calls
        # First call is get_item_details, second is the actual download
        mock_get.side_effect = [
            mock_get_item_details_response,  # For get_item_details() call within download_audio_file
            mock_download_stream_response,  # For the actual file download call
        ]

        client = AudiobookshelfClient(MOCK_CONFIG)

        temp_dir = str(tmp_path)
        output_filename = "downloaded_audio.mp3"
        expected_output_path = os.path.join(temp_dir, output_filename)

        # Call the method to test
        result_path = client.download_audio_file("lib-item-123", expected_output_path)

        assert result_path == expected_output_path
        assert os.path.exists(expected_output_path)
        with open(expected_output_path, "rb") as f:
            assert f.read() == mock_audio_content

        # Verify API calls
        assert mock_get.call_count == 2

        # Call 1: get_item_details
        mock_get.assert_any_call(
            f"{MOCK_SERVER_URL}/api/items/lib-item-123",
            headers={"Authorization": f"Bearer {MOCK_API_KEY}"},
            timeout=MOCK_CONFIG["audiobookshelf"]["timeout"],
        )
        # Call 2: actual file download
        mock_get.assert_any_call(
            f"{MOCK_SERVER_URL}/api/items/lib-item-123/file/audiofile-ino-12345",
            headers={"Authorization": f"Bearer {MOCK_API_KEY}"},
            stream=True,
            timeout=MOCK_CONFIG["audiobookshelf"]["timeout"],
        )

    @patch("absrefined.client.abs_client.requests.get")
    def test_download_audio_file_no_content_length(
        self, mock_get, mock_abs_response, tmp_path
    ):
        """Test downloading an audio file with no Content-Length header."""
        # Setup the mock responses
        mock_audio_content = b"test audio data"

        # Mock item details response with audioFiles
        item_details_with_audio_info = {
            "id": "lib-item-123",
            "media": {
                "audioFiles": [
                    {
                        "ino": "audiofile-ino-12345",
                        "relPath": "test.mp3",
                    }
                ]
            },
        }

        # Mock responses: first for get_item_details, second for file download (without Content-Length)
        mock_get_item_details_response = mock_abs_response(
            json_data=item_details_with_audio_info
        )
        mock_download_stream_response = mock_abs_response(
            content=mock_audio_content, headers={}
        )

        mock_get.side_effect = [
            mock_get_item_details_response,
            mock_download_stream_response,
        ]

        client = AudiobookshelfClient(MOCK_CONFIG)

        temp_dir = str(tmp_path)
        output_filename = "downloaded_audio_no_length.mp3"
        expected_output_path = os.path.join(temp_dir, output_filename)

        # Call the method to test
        result_path = client.download_audio_file("lib-item-123", expected_output_path)

        assert result_path == expected_output_path
        assert os.path.exists(expected_output_path)
        with open(expected_output_path, "rb") as f:
            assert f.read() == mock_audio_content

    @patch("absrefined.client.abs_client.requests.get")
    def test_download_audio_file_no_audio_files(self, mock_get, mock_abs_response):
        """Test downloading an audio file when no audio files exist."""
        # Mock item details response with no audioFiles
        item_details_without_audio = {
            "id": "lib-item-123",
            "media": {},  # No audioFiles
        }

        mock_get.return_value = mock_abs_response(json_data=item_details_without_audio)

        client = AudiobookshelfClient(MOCK_CONFIG)
        result_path = client.download_audio_file("lib-item-123", "output.mp3")

        # Should return empty string if no audio files
        assert result_path == ""
        mock_get.assert_called_once()

    @patch("absrefined.client.abs_client.requests.get")
    def test_download_audio_file_empty_audio_files(self, mock_get, mock_abs_response):
        """Test downloading an audio file when audio files array is empty."""
        # Mock item details response with empty audioFiles array
        item_details_with_empty_audio = {
            "id": "lib-item-123",
            "media": {
                "audioFiles": []  # Empty array
            },
        }

        mock_get.return_value = mock_abs_response(
            json_data=item_details_with_empty_audio
        )

        client = AudiobookshelfClient(MOCK_CONFIG)
        result_path = client.download_audio_file("lib-item-123", "output.mp3")

        # Should return empty string if empty audio files array
        assert result_path == ""
        mock_get.assert_called_once()

    @patch("absrefined.client.abs_client.requests.get")
    def test_download_audio_file_missing_ino(self, mock_get, mock_abs_response):
        """Test downloading an audio file when the ino field is missing."""
        # Mock item details response with audioFiles but missing ino
        item_details_with_missing_ino = {
            "id": "lib-item-123",
            "media": {
                "audioFiles": [
                    {
                        "index": 0,
                        "relPath": "test.mp3",
                        # No ino field
                    }
                ]
            },
        }

        mock_get.return_value = mock_abs_response(
            json_data=item_details_with_missing_ino
        )

        client = AudiobookshelfClient(MOCK_CONFIG)
        result_path = client.download_audio_file("lib-item-123", "output.mp3")

        # Should return empty string if ino is missing
        assert result_path == ""
        mock_get.assert_called_once()

    @patch("absrefined.client.abs_client.requests.get")
    def test_download_audio_file_http_error(self, mock_get, mock_abs_response):
        """Test download_audio_file with HTTP error during download."""
        # Mock item details response with audioFiles
        item_details_with_audio = {
            "id": "lib-item-123",
            "media": {
                "audioFiles": [
                    {
                        "ino": "audiofile-ino-12345",
                        "relPath": "test.mp3",
                    }
                ]
            },
        }

        # First call succeeds for get_item_details, second call (download) fails
        mock_get_details_response = mock_abs_response(json_data=item_details_with_audio)

        http_error = requests.exceptions.HTTPError("404 Client Error")
        mock_download_error_response = mock_abs_response(
            status_code=404, raise_for_status=http_error
        )

        mock_get.side_effect = [
            mock_get_details_response,  # get_item_details succeeds
            mock_download_error_response,  # download fails
        ]

        client = AudiobookshelfClient(MOCK_CONFIG)
        result_path = client.download_audio_file("lib-item-123", "output.mp3")

        # Should return empty string on download error
        assert result_path == ""
        assert mock_get.call_count == 2

    @patch("absrefined.client.abs_client.requests.get")
    def test_download_audio_file_io_error(self, mock_get, mock_abs_response, tmp_path):
        """Test download_audio_file with IO error during file writing."""
        # Setup the mock responses
        mock_audio_content = b"test audio data"

        # Mock item details response with audioFiles
        item_details_with_audio = {
            "id": "lib-item-123",
            "media": {
                "audioFiles": [
                    {
                        "ino": "audiofile-ino-12345",
                        "relPath": "test.mp3",
                    }
                ]
            },
        }

        mock_get_details_response = mock_abs_response(json_data=item_details_with_audio)
        mock_download_response = mock_abs_response(
            content=mock_audio_content, headers={"Content-Length": "100"}
        )

        mock_get.side_effect = [mock_get_details_response, mock_download_response]

        client = AudiobookshelfClient(MOCK_CONFIG)

        # Create a directory where we want the file to be - this will cause an IOError
        # because we're trying to write to a directory, not a file
        invalid_path = os.path.join(str(tmp_path), "this_is_a_dir")
        os.makedirs(invalid_path)

        with patch("builtins.open", side_effect=IOError("Is a directory")):
            result_path = client.download_audio_file("lib-item-123", invalid_path)

        # Should return empty string on IO error
        assert result_path == ""
        assert mock_get.call_count == 2

    @patch("absrefined.client.abs_client.requests.post")
    @patch.object(AudiobookshelfClient, "get_item_chapters")
    def test_update_chapters_start_time(
        self, mock_get_chapters, mock_post, mock_abs_response
    ):
        """Test updating chapter start times."""
        # Setup mock for get_item_chapters
        original_chapters = [
            {"id": "ch1", "start": 10.0, "end": 20.0, "title": "Chapter 1"},
            {"id": "ch2", "start": 20.0, "end": 30.0, "title": "Chapter 2"},
            {"id": "ch3", "start": 30.0, "end": 40.0, "title": "Chapter 3"},
        ]
        mock_get_chapters.return_value = [
            chap.copy() for chap in original_chapters
        ]  # Return copies

        # Setup mock for the POST request
        mock_post.return_value = mock_abs_response(status_code=200)

        client = AudiobookshelfClient(MOCK_CONFIG)

        chapter_updates = [{"id": "ch1", "start": 12.5}, {"id": "ch3", "start": 35.5}]

        result = client.update_chapters_start_time("lib-item-123", chapter_updates)

        assert result is True
        mock_get_chapters.assert_called_once_with("lib-item-123")

        # Expected final chapters after updates (ch1 and ch3 start times updated)
        expected_payload_chapters = [
            {"id": "ch1", "start": 12.5, "end": 20.0, "title": "Chapter 1"},
            {"id": "ch2", "start": 20.0, "end": 35.5, "title": "Chapter 2"},
            {"id": "ch3", "start": 35.5, "end": 40.0, "title": "Chapter 3"},
        ]

        mock_post.assert_called_once_with(
            f"{MOCK_SERVER_URL}/api/items/lib-item-123/chapters",
            json={"chapters": expected_payload_chapters},
            headers={
                "Authorization": f"Bearer {MOCK_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=MOCK_CONFIG["audiobookshelf"]["timeout"],
        )

    @patch("absrefined.client.abs_client.requests.post")
    @patch.object(AudiobookshelfClient, "get_item_chapters")
    def test_update_chapters_start_time_no_chapters(self, mock_get_chapters, mock_post):
        """Test updating chapter start times when no chapters exist."""
        # Setup mock for get_item_chapters to return empty list
        mock_get_chapters.return_value = []

        client = AudiobookshelfClient(MOCK_CONFIG)

        chapter_updates = [{"id": "ch1", "start": 12.5}]

        result = client.update_chapters_start_time("lib-item-123", chapter_updates)

        # Should return False if no chapters
        assert result is False
        mock_get_chapters.assert_called_once_with("lib-item-123")
        mock_post.assert_not_called()

    @patch("absrefined.client.abs_client.requests.post")
    @patch.object(AudiobookshelfClient, "get_item_chapters")
    def test_update_chapters_start_time_invalid_updates(
        self, mock_get_chapters, mock_post
    ):
        """Test updating chapter start times with invalid update data."""
        # Setup mock for get_item_chapters
        original_chapters = [
            {"id": "ch1", "start": 10.0, "end": 20.0, "title": "Chapter 1"},
            {"id": "ch2", "start": 20.0, "end": 30.0, "title": "Chapter 2"},
        ]
        mock_get_chapters.return_value = [chap.copy() for chap in original_chapters]

        client = AudiobookshelfClient(MOCK_CONFIG)

        # Invalid update - missing 'id'
        invalid_updates = [{"start": 12.5}]

        result = client.update_chapters_start_time("lib-item-123", invalid_updates)

        # Should still return True (we just log warnings for invalid updates)
        assert result is True
        mock_get_chapters.assert_called_once_with("lib-item-123")

        # POST should NOT be called since no updates were made (updated_count == 0)
        mock_post.assert_not_called()

    @patch("absrefined.client.abs_client.requests.post")
    @patch.object(AudiobookshelfClient, "get_item_chapters")
    def test_update_chapters_start_time_nonexistent_id(
        self, mock_get_chapters, mock_post, mock_abs_response
    ):
        """Test updating chapter start times with non-existent chapter ID."""
        # Setup mock for get_item_chapters
        original_chapters = [
            {"id": "ch1", "start": 10.0, "end": 20.0, "title": "Chapter 1"},
            {"id": "ch2", "start": 20.0, "end": 30.0, "title": "Chapter 2"},
        ]
        mock_get_chapters.return_value = [chap.copy() for chap in original_chapters]

        # Setup mock for the POST request
        mock_post.return_value = mock_abs_response(status_code=200)

        client = AudiobookshelfClient(MOCK_CONFIG)

        # Update with non-existent chapter ID
        nonexistent_updates = [{"id": "ch3", "start": 35.5}]

        result = client.update_chapters_start_time("lib-item-123", nonexistent_updates)

        # Should still return True (we just skip non-existent chapters)
        assert result is True
        mock_get_chapters.assert_called_once_with("lib-item-123")

        # POST should NOT be called since no updates were made (updated_count == 0)
        mock_post.assert_not_called()

    @patch("absrefined.client.abs_client.requests.post")
    @patch.object(AudiobookshelfClient, "get_item_chapters")
    def test_update_chapters_start_time_invalid_start_time(
        self, mock_get_chapters, mock_post, mock_abs_response
    ):
        """Test updating chapter start times with invalid start time values."""
        # Setup mock for get_item_chapters
        original_chapters = [
            {"id": "ch1", "start": 10.0, "end": 20.0, "title": "Chapter 1"},
            {"id": "ch2", "start": 20.0, "end": 30.0, "title": "Chapter 2"},
        ]
        mock_get_chapters.return_value = [chap.copy() for chap in original_chapters]

        # Setup mock for the POST request
        mock_post.return_value = mock_abs_response(status_code=200)

        client = AudiobookshelfClient(MOCK_CONFIG)

        # Update with non-numeric start time
        invalid_updates = [{"id": "ch1", "start": "not-a-number"}]

        result = client.update_chapters_start_time("lib-item-123", invalid_updates)

        # Should still return True (we just skip invalid start times)
        assert result is True
        mock_get_chapters.assert_called_once_with("lib-item-123")

        # POST should NOT be called since no updates were made (updated_count == 0)
        mock_post.assert_not_called()

    @patch("absrefined.client.abs_client.requests.post")
    @patch.object(AudiobookshelfClient, "get_item_chapters")
    def test_update_chapters_start_time_no_change(
        self, mock_get_chapters, mock_post, mock_abs_response
    ):
        """Test updating chapter start times with values that match current times."""
        # Setup mock for get_item_chapters
        original_chapters = [
            {"id": "ch1", "start": 10.0, "end": 20.0, "title": "Chapter 1"},
            {"id": "ch2", "start": 20.0, "end": 30.0, "title": "Chapter 2"},
        ]
        mock_get_chapters.return_value = [chap.copy() for chap in original_chapters]

        # Setup mock for the POST request
        mock_post.return_value = mock_abs_response(status_code=200)

        client = AudiobookshelfClient(MOCK_CONFIG)

        # Update with the same start times
        no_change_updates = [{"id": "ch1", "start": 10.0}, {"id": "ch2", "start": 20.0}]

        result = client.update_chapters_start_time("lib-item-123", no_change_updates)

        # Should still return True
        assert result is True
        mock_get_chapters.assert_called_once_with("lib-item-123")

        # POST should NOT be called since no changes to start times were needed
        mock_post.assert_not_called()

    @patch("absrefined.client.abs_client.requests.post")
    @patch.object(AudiobookshelfClient, "get_item_chapters")
    def test_update_chapters_start_time_http_error(
        self, mock_get_chapters, mock_post, mock_abs_response
    ):
        """Test updating chapter start times with HTTP error."""
        # Setup mock for get_item_chapters
        original_chapters = [
            {"id": "ch1", "start": 10.0, "end": 20.0, "title": "Chapter 1"},
            {"id": "ch2", "start": 20.0, "end": 30.0, "title": "Chapter 2"},
        ]
        mock_get_chapters.return_value = [chap.copy() for chap in original_chapters]

        # Setup the mock to raise an HTTP error
        http_error = requests.exceptions.HTTPError("500 Server Error")
        mock_post.return_value = mock_abs_response(
            status_code=500, raise_for_status=http_error
        )

        client = AudiobookshelfClient(MOCK_CONFIG)

        chapter_updates = [{"id": "ch1", "start": 12.5}]

        result = client.update_chapters_start_time("lib-item-123", chapter_updates)

        # Should return False on HTTP error
        assert result is False
        mock_get_chapters.assert_called_once_with("lib-item-123")
        mock_post.assert_called_once()

    @patch("absrefined.client.abs_client.requests.get")
    def test_get_item_details_with_all_keys(self, mock_get, mock_abs_response):
        """Test get_item_details with all possible keys present in response."""
        # Setup a mock response that contains all possible keys we want to log
        complete_item_data = {
            "id": "lib-item-123",
            "mediaType": "book",
            "path": "/path/to/audiobook",
            "audioFiles": [{"id": "audio1"}, {"id": "audio2"}],  # Two audio files
            "media": {
                "metadata": {"title": "Test Book", "author": "Test Author"},
                "chapters": [],
            },
        }

        mock_get.return_value = mock_abs_response(json_data=complete_item_data)

        client = AudiobookshelfClient(MOCK_CONFIG)
        result = client.get_item_details("lib-item-123")

        # Verify we got the expected result
        assert result == complete_item_data
        mock_get.assert_called_once()

    @patch("absrefined.client.abs_client.requests.get")
    def test_download_audio_file_with_debug_logs(self, mock_get, mock_abs_response):
        """Test download_audio_file with debug logging."""
        # Create a response with more detailed audio file info
        audio_file_info = {
            "id": "lib-item-123",
            "media": {
                "audioFiles": [
                    {
                        "index": 0,
                        "ino": "audiofile-ino-12345",
                        "relPath": "test.mp3",
                        "duration": 200.0,
                        "bitRate": 128000,
                        "codec": "mp3",
                        "format": "MP3",
                    }
                ]
            },
        }

        # Configure mock to return detailed item info and then successful download
        mock_get_details_response = mock_abs_response(json_data=audio_file_info)
        mock_download_response = mock_abs_response(
            content=b"test audio data", headers={"Content-Length": "14"}
        )

        mock_get.side_effect = [mock_get_details_response, mock_download_response]

        # Create a client with higher log level to test debug logs
        debug_config = {
            "audiobookshelf": {"host": MOCK_SERVER_URL, "api_key": MOCK_API_KEY},
            "logging": {"level": "DEBUG"},
        }

        client = AudiobookshelfClient(debug_config)

        with patch("os.makedirs"):
            with patch("builtins.open", mock_open()) as mock_file:
                result = client.download_audio_file("lib-item-123", "output.mp3")

        assert result == "output.mp3"
        assert mock_get.call_count == 2

    @patch("absrefined.client.abs_client.requests.post")
    @patch.object(AudiobookshelfClient, "get_item_chapters")
    def test_update_chapters_start_time_request_exception_with_response(
        self, mock_get_chapters, mock_post
    ):
        """Test update_chapters_start_time with a request exception that has a response."""
        # Setup mock for get_item_chapters
        original_chapters = [
            {"id": "ch1", "start": 10.0, "end": 20.0, "title": "Chapter 1"},
            {"id": "ch2", "start": 20.0, "end": 30.0, "title": "Chapter 2"},
        ]
        mock_get_chapters.return_value = [chap.copy() for chap in original_chapters]

        # Create a RequestException with a response attribute
        mock_response = MagicMock()
        mock_response.text = "Error updating chapters: Invalid data format"
        request_exception = requests.exceptions.RequestException("Server error")
        request_exception.response = mock_response

        # Configure mock to raise the exception
        mock_post.side_effect = request_exception

        client = AudiobookshelfClient(MOCK_CONFIG)

        # Update with valid start time to ensure POST is called
        valid_updates = [{"id": "ch1", "start": 12.5}]

        result = client.update_chapters_start_time("lib-item-123", valid_updates)

        # Should return False on exception
        assert result is False
        mock_get_chapters.assert_called_once_with("lib-item-123")
        mock_post.assert_called_once()

    @patch("absrefined.client.abs_client.requests.get")
    def test_download_audio_file_no_item_details(self, mock_get):
        """Test download_audio_file when get_item_details returns empty dict."""
        # Mock get_item_details to return empty dict (simulating failure to get details)
        mock_get.return_value = MagicMock(json=MagicMock(return_value={}))

        client = AudiobookshelfClient(MOCK_CONFIG)
        result = client.download_audio_file("lib-item-123", "output.mp3")

        # Should return empty string when get_item_details fails
        assert result == ""
        mock_get.assert_called_once()

    @patch("absrefined.client.abs_client.requests.post")
    @patch.object(AudiobookshelfClient, "get_item_chapters")
    def test_update_chapters_start_time_unexpected_exception(
        self, mock_get_chapters, mock_post
    ):
        """Test update_chapters_start_time with an unexpected exception."""
        # Setup mock for get_item_chapters
        original_chapters = [
            {"id": "ch1", "start": 10.0, "end": 20.0, "title": "Chapter 1"},
            {"id": "ch2", "start": 20.0, "end": 30.0, "title": "Chapter 2"},
        ]
        mock_get_chapters.return_value = [chap.copy() for chap in original_chapters]

        # Configure mock to raise a generic exception
        mock_post.side_effect = Exception("Unexpected error")

        client = AudiobookshelfClient(MOCK_CONFIG)

        # Update with valid start time to ensure POST is called
        valid_updates = [{"id": "ch1", "start": 12.5}]

        result = client.update_chapters_start_time("lib-item-123", valid_updates)

        # Should return False on exception
        assert result is False
        mock_get_chapters.assert_called_once_with("lib-item-123")
        mock_post.assert_called_once()
