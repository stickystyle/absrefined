import pytest
from unittest.mock import patch, MagicMock, mock_open
import json
import os
import requests
# New imports for ZIP testing
import io
import zipfile
import tempfile # Though we mock client's tempfile, it's good practice for clarity
import shutil # Though we mock client's shutil, it's good practice for clarity
import subprocess # Though we mock client's subprocess, it's good practice for clarity

from absrefined.client.abs_client import AudiobookshelfClient


# Mock config for the client
MOCK_SERVER_URL = "http://test-server.com"
MOCK_API_KEY = "test_api_key_123"
MOCK_CONFIG = {
    "audiobookshelf": {"host": MOCK_SERVER_URL, "api_key": MOCK_API_KEY, "timeout": 15},
    "logging": {"level": "DEBUG"},  # Add a basic logging config
}

# Helper to create a dummy ZIP file in bytes for mocking downloaded content
def create_dummy_zip_bytes(file_list_with_content: list[tuple[str, str]]) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, content_str in file_list_with_content:
            zf.writestr(filename, content_str.encode('utf-8')) # Store content as bytes
    return zip_buffer.getvalue()

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
        """Test downloading a non-ZIP audio file."""
        mock_audio_content = b"test audio data"
        item_details_with_audio_info = {
            "id": "lib-item-123",
            "media": {
                "audioFiles": [{"ino": "audiofile-ino-12345", "relPath": "test.mp3"}]
            },
        }
        mock_get_item_details_response = mock_abs_response(json_data=item_details_with_audio_info)
        mock_download_stream_response = mock_abs_response(
            content=mock_audio_content,
            headers={"Content-Length": str(len(mock_audio_content)), "Content-Type": "audio/mpeg"} # Specify non-zip type
        )
        mock_get.side_effect = [mock_get_item_details_response, mock_download_stream_response]

        client = AudiobookshelfClient(MOCK_CONFIG)
        output_path = os.path.join(str(tmp_path), "downloaded_audio.mp3")
        
        result_path = client.download_audio_file("lib-item-123", output_path, debug_preserve_files=False)

        assert result_path == output_path
        assert os.path.exists(output_path)
        with open(output_path, "rb") as f:
            assert f.read() == mock_audio_content
        assert mock_get.call_count == 2

    @patch("absrefined.client.abs_client.requests.get")
    def test_download_audio_file_no_content_length(
        self, mock_get, mock_abs_response, tmp_path
    ):
        """Test downloading an audio file with no Content-Length header."""
        mock_audio_content = b"test audio data"
        item_details_with_audio_info = {
            "id": "lib-item-123",
            "media": {"audioFiles": [{"ino": "audiofile-ino-12345"}]},
        }
        mock_get_item_details_response = mock_abs_response(json_data=item_details_with_audio_info)
        mock_download_stream_response = mock_abs_response(content=mock_audio_content, headers={"Content-Type": "audio/mpeg"})
        mock_get.side_effect = [mock_get_item_details_response, mock_download_stream_response]
        client = AudiobookshelfClient(MOCK_CONFIG)
        output_path = os.path.join(str(tmp_path), "downloaded_audio_no_length.mp3")
        
        result_path = client.download_audio_file("lib-item-123", output_path, debug_preserve_files=False)

        assert result_path == output_path
        assert os.path.exists(output_path)
        with open(output_path, "rb") as f:
            assert f.read() == mock_audio_content

    @patch("absrefined.client.abs_client.requests.get")
    def test_download_audio_file_no_audio_files(self, mock_get, mock_abs_response):
        """Test downloading an audio file when no audio files exist."""
        item_details_without_audio = {"id": "lib-item-123", "media": {}}
        mock_get.return_value = mock_abs_response(json_data=item_details_without_audio)
        client = AudiobookshelfClient(MOCK_CONFIG)
        result_path = client.download_audio_file("lib-item-123", "output.mp3", debug_preserve_files=False)
        assert result_path == ""

    @patch("absrefined.client.abs_client.requests.get")
    def test_download_audio_file_empty_audio_files(self, mock_get, mock_abs_response):
        """Test downloading an audio file when audio files array is empty."""
        item_details_with_empty_audio = {"id": "lib-item-123", "media": {"audioFiles": []}}
        mock_get.return_value = mock_abs_response(json_data=item_details_with_empty_audio)
        client = AudiobookshelfClient(MOCK_CONFIG)
        result_path = client.download_audio_file("lib-item-123", "output.mp3", debug_preserve_files=False)
        assert result_path == ""

    @patch("absrefined.client.abs_client.requests.get")
    def test_download_audio_file_missing_ino(self, mock_get, mock_abs_response):
        """Test downloading an audio file when the ino field is missing."""
        item_details_with_missing_ino = {"id": "lib-item-123", "media": {"audioFiles": [{"relPath": "test.mp3"}]}}
        mock_get.return_value = mock_abs_response(json_data=item_details_with_missing_ino)
        client = AudiobookshelfClient(MOCK_CONFIG)
        result_path = client.download_audio_file("lib-item-123", "output.mp3", debug_preserve_files=False)
        assert result_path == ""

    @patch("absrefined.client.abs_client.requests.get")
    def test_download_audio_file_http_error(self, mock_get, mock_abs_response):
        """Test download_audio_file with HTTP error during download."""
        item_details_with_audio = {"id": "lib-item-123", "media": {"audioFiles": [{"ino": "audiofile-ino-12345"}]}}
        mock_get_details_response = mock_abs_response(json_data=item_details_with_audio)
        http_error = requests.exceptions.HTTPError("404 Client Error")
        mock_download_error_response = mock_abs_response(status_code=404, raise_for_status=http_error)
        mock_get.side_effect = [mock_get_details_response, mock_download_error_response]
        client = AudiobookshelfClient(MOCK_CONFIG)
        result_path = client.download_audio_file("lib-item-123", "output.mp3", debug_preserve_files=False)
        assert result_path == ""

    @patch("absrefined.client.abs_client.requests.get")
    def test_download_audio_file_io_error(self, mock_get, mock_abs_response, tmp_path):
        """Test download_audio_file with IO error during file writing for non-ZIP."""
        mock_audio_content = b"test audio data"
        item_details_with_audio = {"id": "lib-item-123", "media": {"audioFiles": [{"ino": "audiofile-ino-12345"}]}}
        mock_get_details_response = mock_abs_response(json_data=item_details_with_audio)
        mock_download_response = mock_abs_response(content=mock_audio_content, headers={"Content-Type": "audio/mpeg"})
        mock_get.side_effect = [mock_get_details_response, mock_download_response]
        client = AudiobookshelfClient(MOCK_CONFIG)
        invalid_path = os.path.join(str(tmp_path), "this_is_a_dir")
        os.makedirs(invalid_path) # Make it a directory to cause IO error on open('wb')

        with patch("builtins.open", side_effect=IOError("Is a directory")):
            result_path = client.download_audio_file("lib-item-123", invalid_path, debug_preserve_files=False)
        assert result_path == ""
    
    @patch("absrefined.client.abs_client.requests.get")
    def test_download_audio_file_with_debug_logs(self, mock_get, mock_abs_response, tmp_path):
        """Test download_audio_file with debug logging (non-ZIP)."""
        audio_file_info = {
            "id": "lib-item-123",
            "media": {"audioFiles": [{"ino": "audiofile-ino-12345", "relPath": "test.mp3"}]},
        }
        mock_get_details_response = mock_abs_response(json_data=audio_file_info)
        mock_download_response = mock_abs_response(content=b"test audio data", headers={"Content-Length": "14", "Content-Type": "audio/mpeg"})
        mock_get.side_effect = [mock_get_details_response, mock_download_response]
        
        # Create a client with higher log level to test debug logs
        debug_config = MOCK_CONFIG.copy() # Already has DEBUG level
        client = AudiobookshelfClient(debug_config)
        
        output_path = str(tmp_path / "output.mp3")

        # Patch os.makedirs as it's called before open
        with patch("absrefined.client.abs_client.os.makedirs"):
            # Patch builtins.open since we are testing the non-zip path here
            with patch("builtins.open", mock_open()) as mock_file_open:
                result = client.download_audio_file("lib-item-123", output_path, debug_preserve_files=False)

        assert result == output_path
        mock_file_open.assert_called_with(output_path, "wb")

    @patch("absrefined.client.abs_client.requests.get")
    def test_download_audio_file_no_item_details(self, mock_get, mock_abs_response):
        """Test download_audio_file when get_item_details returns empty dict."""
        # Mock get_item_details to return empty dict by having response.json() be empty
        mock_get.return_value = mock_abs_response(json_data={}) # This makes get_item_details return {}
        
        client = AudiobookshelfClient(MOCK_CONFIG)
        result = client.download_audio_file("lib-item-123", "output.mp3", debug_preserve_files=False)
        assert result == ""

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
    def test_download_audio_file_no_item_details(self, mock_get, mock_abs_response):
        """Test download_audio_file when get_item_details returns empty dict."""
        # Mock get_item_details to return empty dict by having response.json() be empty
        mock_get.return_value = mock_abs_response(json_data={}) # This makes get_item_details return {}
        
        client = AudiobookshelfClient(MOCK_CONFIG)
        result = client.download_audio_file("lib-item-123", "output.mp3", debug_preserve_files=False)
        assert result == ""

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

    @patch("absrefined.client.abs_client.shutil.rmtree")
    @patch("absrefined.client.abs_client.subprocess.run")
    @patch("builtins.open", new_callable=mock_open)
    @patch("absrefined.client.abs_client.os.path.isfile")
    @patch("absrefined.client.abs_client.os.listdir")
    @patch("absrefined.client.abs_client.zipfile.ZipFile")
    @patch("absrefined.client.abs_client.tempfile.mkdtemp")
    @patch("absrefined.client.abs_client.requests.get")
    @patch("absrefined.client.abs_client.shutil.copy")
    def test_download_audio_file_zip_success_debug_preserve(
        self, mock_shutil_copy, mock_requests_get, mock_mkdtemp, mock_zipfile_class, mock_os_listdir, 
        mock_os_path_isfile,
        mock_builtin_open_instance, mock_subprocess_run, mock_shutil_rmtree, 
        mock_abs_response, tmp_path
    ):
        """Test ZIP success with debug_preserve_files=True (no cleanup)."""
        item_id = "zip-debug-item"
        output_path = str(tmp_path / "concatenated_audio_debug.mp3")
        temp_processing_dir = str(tmp_path / "zip_processing_debug")
        extracted_dir = os.path.join(temp_processing_dir, "extracted")
        test_audio_file = os.path.join(extracted_dir, "t1.mp3")
        
        # Basic setup for mocks
        item_details = {"id": item_id, "media": {"audioFiles": [{"ino": "zip-debug-ino"}]}}
        mock_item_details_resp = mock_abs_response(json_data=item_details)
        zip_content_bytes = create_dummy_zip_bytes([("t1.mp3", "d1")])
        mock_zip_download_resp = mock_abs_response(content=zip_content_bytes, headers={"Content-Type": "application/zip"})
        mock_requests_get.side_effect = [mock_item_details_resp, mock_zip_download_resp]

        mock_mkdtemp.return_value = temp_processing_dir
        mock_zip_instance = MagicMock()
        mock_zipfile_class.return_value.__enter__.return_value = mock_zip_instance
        
        # Mock os.listdir to return our audio file name
        mock_os_listdir.return_value = ["t1.mp3"]
        
        # Mock os.path.isfile to return True for our audio file
        mock_os_path_isfile.return_value = True
        
        # Mock shutil.copy to do nothing but succeed
        mock_shutil_copy.return_value = None
        
        # Set up open mock to handle file operations
        mock_file = mock_builtin_open_instance.return_value
        mock_file.write = lambda data: len(data) if isinstance(data, bytes) else len(data.encode())

        # Create our client and run the download
        client = AudiobookshelfClient(MOCK_CONFIG)
        with patch("absrefined.client.abs_client.os.path.abspath", side_effect=lambda p: p):
            result_path = client.download_audio_file(item_id, output_path, debug_preserve_files=True)
        
        # Verify the result
        assert result_path == output_path
        
        # Verify API calls
        mock_requests_get.assert_any_call(f"{MOCK_SERVER_URL}/api/items/{item_id}", headers={"Authorization": f"Bearer {MOCK_API_KEY}"}, timeout=MOCK_CONFIG["audiobookshelf"]["timeout"])
        mock_requests_get.assert_any_call(f"{MOCK_SERVER_URL}/api/items/{item_id}/download", stream=True, timeout=MOCK_CONFIG["audiobookshelf"]["timeout"], params={"token": MOCK_API_KEY})
        
        # Verify shutil.copy was called with the right paths
        mock_shutil_copy.assert_called_with(test_audio_file, output_path)
        
        # Since debug_preserve_files=True, verify temp directory is not removed
        mock_shutil_rmtree.assert_not_called()

    @patch("absrefined.client.abs_client.shutil.rmtree")
    @patch("absrefined.client.abs_client.subprocess.run")
    @patch("builtins.open", new_callable=mock_open)
    @patch("absrefined.client.abs_client.os.path.isfile")
    @patch("absrefined.client.abs_client.os.listdir")
    @patch("absrefined.client.abs_client.zipfile.ZipFile")
    @patch("absrefined.client.abs_client.tempfile.mkdtemp")
    @patch("absrefined.client.abs_client.requests.get")
    def test_download_audio_file_zip_ffmpeg_fails(
        self, mock_requests_get, mock_mkdtemp, mock_zipfile_class, mock_os_listdir, 
        mock_os_path_isfile,
        mock_builtin_open_instance, mock_subprocess_run, mock_shutil_rmtree, 
        mock_abs_response, tmp_path
    ):
        """Test ZIP handling when ffmpeg concatenation fails."""
        item_id = "zip-ffmpegfail-item"
        output_path = str(tmp_path / "concat_ffmpeg_fail.mp3")
        temp_processing_dir = str(tmp_path / "zip_ffmpeg_processing")

        item_details = {"id": item_id, "media": {"audioFiles": [{"ino": "zip-ffmpeg-ino"}]}}
        mock_item_details_resp = mock_abs_response(json_data=item_details)
        zip_content_bytes = create_dummy_zip_bytes([("trackA.mp3", "audioA"), ("trackB.mp3", "audioB")])
        mock_zip_download_resp = mock_abs_response(content=zip_content_bytes, headers={"Content-Type": "application/zip"})
        mock_requests_get.side_effect = [mock_item_details_resp, mock_zip_download_resp]
        
        mock_mkdtemp.return_value = temp_processing_dir
        mock_zip_instance = MagicMock()
        mock_zipfile_class.return_value.__enter__.return_value = mock_zip_instance
        mock_os_listdir.return_value = ["trackA.mp3", "trackB.mp3"]
        mock_os_path_isfile.return_value = True

        # Set up the mock for file writing
        mock_file = mock_builtin_open_instance.return_value
        mock_file.write = lambda data: len(data) if isinstance(data, bytes) else len(data.encode())

        # Configure ffmpeg to fail
        mock_subprocess_run.return_value = MagicMock(returncode=1, stdout="ffmpeg output", stderr="ffmpeg error details")
        
        client = AudiobookshelfClient(MOCK_CONFIG)
        result_path = client.download_audio_file(item_id, output_path, debug_preserve_files=False)

        # ffmpeg failure should return an empty string
        assert result_path == ""
        # Verify ffmpeg was called
        mock_subprocess_run.assert_called_once()
        # Verify cleanup happens even when ffmpeg fails
        mock_shutil_rmtree.assert_called_once()

    @patch("absrefined.client.abs_client.shutil.rmtree")
    @patch("absrefined.client.abs_client.os.path.isfile")
    @patch("absrefined.client.abs_client.os.listdir")
    @patch("absrefined.client.abs_client.zipfile.ZipFile")
    @patch("absrefined.client.abs_client.tempfile.mkdtemp")
    @patch("absrefined.client.abs_client.requests.get")
    def test_download_audio_file_zip_no_audio_files_in_zip(
        self, mock_requests_get, mock_mkdtemp, mock_zipfile_class, mock_os_listdir,
        mock_os_path_isfile,
        mock_shutil_rmtree, mock_abs_response, tmp_path
    ):
        """Test ZIP handling when no audio files are found after extraction."""
        item_id = "zip-noaudio-item"
        output_path = str(tmp_path / "concat_noaudio.mp3")
        temp_processing_dir = str(tmp_path / "zip_noaudio_processing")

        item_details = {"id": item_id, "media": {"audioFiles": [{"ino": "zip-noaudio-ino"}]}}
        mock_item_details_resp = mock_abs_response(json_data=item_details)
        zip_content_bytes = create_dummy_zip_bytes([("readme.txt", "info only")])
        mock_zip_download_resp = mock_abs_response(content=zip_content_bytes, headers={"Content-Type": "application/zip"})
        mock_requests_get.side_effect = [mock_item_details_resp, mock_zip_download_resp]
        
        mock_mkdtemp.return_value = temp_processing_dir
        mock_zip_instance = MagicMock()
        mock_zipfile_class.return_value.__enter__.return_value = mock_zip_instance
        mock_os_listdir.return_value = ["readme.txt"]
        mock_os_path_isfile.return_value = True

        client = AudiobookshelfClient(MOCK_CONFIG)
        result_path = client.download_audio_file(item_id, output_path, debug_preserve_files=False)

        assert result_path == ""
        mock_shutil_rmtree.assert_called_once_with(temp_processing_dir)

    @patch("absrefined.client.abs_client.shutil.rmtree")
    @patch("absrefined.client.abs_client.subprocess.run")
    @patch("builtins.open", new_callable=mock_open)
    @patch("absrefined.client.abs_client.os.path.isfile")
    @patch("absrefined.client.abs_client.os.listdir")
    @patch("absrefined.client.abs_client.zipfile.ZipFile")
    @patch("absrefined.client.abs_client.tempfile.mkdtemp")
    @patch("absrefined.client.abs_client.requests.get")
    def test_download_audio_file_zip_success(
        self, mock_requests_get, mock_mkdtemp, mock_zipfile_class, mock_os_listdir, 
        mock_os_path_isfile,
        mock_builtin_open_instance, mock_subprocess_run, mock_shutil_rmtree, 
        mock_abs_response, tmp_path
    ):
        """Test successful download, extraction, and concatenation of a ZIP file."""
        item_id = "zip-item-123"
        output_path = str(tmp_path / "concatenated_audio.mp3")
        temp_processing_dir = str(tmp_path / "zip_processing")

        item_details = {"id": item_id, "media": {"audioFiles": [{"ino": "zip-ino-456"}]}}
        mock_item_details_resp = mock_abs_response(json_data=item_details)

        dummy_files_in_zip = [("track02.mp3", "data2"), ("track01.mp3", "data1")]
        sorted_dummy_files = sorted(dummy_files_in_zip, key=lambda x: x[0])
        zip_content_bytes = create_dummy_zip_bytes(dummy_files_in_zip)
        mock_zip_download_resp = mock_abs_response(
            content=zip_content_bytes,
            headers={"Content-Type": "application/zip", "Content-Length": str(len(zip_content_bytes))}
        )
        mock_requests_get.side_effect = [mock_item_details_resp, mock_zip_download_resp]

        mock_mkdtemp.return_value = temp_processing_dir

        mock_zip_instance = MagicMock()
        mock_zipfile_class.return_value.__enter__.return_value = mock_zip_instance

        mock_os_listdir.return_value = [fname for fname, _ in sorted_dummy_files]
        mock_os_path_isfile.return_value = True

        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        
        # Set up the open mock to properly handle file I/O
        mock_file = mock_builtin_open_instance.return_value
        mock_file.write = lambda data: len(data) if isinstance(data, bytes) else len(data.encode())

        client = AudiobookshelfClient(MOCK_CONFIG)
        result_path = client.download_audio_file(item_id, output_path, debug_preserve_files=False)

        assert result_path == output_path
        auth_headers = {"Authorization": f"Bearer {MOCK_API_KEY}"}
        # Check that both the item_details and the file download are called with correct URLs
        mock_requests_get.assert_any_call(f"{MOCK_SERVER_URL}/api/items/{item_id}", headers=auth_headers, timeout=MOCK_CONFIG["audiobookshelf"]["timeout"])
        mock_requests_get.assert_any_call(f"{MOCK_SERVER_URL}/api/items/{item_id}/download", stream=True, timeout=MOCK_CONFIG["audiobookshelf"]["timeout"], params={"token": MOCK_API_KEY})
        mock_mkdtemp.assert_called_once_with(prefix=f"abs_zip_{item_id}_", dir=str(tmp_path))
        
        downloaded_zip_path = os.path.join(temp_processing_dir, f"{item_id}_source.zip")
        mock_builtin_open_instance.assert_any_call(downloaded_zip_path, "wb")

        mock_zipfile_class.assert_called_with(downloaded_zip_path, 'r')
        extracted_files_dir = os.path.join(temp_processing_dir, "extracted")
        mock_zip_instance.extractall.assert_called_with(extracted_files_dir)
        
        mock_os_listdir.assert_called_with(extracted_files_dir)
        for fname, _ in sorted_dummy_files:
            mock_os_path_isfile.assert_any_call(os.path.join(extracted_files_dir, fname))

        ffmpeg_list_path = os.path.join(temp_processing_dir, "ffmpeg_concat_list.txt")
        mock_builtin_open_instance.assert_any_call(ffmpeg_list_path, "w", encoding="utf-8")
        
        expected_ffmpeg_cmd = [
            "ffmpeg", "-y", 
            "-f", "concat",
            "-safe", "0", 
            "-i", ffmpeg_list_path,
            "-map", "0a",
            "-c:a", "acc",
            "-ac", "2",
            "-b:a", "128k",
            output_path
        ]
        mock_subprocess_run.assert_called_once_with(expected_ffmpeg_cmd, capture_output=True, text=True, check=False, encoding='utf-8')
        
        mock_shutil_rmtree.assert_called_once_with(temp_processing_dir)
