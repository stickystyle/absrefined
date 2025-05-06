import pytest
from unittest.mock import patch, MagicMock
import json
import os
from absrefined.client.abs_client import AudiobookshelfClient


# Mock config for the client
MOCK_SERVER_URL = "http://test-server.com"
MOCK_API_KEY = "test_api_key_123"
MOCK_CONFIG = {
    "audiobookshelf": {
        "host": MOCK_SERVER_URL,
        "api_key": MOCK_API_KEY,
        "timeout": 15
    },
    "logging": {"level": "DEBUG"} # Add a basic logging config
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
            timeout=MOCK_CONFIG["audiobookshelf"]["timeout"]
        )
    
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
            f"{MOCK_SERVER_URL}/api/items/lib-item-123", # Called by get_item_details
            headers={"Authorization": f"Bearer {MOCK_API_KEY}"},
            timeout=MOCK_CONFIG["audiobookshelf"]["timeout"]
        )
    
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
                        "index": 0, # Or 1, depending on server
                        "ino": "audiofile-ino-12345",
                        "relPath": "test.mp3",
                        "duration": 200.0
                    }
                ]
            }
        }
        
        # Mock responses: first for get_item_details, second for file download
        mock_get_item_details_response = mock_abs_response(json_data=item_details_with_audio_info)
        mock_download_stream_response = mock_abs_response(content=mock_audio_content, headers={"Content-Length": str(len(mock_audio_content))})

        # Configure mock to return different responses for different URLs/calls
        # First call is get_item_details, second is the actual download
        mock_get.side_effect = [
            mock_get_item_details_response, # For get_item_details() call within download_audio_file
            mock_download_stream_response   # For the actual file download call
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
            timeout=MOCK_CONFIG["audiobookshelf"]["timeout"]
        )
        # Call 2: actual file download
        mock_get.assert_any_call(
            f"{MOCK_SERVER_URL}/api/items/lib-item-123/file/audiofile-ino-12345",
            headers={"Authorization": f"Bearer {MOCK_API_KEY}"},
            stream=True,
            timeout=MOCK_CONFIG["audiobookshelf"]["timeout"]
        )

    @patch("absrefined.client.abs_client.requests.post")
    @patch.object(AudiobookshelfClient, "get_item_chapters")
    def test_update_chapters_start_time(self, mock_get_chapters, mock_post, mock_abs_response):
        """Test updating chapter start times."""
        # Setup mock for get_item_chapters
        original_chapters = [
            {"id": "ch1", "start": 10.0, "end": 20.0, "title": "Chapter 1"},
            {"id": "ch2", "start": 20.0, "end": 30.0, "title": "Chapter 2"},
            {"id": "ch3", "start": 30.0, "end": 40.0, "title": "Chapter 3"}
        ]
        mock_get_chapters.return_value = [chap.copy() for chap in original_chapters] # Return copies

        # Setup mock for the POST request
        mock_post.return_value = mock_abs_response(status_code=200)
        
        client = AudiobookshelfClient(MOCK_CONFIG)
        
        chapter_updates = [
            {"id": "ch1", "start": 12.5},
            {"id": "ch3", "start": 35.5}
        ]
        
        result = client.update_chapters_start_time("lib-item-123", chapter_updates)
        
        assert result is True
        mock_get_chapters.assert_called_once_with("lib-item-123")
        
        # Expected final chapter list after updates and sorting and end-time fixing
        expected_final_chapters = [
            {"id": "ch1", "start": 12.5, "end": 20.0, "title": "Chapter 1"}, # end will be ch2's start
            {"id": "ch2", "start": 20.0, "end": 35.5, "title": "Chapter 2"}, # end will be ch3's new start
            {"id": "ch3", "start": 35.5, "end": 40.0, "title": "Chapter 3"}  # original end retained for last
        ]
        # Client internal logic re-sorts and fixes end times:
        # ch1: start 12.5, end becomes ch2's start (20.0)
        # ch2: start 20.0 (unchanged), end becomes ch3's new start (35.5)
        # ch3: start 35.5, end remains 40.0 (original end of last chapter)

        # The client sorts by the new start times
        # original: ch1 (10), ch2 (20), ch3 (30)
        # updated:  ch1 (12.5), ch2 (20), ch3 (35.5)
        # sorted:   ch1 (12.5), ch2 (20), ch3 (35.5)
        
        # Recalculate expected_final_chapters based on client logic
        # 1. Apply updates
        # ch1.start = 12.5
        # ch3.start = 35.5
        # 2. Sort by start time (already sorted in this case)
        # ch1: {"id": "ch1", "start": 12.5, "end": 20.0, "title": "Chapter 1"}
        # ch2: {"id": "ch2", "start": 20.0, "end": 30.0, "title": "Chapter 2"}
        # ch3: {"id": "ch3", "start": 35.5, "end": 40.0, "title": "Chapter 3"}
        # 3. Fix end times
        # ch1['end'] = ch2['start'] = 20.0
        # ch2['end'] = ch3['start'] = 35.5
        # ch3['end'] remains 40.0
        
        expected_payload_chapters = [
            {"id": "ch1", "start": 12.5, "end": 20.0, "title": "Chapter 1"},
            {"id": "ch2", "start": 20.0, "end": 35.5, "title": "Chapter 2"},
            {"id": "ch3", "start": 35.5, "end": 40.0, "title": "Chapter 3"}
        ]

        mock_post.assert_called_once_with(
            f"{MOCK_SERVER_URL}/api/items/lib-item-123/chapters",
            json={"chapters": expected_payload_chapters},
            headers={"Authorization": f"Bearer {MOCK_API_KEY}", "Content-Type": "application/json"},
            timeout=MOCK_CONFIG["audiobookshelf"]["timeout"]
        ) 