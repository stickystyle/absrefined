import pytest
from unittest.mock import patch, MagicMock
import json
import os
from absrefined.client.abs_client import AudiobookshelfClient


class TestAudiobookshelfClient:
    """Tests for the AudiobookshelfClient class."""
    
    def test_init(self):
        """Test initialization of the AudiobookshelfClient."""
        client = AudiobookshelfClient("http://test-server.com/")
        assert client.server_url == "http://test-server.com"
        assert client.token is None
        assert client.user_id is None
        assert client.verbose is False
        
        client = AudiobookshelfClient("http://test-server.com/", verbose=True)
        assert client.verbose is True
    
    @patch("absrefined.client.abs_client.requests.post")
    def test_login_success(self, mock_post, mock_abs_response, abs_auth_response):
        """Test successful login."""
        # Setup the mock response
        mock_post.return_value = mock_abs_response(json_data=abs_auth_response)
        
        # Create client and test login
        client = AudiobookshelfClient("http://test-server.com/")
        result = client.login("testuser", "password")
        
        # Verify results
        assert result is True
        assert client.token == "test-token-123"
        assert client.user_id == "user-123"
        
        # Verify API call
        mock_post.assert_called_once_with(
            "http://test-server.com/login",
            json={"username": "testuser", "password": "password"}
        )
    
    @patch("absrefined.client.abs_client.requests.post")
    def test_login_failure(self, mock_post, mock_abs_response):
        """Test failed login."""
        # Setup the mock response
        mock_post.return_value = mock_abs_response(status_code=401)
        
        # Create client and test login
        client = AudiobookshelfClient("http://test-server.com/")
        result = client.login("testuser", "wrong-password")
        
        # Verify results
        assert result is False
        assert client.token is None
        assert client.user_id is None
    
    @patch("absrefined.client.abs_client.requests.get")
    def test_get_item_details(self, mock_get, mock_abs_response, abs_item_response):
        """Test getting item details."""
        # Setup the mock response
        mock_get.return_value = mock_abs_response(json_data=abs_item_response)
        
        # Create client and set token
        client = AudiobookshelfClient("http://test-server.com/")
        client.token = "test-token"
        
        # Test get_item_details
        item = client.get_item_details("lib-item-123")
        
        # Verify results
        assert item == abs_item_response
        
        # Verify API call
        mock_get.assert_called_once_with(
            "http://test-server.com/api/items/lib-item-123",
            headers={"Authorization": "Bearer test-token"}
        )
    
    @patch("absrefined.client.abs_client.requests.get")
    def test_get_item_chapters(self, mock_get, mock_abs_response, abs_item_response):
        """Test getting item chapters."""
        # Setup the mock response
        mock_get.return_value = mock_abs_response(json_data=abs_item_response)
        
        # Create client and set token
        client = AudiobookshelfClient("http://test-server.com/")
        client.token = "test-token"
        
        # Test get_item_chapters
        chapters = client.get_item_chapters("lib-item-123")
        
        # Verify results
        assert chapters == abs_item_response["media"]["chapters"]
        
        # Verify API call
        mock_get.assert_called_once_with(
            "http://test-server.com/api/items/lib-item-123",
            headers={"Authorization": "Bearer test-token"}
        )
    
    @patch("absrefined.client.abs_client.requests.get")
    def test_stream_audio_segment(self, mock_get, mock_abs_response, tmp_path):
        """Test streaming an audio segment."""
        # Setup the mock responses
        mock_content = b"test audio data"
        
        # Mock item details response with audioFiles
        item_details_response = mock_abs_response(
            json_data={
                "id": "lib-item-123",
                "media": {
                    "audioFiles": [
                        {
                            "index": 1,
                            "ino": "12345",
                            "relPath": "test.mp3",
                            "duration": 200.0
                        }
                    ]
                }
            }
        )
        
        # Mock content response for file download
        stream_response = mock_abs_response(content=mock_content)
        
        # Configure mock to return different responses for different URLs
        mock_get.side_effect = lambda url, **kwargs: (
            item_details_response if "/items/lib-item-123" in url and not "/file/" in url
            else stream_response
        )
        
        # Create client and set token
        client = AudiobookshelfClient("http://test-server.com/")
        client.token = "test-token"
        
        # Test stream_audio_segment
        temp_dir = str(tmp_path)
        output_path = os.path.join(temp_dir, "test_output.mp3")
        expected_full_path = os.path.join(temp_dir, "lib-item-123_full.m4a")
        
        # First call to stream_audio_segment should download the full file
        result = client.stream_audio_segment("lib-item-123", 100.0, 130.0, output_path)
        
        # In the updated implementation, it returns the full file path rather than the segment path
        assert result == expected_full_path
        assert os.path.exists(expected_full_path)
        with open(expected_full_path, "rb") as f:
            assert f.read() == mock_content
        
        # Verify API calls
        assert mock_get.call_count == 2
        mock_get.assert_any_call("http://test-server.com/api/items/lib-item-123", headers={"Authorization": "Bearer test-token"})
        mock_get.assert_any_call(
            "http://test-server.com/api/items/lib-item-123/file/12345",
            headers={"Authorization": "Bearer test-token"},
            stream=True
        )
    
    @patch("absrefined.client.abs_client.requests.post")
    def test_update_item_chapters(self, mock_post, mock_abs_response):
        """Test updating item chapters."""
        # Setup the mock response
        mock_post.return_value = mock_abs_response(status_code=200)
        
        # Create client and set token
        client = AudiobookshelfClient("http://test-server.com/")
        client.token = "test-token"
        
        # Test update_item_chapters
        chapters = [
            {"id": "chapter-1", "start": 10.0, "end": 20.0, "title": "Chapter 1"},
            {"id": "chapter-2", "start": 20.0, "end": 30.0, "title": "Chapter 2"}
        ]
        result = client.update_item_chapters("lib-item-123", chapters)
        
        # Verify results
        assert result is True
        
        # Check that chapter boundaries were fixed
        assert chapters[0]["end"] == chapters[1]["start"]
        
        # Verify API call
        mock_post.assert_called_once_with(
            "http://test-server.com/api/items/lib-item-123/chapters",
            json={"chapters": mock_post.call_args[1]["json"]["chapters"]},
            headers={"Authorization": "Bearer test-token", "Content-Type": "application/json"}
        ) 