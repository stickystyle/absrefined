import pytest
from unittest.mock import patch, MagicMock, call, ANY, mock_open
import json
import os
from pathlib import Path

from absrefined.refinement_tool.chapter_refinement_tool import ChapterRefinementTool
from absrefined.client.abs_client import AudiobookshelfClient
from absrefined.transcriber.audio_transcriber import AudioTranscriber
from absrefined.refiner.chapter_refiner import ChapterRefiner
from absrefined.utils.timestamp import format_timestamp


class TestChapterRefinementTool:
    """Tests for the ChapterRefinementTool class."""
    
    def test_init(self, tmp_path):
        """Test initialization of the ChapterRefinementTool."""
        # Create mocks
        abs_client = MagicMock(spec=AudiobookshelfClient)
        transcriber = MagicMock(spec=AudioTranscriber)
        refiner = MagicMock(spec=ChapterRefiner)
        
        # Test initialization with default parameters
        temp_dir = tmp_path / "temp"
        temp_dir_str = str(temp_dir)
        tool = ChapterRefinementTool(abs_client, transcriber, refiner, 
                                   verbose=True, temp_dir=temp_dir_str)
        
        # Verify attributes
        assert tool.abs_client is abs_client
        assert tool.transcriber is transcriber
        assert tool.refiner is refiner
        assert tool.verbose is True
        assert tool.temp_dir == temp_dir_str
        assert Path(tool.temp_dir).exists()
        assert tool.dry_run is False
        
        # Test initialization with dry_run=True
        tool_dry_run = ChapterRefinementTool(abs_client, transcriber, refiner, 
                                          verbose=True, temp_dir=temp_dir_str, dry_run=True)
        assert tool_dry_run.dry_run is True
    
    def test_extract_item_id_from_url(self):
        """Test extracting item ID from URL."""
        # Create mocks
        abs_client = MagicMock(spec=AudiobookshelfClient)
        transcriber = MagicMock(spec=AudioTranscriber)
        refiner = MagicMock(spec=ChapterRefiner)
        
        # Create tool
        tool = ChapterRefinementTool(abs_client, transcriber, refiner)
        
        # Test with valid URLs
        url = "http://abs-server.com/item/lib-item-123"
        assert tool.extract_item_id_from_url(url) == "lib-item-123"
        
        url = "http://abs-server.com/item/lib-item-123/details"
        assert tool.extract_item_id_from_url(url) == "lib-item-123"
        
        # Test with invalid URL
        url = "http://abs-server.com/not-an-item"
        assert tool.extract_item_id_from_url(url) == ""
    
    def test_process_item(self, tmp_path, abs_item_response, mock_transcript_data):
        """Test processing an item to refine chapter markers."""
        # Create mocks
        abs_client = MagicMock(spec=AudiobookshelfClient)
        transcriber = MagicMock(spec=AudioTranscriber)
        refiner = MagicMock(spec=ChapterRefiner)
        
        # Configure mocks
        abs_client.get_item_details.return_value = abs_item_response
        abs_client.get_item_chapters.return_value = abs_item_response["media"]["chapters"]
        
        # Set up transcriber mock to return sample segments
        temp_jsonl_files = []
        def mock_transcribe_audio(audio_file, output_file=None):
            # Save the output path if provided
            if output_file:
                temp_jsonl_files.append(output_file)
            
            # Get chapter name from audio file
            chapter_name = Path(audio_file).stem
            
            # Return mock segments if available, otherwise empty list
            return mock_transcript_data.get(chapter_name, [])
        
        transcriber.transcribe_audio.side_effect = mock_transcribe_audio
        transcriber.read_transcription.return_value = []
        
        # Configure refiner to return mocked chapter starts
        def mock_detect_chapter_start(segments, chapter_name, chapter_time=None):
            # Return a mock result with original timestamp
            if chapter_name == "Chapter 1":
                return {
                    "timestamp": 291.18,
                    "text": "Chapter 1",
                    "refined": True
                }
            elif chapter_name == "Chapter 2":
                return {
                    "timestamp": 1845.2,
                    "text": "Chapter 2",
                    "refined": True
                }
            else:
                return {
                    "timestamp": 2257.88,
                    "text": "Chapter 3",
                    "refined": True
                }
        
        refiner.detect_chapter_start.side_effect = mock_detect_chapter_start
        
        # Setup abs_client.stream_audio_segment to create temp files
        def mock_stream_audio(item_id, start_time, end_time, output_path):
            # Create an empty file
            Path(output_path).touch()
            return output_path
        
        abs_client.stream_audio_segment.side_effect = mock_stream_audio
        
        # Mock download_audio_file
        def mock_download_audio(item_id, output_path):
            Path(output_path).touch()
            return output_path
            
        abs_client.download_audio_file.side_effect = mock_download_audio
        
        # Create tool with dry_run to avoid prompting for updates
        tool = ChapterRefinementTool(abs_client, transcriber, refiner, temp_dir=str(tmp_path), dry_run=True)

        # --- Mock internal methods to isolate process_item logic --- #
        # Mock the methods called *within* process_item
        original_chapters = abs_item_response["media"]["chapters"]
        refined_chapters_mock_return = [c.copy() for c in original_chapters] # Simulate some result
        for chap in refined_chapters_mock_return:
             chap["start"] = float(chap["start"]) + 1.0 # Simulate refinement

        tool._ensure_full_audio_downloaded = MagicMock(return_value=str(tmp_path / "lib-item-123_full.m4a"))
        tool._get_audio_duration = MagicMock(return_value=3483.66) # Mock duration directly
        tool.process_chapters = MagicMock(return_value=refined_chapters_mock_return)
        tool.compare_and_update = MagicMock(return_value=(True, 3)) # Mock comparison result (updated, count)
        # --- End Mocking internal methods --- #

        # Test process_item
        results = tool.process_item("lib-item-123")

        # Verify results based on mocked compare_and_update
        assert results is not None
        assert results["item_id"] == "lib-item-123"
        assert results["total_chapters"] == 3
        assert results["refined_chapters"] == 3  # From mocked compare_and_update
        assert results["updated_on_server"] is True # From mocked compare_and_update
        assert results["error"] is None

        # Verify calls to mocked internal methods
        abs_client.get_item_chapters.assert_called_once_with("lib-item-123")
        tool._ensure_full_audio_downloaded.assert_called_once_with("lib-item-123")
        tool._get_audio_duration.assert_called_once_with(str(tmp_path / "lib-item-123_full.m4a"))
        tool.process_chapters.assert_called_once_with("lib-item-123", original_chapters, str(tmp_path / "lib-item-123_full.m4a"), 3483.66)
        tool.compare_and_update.assert_called_once_with("lib-item-123", original_chapters, refined_chapters_mock_return, True) # True for dry_run

    @patch("builtins.input", side_effect=["y", "n", "y"]) # Confirm first, deny second, confirm third
    @patch("builtins.print")  # Mock print to prevent output during tests
    def test_compare_and_update(self, mock_print, mock_input):
        """Test comparing original and refined chapters and prompting for updates."""
        # Create mocks
        abs_client = MagicMock(spec=AudiobookshelfClient)
        # Configure abs_client.update_item_chapters to return True
        abs_client.update_item_chapters.return_value = True

        # Dummy transcriber/refiner needed for init, but not used in this test
        transcriber = MagicMock(spec=AudioTranscriber)
        refiner = MagicMock(spec=ChapterRefiner)

        # Create tool
        tool = ChapterRefinementTool(abs_client, transcriber, refiner, verbose=True, dry_run=False)

        # Create test data
        original_chapters = [
            {"id": "ch1", "start": 0.0, "end": 500.0, "title": "Chapter 1"},
            {"id": "ch2", "start": 500.0, "end": 700.0, "title": "Chapter 2"},
            {"id": "ch3", "start": 700.0, "end": 900.0, "title": "Chapter 3"}
        ]

        refined_chapters = [
            {"id": "ch1", "start": 0.0, "end": 500.0, "title": "Chapter 1"}, # No change (first chapter), no flag needed
            {"id": "ch2", "start": 505.5, "end": 700.0, "title": "Chapter 2", "refined": True}, # Significant change, ADD FLAG
            {"id": "ch3", "start": 700.2, "end": 900.0, "title": "Chapter 3"}  # Insignificant change, no flag needed
        ]

        # Test compare_and_update
        updated, refined_count = tool.compare_and_update("lib-item-123", original_chapters, refined_chapters, dry_run=False)

        # Verify results - Expect 1 significant change confirmed
        assert updated is True # Update should have been attempted
        assert refined_count == 1 # refined_count based on input flag, should be 1 now

        # Verify inputs were called (1 significant change > threshold)
        assert mock_input.call_count == 1 # Only prompted for ch2

        # Verify abs_client.update_item_chapters was called once with the correct final list
        # Note: Timestamps are formatted as strings before the update call
        final_expected_chapters = [
            {"id": "ch1", "start": format_timestamp(0.0), "end": 505.5, "title": "Chapter 1"}, # Chapter 1 always 0, end is ch2 start
            {"id": "ch2", "start": format_timestamp(505.5), "end": 700.2, "title": "Chapter 2"}, # Chapter 2 updated (confirmed 'y'), end is ch3 start (using refined time)
            # Chapter 3 keeps refined start (700.2) because the code assigns it regardless of significance/confirmation
            {"id": "ch3", "start": format_timestamp(700.2), "end": 900.0, "title": "Chapter 3"}  # End boundary is original because it's the last chapter
        ]
        abs_client.update_item_chapters.assert_called_once_with("lib-item-123", final_expected_chapters) 