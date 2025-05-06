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


# Mock config for the ChapterRefinementTool and its components
MOCK_TOOL_PROCESSING_DOWNLOAD_PATH = "test_tool_temp_audio"
MOCK_TOOL_LOGGING_DEBUG_FILES = False
MOCK_TOOL_REFINER_API_URL = "http://localhost:1111/v1"
MOCK_TOOL_REFINER_API_KEY = "key_for_tool_refiner_transcriber"
MOCK_TOOL_REFINER_MODEL_NAME = "tool-refiner-model"
MOCK_TOOL_WHISPER_MODEL_NAME = "tool-whisper-model"

MOCK_TOOL_CONFIG = {
    "processing": {
        "download_path": MOCK_TOOL_PROCESSING_DOWNLOAD_PATH,
        "search_window_seconds": 30 # Default, can be overridden in tests
    },
    "logging": {
        "level": "DEBUG",
        "debug_files": MOCK_TOOL_LOGGING_DEBUG_FILES
    },
    "refiner": { # For internal Transcriber and Refiner
        "openai_api_url": MOCK_TOOL_REFINER_API_URL,
        "openai_api_key": MOCK_TOOL_REFINER_API_KEY,
        "model_name": MOCK_TOOL_REFINER_MODEL_NAME,
        "whisper_model_name": MOCK_TOOL_WHISPER_MODEL_NAME
    },
    "costs": {} # Add empty costs section for completeness
}


class TestChapterRefinementTool:
    """Tests for the ChapterRefinementTool class."""
    
    @patch("absrefined.refinement_tool.chapter_refinement_tool.AudioTranscriber")
    @patch("absrefined.refinement_tool.chapter_refinement_tool.ChapterRefiner")
    @patch("shutil.which", return_value="ffmpeg_path") # Mock ffmpeg check
    def test_init(self, mock_shutil_which, MockChapterRefiner, MockAudioTranscriber, tmp_path):
        """Test initialization of the ChapterRefinementTool."""
        mock_abs_client = MagicMock(spec=AudiobookshelfClient)
        
        # Create a specific config for this test to override temp_dir to use tmp_path
        test_specific_config = MOCK_TOOL_CONFIG.copy()
        test_specific_config["processing"] = test_specific_config["processing"].copy()
        test_specific_config["processing"]["download_path"] = str(tmp_path / "tool_temp")
        test_specific_config["logging"] = {"level": "DEBUG", "debug_files": True}

        tool = ChapterRefinementTool(config=test_specific_config, client=mock_abs_client)
        
        assert tool.config == test_specific_config
        assert tool.abs_client is mock_abs_client
        assert tool.transcriber is MockAudioTranscriber.return_value
        assert tool.refiner is MockChapterRefiner.return_value
        MockAudioTranscriber.assert_called_once_with(config=test_specific_config)
        MockChapterRefiner.assert_called_once_with(config=test_specific_config)
        
        assert tool.base_temp_dir == str(tmp_path / "tool_temp")
        assert Path(tool.base_temp_dir).exists() # Ensure tool creates its temp dir
        assert tool.debug_preserve_files is True # From test_specific_config
        # dry_run is not an init parameter anymore, it's passed to methods like process_item

    @patch("shutil.which", return_value=None) # Mock ffmpeg check to simulate not found
    @patch("absrefined.refinement_tool.chapter_refinement_tool.AudiobookshelfClient") # Mock client still needed for init
    def test_init_ffmpeg_not_found(self, MockAbsClient, mock_shutil_which):
        """Test ChapterRefinementTool initialization when ffmpeg is not found."""
        mock_abs_client_instance = MockAbsClient()
        with pytest.raises(OSError) as excinfo:
            ChapterRefinementTool(config=MOCK_TOOL_CONFIG, client=mock_abs_client_instance)
        assert "ffmpeg not found. Cannot proceed." in str(excinfo.value)

    @patch("absrefined.refinement_tool.chapter_refinement_tool.AudioTranscriber")
    @patch("absrefined.refinement_tool.chapter_refinement_tool.ChapterRefiner")
    @patch("shutil.which", return_value="ffmpeg_path") # Mock ffmpeg check
    def test_extract_item_id_from_url(self, mock_shutil_which, MockChapterRefiner, MockAudioTranscriber):
        """Test extracting item ID from URL."""
        mock_abs_client = MagicMock(spec=AudiobookshelfClient)
        
        # Tool needs to be initialized, even if its complex parts aren't used by this method
        tool = ChapterRefinementTool(config=MOCK_TOOL_CONFIG, client=mock_abs_client)
        
        # Test with valid URLs
        url = "http://abs-server.com/item/lib-item-123"
        assert tool.extract_item_id_from_url(url) == "lib-item-123"
        
        url = "http://abs-server.com/item/lib-item-123/details"
        assert tool.extract_item_id_from_url(url) == "lib-item-123"
        
        # Test with invalid URL
        url = "http://abs-server.com/not-an-item"
        assert tool.extract_item_id_from_url(url) == ""
    
    @patch("absrefined.refinement_tool.chapter_refinement_tool.AudiobookshelfClient")
    @patch("absrefined.refinement_tool.chapter_refinement_tool.AudioTranscriber")
    @patch("absrefined.refinement_tool.chapter_refinement_tool.ChapterRefiner")
    @patch("shutil.which", return_value="ffmpeg_path") # Mock ffmpeg check
    def test_process_item(self, mock_shutil_which, MockChapterRefiner, MockAudioTranscriber, MockAbsClient, 
                          tmp_path, abs_item_response, mock_transcript_data):
        """Test processing an item to refine chapter markers."""
        # Configure instances that will be created inside ChapterRefinementTool
        mock_abs_client_instance = MockAbsClient.return_value
        mock_transcriber_instance = MockAudioTranscriber.return_value
        mock_refiner_instance = MockChapterRefiner.return_value

        # Configure mock behaviors for client methods
        mock_abs_client_instance.get_item_chapters.return_value = abs_item_response["media"]["chapters"]
        # _ensure_full_audio_downloaded calls abs_client.download_audio_file internally if file not found.
        # For this test, let's mock _ensure_full_audio_downloaded directly to simplify, 
        # or ensure download_audio_file is robustly mocked.
        # To keep closer to the original test structure which mocked internal methods of the tool:
        # We will mock the tool's internal methods for now.

        # Create tool - Transcriber and Refiner are created internally using their (mocked) classes
        # The client instance passed to the tool is our top-level mock_abs_client_instance
        tool = ChapterRefinementTool(config=MOCK_TOOL_CONFIG, client=mock_abs_client_instance)

        # --- Mock internal methods OF THE TOOL to isolate process_item logic --- #
        original_chapters = abs_item_response["media"]["chapters"]
        
        # Mock return from tool.process_chapters
        # This list should contain dicts with "original_start" and "refined_start" keys.
        processed_chapters_output = []
        original_chapter_data = abs_item_response["media"]["chapters"]
        significant_change_threshold = 0.1 # From process_item

        # Chapter 0: No refinement (refined_start is same as original_start or None)
        chap0_data = original_chapter_data[0].copy()
        chap0_orig_start_float = float(chap0_data["start"])
        processed_chapters_output.append({
            **chap0_data, 
            "original_start": chap0_orig_start_float,
            "refined_start": chap0_orig_start_float, # Same as original, so not counted as refined by process_item
            "usage_data": None, "chunk_duration_seconds": 0 # Add other expected keys for loop in process_item
        })

        # Chapter 1: Refined (difference > threshold)
        chap1_data = original_chapter_data[1].copy()
        chap1_orig_start_float = float(chap1_data["start"])
        chap1_refined_start_float = chap1_orig_start_float + (significant_change_threshold + 0.5) # Clearly different and > threshold
        processed_chapters_output.append({
            **chap1_data,
            "original_start": chap1_orig_start_float,
            "refined_start": chap1_refined_start_float,
            "usage_data": {"prompt_tokens":1, "completion_tokens":1, "total_tokens":2}, "chunk_duration_seconds": 10
        })

        # Chapter 2: Refined (difference > threshold)
        chap2_data = original_chapter_data[2].copy()
        chap2_orig_start_float = float(chap2_data["start"])
        chap2_refined_start_float = chap2_orig_start_float + (significant_change_threshold + 1.0) # Clearly different and > threshold
        processed_chapters_output.append({
            **chap2_data,
            "original_start": chap2_orig_start_float,
            "refined_start": chap2_refined_start_float,
            "usage_data": {"prompt_tokens":1, "completion_tokens":1, "total_tokens":2}, "chunk_duration_seconds": 10
        })
        
        expected_chapters_to_be_updated_count = 2 

        mock_full_audio_path = str(tmp_path / "lib-item-123_full.m4a")
        Path(mock_full_audio_path).touch()
        tool._ensure_full_audio_downloaded = MagicMock(return_value=mock_full_audio_path)
        tool._get_audio_duration = MagicMock(return_value=3483.66)
        tool.process_chapters = MagicMock(return_value=processed_chapters_output)
        # tool.compare_and_update is removed, so no mock for it.
        # --- End Mocking internal methods --- #

        # Test process_item - dry_run is now a parameter to process_item
        # search_window_seconds also comes from config or can be overridden via param
        search_window_param = MOCK_TOOL_CONFIG["processing"]["search_window_seconds"]
        model_override_param = None
        dry_run_param = True

        results = tool.process_item(
            item_id="lib-item-123", 
            search_window_seconds=search_window_param,
            model_name_override=model_override_param,
            dry_run=dry_run_param
        )

        assert results is not None
        assert results["item_id"] == "lib-item-123"
        assert results["total_chapters"] == len(original_chapters)
        # "refined_chapters" key in result dict stores the count of chapters that would be/were updated.
        assert results["refined_chapters"] == expected_chapters_to_be_updated_count
        # In dry_run=True, "updated_on_server" key is not expected to be in results dict by current app logic
        assert "updated_on_server" not in results 
        assert results["error"] is None

        # Verify calls to mocked tool internal methods
        mock_abs_client_instance.get_item_chapters.assert_called_once_with("lib-item-123")
        tool._ensure_full_audio_downloaded.assert_called_once_with("lib-item-123")
        tool._get_audio_duration.assert_called_once_with(mock_full_audio_path)
        tool.process_chapters.assert_called_once_with(
            item_id="lib-item-123", 
            original_chapters=original_chapters, 
            full_audio_path=mock_full_audio_path, 
            audio_duration=3483.66,
            search_window_seconds=search_window_param,
            model_name_override=model_override_param
        )
        # tool.compare_and_update call is removed.

    # Target for deletion begins here (comment and method):
    # Removing the following test as compare_and_update method no longer exists on ChapterRefinementTool.
    # @patch("absrefined.refinement_tool.chapter_refinement_tool.AudioTranscriber") 
    # @patch("absrefined.refinement_tool.chapter_refinement_tool.ChapterRefiner")  
    # @patch("shutil.which", return_value="ffmpeg_path") 
    # @patch("builtins.input") 
    # @patch("builtins.print")
    # def test_compare_and_update(self, mock_print, mock_input, mock_shutil_which, 
    #                             MockChapterRefiner, MockAudioTranscriber, 
    #                             abs_item_response): 
    #     """Test comparing original and refined chapters and prompting for updates."""
    #     # ... (entire method content was here)
    #     pass # Original content removed 
    # End of target for deletion 

    # --- Tests for _ensure_full_audio_downloaded --- #

    @patch("shutil.which", return_value="ffmpeg_path")
    @patch("absrefined.refinement_tool.chapter_refinement_tool.os.path.exists")
    def test_ensure_full_audio_downloaded_exists_valid_m4a(self, mock_os_path_exists, mock_shutil_which, tmp_path):
        mock_abs_client = MagicMock(spec=AudiobookshelfClient)
        test_config = MOCK_TOOL_CONFIG.copy()
        mock_download_dir_str = str(tmp_path / "test_downloads")
        test_config["processing"]["download_path"] = mock_download_dir_str
        Path(mock_download_dir_str).mkdir(parents=True, exist_ok=True)
        tool = ChapterRefinementTool(config=test_config, client=mock_abs_client)
        item_id = "test_item_m4a_exists"
        m4a_path_str = os.path.join(mock_download_dir_str, f"{item_id}_full_audio.m4a")
        # Phantom (T) + Method: .m4a (T), .m4a (T) -> 3 calls total
        mock_os_path_exists.side_effect = [True, True, True] 
        result_path = tool._ensure_full_audio_downloaded(item_id)
        assert result_path == m4a_path_str
        expected_calls_in_method = [call(m4a_path_str), call(m4a_path_str)]
        mock_os_path_exists.assert_has_calls(expected_calls_in_method, any_order=False)
        assert mock_os_path_exists.call_count == 3
        mock_abs_client.download_audio_file.assert_not_called()

    @patch("shutil.which", return_value="ffmpeg_path")
    @patch("absrefined.refinement_tool.chapter_refinement_tool.os.path.exists")
    def test_ensure_full_audio_downloaded_exists_valid_other_ext(self, mock_os_path_exists, mock_shutil_which, tmp_path):
        mock_abs_client = MagicMock(spec=AudiobookshelfClient)
        test_config = MOCK_TOOL_CONFIG.copy()
        mock_download_dir_str = str(tmp_path / "test_downloads")
        test_config["processing"]["download_path"] = mock_download_dir_str
        Path(mock_download_dir_str).mkdir(parents=True, exist_ok=True)
        tool = ChapterRefinementTool(config=test_config, client=mock_abs_client)
        item_id = "test_item_mp3_exists"
        m4a_path_str = os.path.join(mock_download_dir_str, f"{item_id}_full_audio.m4a")
        mp3_path_str = os.path.join(mock_download_dir_str, f"{item_id}_full_audio.mp3")
        # Phantom (T) + Method: .m4a (F), .mp3_loop (T), .mp3_after_loop (T) -> 4 calls
        mock_os_path_exists.side_effect = [True, False, True, True]
        result_path = tool._ensure_full_audio_downloaded(item_id)
        assert result_path == mp3_path_str, f"Expected {mp3_path_str}, got {result_path}"
        expected_calls_in_method = [call(m4a_path_str), call(mp3_path_str), call(mp3_path_str)]
        mock_os_path_exists.assert_has_calls(expected_calls_in_method, any_order=False)
        assert mock_os_path_exists.call_count == 4 
        mock_abs_client.download_audio_file.assert_not_called()

    @patch("shutil.which", return_value="ffmpeg_path")
    @patch("absrefined.refinement_tool.chapter_refinement_tool.os.path.exists")
    def test_ensure_full_audio_downloaded_does_not_exist_success(self, mock_os_path_exists, mock_shutil_which, tmp_path):
        mock_abs_client = MagicMock(spec=AudiobookshelfClient)
        test_config = MOCK_TOOL_CONFIG.copy()
        mock_download_dir_str = str(tmp_path / "test_downloads")
        test_config["processing"]["download_path"] = mock_download_dir_str
        Path(mock_download_dir_str).mkdir(parents=True, exist_ok=True)
        tool = ChapterRefinementTool(config=test_config, client=mock_abs_client)
        item_id = "test_item_needs_download"
        m4a_path_str = os.path.join(mock_download_dir_str, f"{item_id}_full_audio.m4a")
        other_ext_paths_for_expected_calls = [os.path.join(mock_download_dir_str, f"{item_id}_full_audio.{ext}") for ext in ["mp3", "ogg", "opus", "flac", "wav", "m4b"]]
        mock_abs_client.download_audio_file.return_value = m4a_path_str

        # Phantom (True) + 8 Pre-download Falses + 1 Post-download True = 10 calls total
        mock_os_path_exists.side_effect = [True] + [False]*8 + [True]

        result_path = tool._ensure_full_audio_downloaded(item_id)
        assert result_path == m4a_path_str, f"Expected {m4a_path_str}, got {result_path}"
        
        assert mock_os_path_exists.call_count == 10
        expected_calls_in_method = [call(m4a_path_str)] 
        for p_ext in other_ext_paths_for_expected_calls: 
            expected_calls_in_method.append(call(p_ext))
        expected_calls_in_method.append(call(m4a_path_str)) 
        expected_calls_in_method.append(call(m4a_path_str)) 
        mock_os_path_exists.assert_has_calls(expected_calls_in_method, any_order=False)
        mock_abs_client.download_audio_file.assert_called_once_with(item_id, m4a_path_str)
    
    # Add a test for download failure where client returns None
    @patch("shutil.which", return_value="ffmpeg_path")
    @patch("absrefined.refinement_tool.chapter_refinement_tool.os.path.exists")
    def test_ensure_full_audio_downloaded_download_client_fails(self, mock_os_path_exists, mock_shutil_which, tmp_path):
        mock_abs_client = MagicMock(spec=AudiobookshelfClient)
        test_config = MOCK_TOOL_CONFIG.copy()
        mock_download_dir_str = str(tmp_path / "test_downloads")
        test_config["processing"]["download_path"] = mock_download_dir_str
        Path(mock_download_dir_str).mkdir(parents=True, exist_ok=True)
        tool = ChapterRefinementTool(config=test_config, client=mock_abs_client)
        item_id = "test_item_client_fail"
        m4a_path_str = os.path.join(mock_download_dir_str, f"{item_id}_full_audio.m4a")

        # Mock client to simulate download failure (returns None)
        mock_abs_client.download_audio_file.return_value = None 

        # Phantom (T) + 8 Pre-download Falses. Post-download check on path won't happen if downloaded_path is None.
        # The method returns None before the final os.path.exists if downloaded_path is None.
        # So, 1 (Phantom) + 8 (Pre-download logic calls) = 9 calls to os.path.exists
        mock_os_path_exists.side_effect = [True] + [False]*8 

        result_path = tool._ensure_full_audio_downloaded(item_id)
        assert result_path is None
        
        assert mock_os_path_exists.call_count == 9 # 1 phantom + 8 method logic
        mock_abs_client.download_audio_file.assert_called_once_with(item_id, m4a_path_str)

    # Add a test for download success but file still not found post-check
    @patch("shutil.which", return_value="ffmpeg_path")
    @patch("absrefined.refinement_tool.chapter_refinement_tool.os.path.exists")
    def test_ensure_full_audio_downloaded_file_vanishes_after_download(self, mock_os_path_exists, mock_shutil_which, tmp_path):
        mock_abs_client = MagicMock(spec=AudiobookshelfClient)
        test_config = MOCK_TOOL_CONFIG.copy()
        mock_download_dir_str = str(tmp_path / "test_downloads")
        test_config["processing"]["download_path"] = mock_download_dir_str
        Path(mock_download_dir_str).mkdir(parents=True, exist_ok=True)
        tool = ChapterRefinementTool(config=test_config, client=mock_abs_client)
        item_id = "test_item_vanishes"
        m4a_path_str = os.path.join(mock_download_dir_str, f"{item_id}_full_audio.m4a")

        # Client download_audio_file returns the path, simulating success
        mock_abs_client.download_audio_file.return_value = m4a_path_str

        # Phantom (T) + 8 Pre-download Falses + 1 Post-download False (file vanished)
        mock_os_path_exists.side_effect = [True] + [False]*8 + [False] 

        result_path = tool._ensure_full_audio_downloaded(item_id)
        assert result_path is None
        
        assert mock_os_path_exists.call_count == 10 # 1 phantom + 9 method logic
        mock_abs_client.download_audio_file.assert_called_once_with(item_id, m4a_path_str)