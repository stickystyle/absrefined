import pytest
from unittest.mock import patch, MagicMock, call
import os
from pathlib import Path
import logging
import subprocess

from absrefined.refinement_tool.chapter_refinement_tool import ChapterRefinementTool
from absrefined.client.abs_client import AudiobookshelfClient
from absrefined.transcriber.audio_transcriber import AudioTranscriber
from absrefined.refiner.chapter_refiner import ChapterRefiner
from absrefined.utils.url_utils import extract_item_id_from_url  # Added import


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
        "search_window_seconds": 30,  # Default, can be overridden in tests
    },
    "logging": {"level": "DEBUG", "debug_files": MOCK_TOOL_LOGGING_DEBUG_FILES},
    "refiner": {  # For internal Transcriber and Refiner
        "openai_api_url": MOCK_TOOL_REFINER_API_URL,
        "openai_api_key": MOCK_TOOL_REFINER_API_KEY,
        "model_name": MOCK_TOOL_REFINER_MODEL_NAME,
        "whisper_model_name": MOCK_TOOL_WHISPER_MODEL_NAME,
    },
    "costs": {},  # Add empty costs section for completeness
}


class TestChapterRefinementTool:
    """Tests for the ChapterRefinementTool class."""

    @patch("absrefined.refinement_tool.chapter_refinement_tool.AudioTranscriber")
    @patch("absrefined.refinement_tool.chapter_refinement_tool.ChapterRefiner")
    @patch("shutil.which", return_value="ffmpeg_path")  # Mock ffmpeg check
    def test_init(
        self, mock_shutil_which, MockChapterRefiner, MockAudioTranscriber, tmp_path
    ):
        """Test initialization of the ChapterRefinementTool."""
        mock_abs_client = MagicMock(spec=AudiobookshelfClient)

        # Create a specific config for this test to override temp_dir to use tmp_path
        test_specific_config = MOCK_TOOL_CONFIG.copy()
        test_specific_config["processing"] = test_specific_config["processing"].copy()
        test_specific_config["processing"]["download_path"] = str(
            tmp_path / "tool_temp"
        )
        test_specific_config["logging"] = {"level": "DEBUG", "debug_files": True}

        tool = ChapterRefinementTool(
            config=test_specific_config, client=mock_abs_client
        )

        assert tool.config == test_specific_config
        assert tool.abs_client is mock_abs_client
        assert tool.transcriber is MockAudioTranscriber.return_value
        assert tool.refiner is MockChapterRefiner.return_value
        MockAudioTranscriber.assert_called_once_with(config=test_specific_config)
        MockChapterRefiner.assert_called_once_with(config=test_specific_config)

        assert tool.base_temp_dir == str(tmp_path / "tool_temp")
        assert Path(tool.base_temp_dir).exists()  # Ensure tool creates its temp dir
        assert tool.debug_preserve_files is True  # From test_specific_config
        # dry_run is not an init parameter anymore, it's passed to methods like process_item

    @patch("shutil.which", return_value=None)  # Mock ffmpeg check to simulate not found
    @patch(
        "absrefined.refinement_tool.chapter_refinement_tool.AudiobookshelfClient"
    )  # Mock client still needed for init
    def test_init_ffmpeg_not_found(self, MockAbsClient, mock_shutil_which):
        """Test ChapterRefinementTool initialization when ffmpeg is not found."""
        mock_abs_client_instance = MockAbsClient()
        with pytest.raises(OSError) as excinfo:
            ChapterRefinementTool(
                config=MOCK_TOOL_CONFIG, client=mock_abs_client_instance
            )
        assert "ffmpeg not found. Cannot proceed." in str(excinfo.value)

    def test_extract_item_id_from_url(self):
        """Test extracting item ID from URL using the utility function."""
        # mock_abs_client = MagicMock(spec=AudiobookshelfClient) # No longer needed for this test
        # tool = ChapterRefinementTool(config=MOCK_TOOL_CONFIG, client=mock_abs_client) # No longer needed

        # Test with valid URLs
        url1 = "http://abs-server.com/item/lib-item-123"
        assert extract_item_id_from_url(url1) == "lib-item-123"

        url2 = "http://abs-server.com/item/lib-item-123/details"
        assert extract_item_id_from_url(url2) == "lib-item-123"

        url3 = "http://abs-server.com/another/path/item/lib-item-456/bla"
        assert extract_item_id_from_url(url3) == "lib-item-456"

        url4 = "http://localhost/item/some_id_123?query=param"
        assert extract_item_id_from_url(url4) == "some_id_123"

        # Test with invalid URL (no /item/ prefix)
        url_invalid1 = "http://abs-server.com/not-an-item"
        assert extract_item_id_from_url(url_invalid1) == ""

        # Test with URL that has /item/ but no valid ID afterwards according to pattern
        url_invalid2 = "http://abs-server.com/item/"
        assert extract_item_id_from_url(url_invalid2) == ""

        url_invalid3 = "http://abs-server.com/item/!@#$/details"
        assert extract_item_id_from_url(url_invalid3) == ""

        # Test with just an ID (should be caught by the second part of the util function)
        assert extract_item_id_from_url("lib-item-789") == "lib-item-789"
        assert (
            extract_item_id_from_url("cju289sdksj220982nsjjd88l")
            == "cju289sdksj220982nsjjd88l"
        )
        assert extract_item_id_from_url("my-item-id") == "my-item-id"
        assert extract_item_id_from_url("short") == ""  # Assuming min length 7
        assert extract_item_id_from_url("item/with/slash") == ""

    @patch("absrefined.refinement_tool.chapter_refinement_tool.AudiobookshelfClient")
    @patch("absrefined.refinement_tool.chapter_refinement_tool.AudioTranscriber")
    @patch("absrefined.refinement_tool.chapter_refinement_tool.ChapterRefiner")
    @patch("shutil.which", return_value="ffmpeg_path")  # Mock ffmpeg check
    def test_process_item(
        self,
        mock_shutil_which,
        MockChapterRefiner,
        MockAudioTranscriber,
        MockAbsClient,
        tmp_path,
        abs_item_response,
        mock_transcript_data,
    ):
        """Test processing an item to refine chapter markers."""
        # Configure instances that will be created inside ChapterRefinementTool
        mock_abs_client_instance = MockAbsClient.return_value
        mock_transcriber_instance = MockAudioTranscriber.return_value
        mock_refiner_instance = MockChapterRefiner.return_value

        # Configure mock behaviors for client methods
        mock_abs_client_instance.get_item_chapters.return_value = abs_item_response[
            "media"
        ]["chapters"]
        # _ensure_full_audio_downloaded calls abs_client.download_audio_file internally if file not found.
        # For this test, let's mock _ensure_full_audio_downloaded directly to simplify,
        # or ensure download_audio_file is robustly mocked.
        # To keep closer to the original test structure which mocked internal methods of the tool:
        # We will mock the tool's internal methods for now.

        # Create tool - Transcriber and Refiner are created internally using their (mocked) classes
        # The client instance passed to the tool is our top-level mock_abs_client_instance
        tool = ChapterRefinementTool(
            config=MOCK_TOOL_CONFIG, client=mock_abs_client_instance
        )

        # --- Mock internal methods OF THE TOOL to isolate process_item logic --- #
        original_chapters = abs_item_response["media"]["chapters"]

        # Mock return from tool.process_chapters
        # This list should contain dicts with "original_start" and "refined_start" keys.
        processed_chapters_output = []
        original_chapter_data = abs_item_response["media"]["chapters"]
        significant_change_threshold = 0.1  # From process_item

        # Chapter 0: Always start at 0 without actual refinement
        chap0_data = original_chapter_data[0].copy()
        chap0_orig_start_float = float(chap0_data["start"])
        processed_chapters_output.append(
            {
                **chap0_data,
                "original_start": chap0_orig_start_float,
                "refined_start": 0.0,  # Updated to always be 0.0
                "usage_data": None,  # No usage data since we skip processing
                "chunk_duration_seconds": 0,
            }
        )

        # Chapter 1: Refined (difference > threshold)
        chap1_data = original_chapter_data[1].copy()
        chap1_orig_start_float = float(chap1_data["start"])
        chap1_refined_start_float = chap1_orig_start_float + (
            significant_change_threshold + 0.5
        )  # Clearly different and > threshold
        processed_chapters_output.append(
            {
                **chap1_data,
                "original_start": chap1_orig_start_float,
                "refined_start": chap1_refined_start_float,
                "usage_data": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
                "chunk_duration_seconds": 10,
            }
        )

        # Chapter 2: Refined (difference > threshold)
        chap2_data = original_chapter_data[2].copy()
        chap2_orig_start_float = float(chap2_data["start"])
        chap2_refined_start_float = chap2_orig_start_float + (
            significant_change_threshold + 1.0
        )  # Clearly different and > threshold
        processed_chapters_output.append(
            {
                **chap2_data,
                "original_start": chap2_orig_start_float,
                "refined_start": chap2_refined_start_float,
                "usage_data": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
                "chunk_duration_seconds": 10,
            }
        )

        expected_chapters_to_be_updated_count = 2

        mock_full_audio_path = str(tmp_path / "lib-item-123_full.m4a")
        Path(mock_full_audio_path).touch()
        tool._ensure_full_audio_downloaded = MagicMock(
            return_value=mock_full_audio_path
        )
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
            dry_run=dry_run_param,
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
        mock_abs_client_instance.get_item_chapters.assert_called_once_with(
            "lib-item-123"
        )
        tool._ensure_full_audio_downloaded.assert_called_once_with("lib-item-123")
        tool._get_audio_duration.assert_called_once_with(mock_full_audio_path)
        tool.process_chapters.assert_called_once_with(
            item_id="lib-item-123",
            original_chapters=original_chapters,
            full_audio_path=mock_full_audio_path,
            audio_duration=3483.66,
            search_window_seconds=search_window_param,
            model_name_override=model_override_param,
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
    def test_ensure_full_audio_downloaded_exists_valid_m4a(
        self, mock_os_path_exists, mock_shutil_which, tmp_path
    ):
        mock_abs_client = MagicMock(spec=AudiobookshelfClient)
        test_config = MOCK_TOOL_CONFIG.copy()
        mock_download_dir_str = str(tmp_path / "test_downloads")
        test_config["processing"]["download_path"] = mock_download_dir_str
        Path(mock_download_dir_str).mkdir(parents=True, exist_ok=True)
        tool = ChapterRefinementTool(config=test_config, client=mock_abs_client)
        item_id = "test_item_m4a_exists"
        m4a_path_str = os.path.join(mock_download_dir_str, f"{item_id}_full_audio.m4a")
        # Implementation calls os.path.exists 4 times:
        # First check directly for m4a + loop over other extensions + 2 more checks in main path
        mock_os_path_exists.side_effect = [True, True, True]
        result_path = tool._ensure_full_audio_downloaded(item_id)
        assert result_path == m4a_path_str
        expected_calls_in_method = [call(m4a_path_str), call(m4a_path_str)]
        mock_os_path_exists.assert_has_calls(expected_calls_in_method, any_order=False)
        assert mock_os_path_exists.call_count == 4
        mock_abs_client.download_audio_file.assert_not_called()

    @patch("shutil.which", return_value="ffmpeg_path")
    @patch("absrefined.refinement_tool.chapter_refinement_tool.os.path.exists")
    def test_ensure_full_audio_downloaded_exists_valid_other_ext(
        self, mock_os_path_exists, mock_shutil_which, tmp_path
    ):
        """Test when a non-m4a file exists and is used."""
        # Setup logging for debugging
        logging.basicConfig(level=logging.DEBUG)

        mock_abs_client = MagicMock(spec=AudiobookshelfClient)
        test_config = MOCK_TOOL_CONFIG.copy()
        mock_download_dir_str = str(tmp_path / "test_downloads")
        test_config["processing"]["download_path"] = mock_download_dir_str
        Path(mock_download_dir_str).mkdir(parents=True, exist_ok=True)
        tool = ChapterRefinementTool(config=test_config, client=mock_abs_client)

        # Set up a real logger for debugging
        tool.logger = logging.getLogger("TestLogger")

        item_id = "test_item_mp3_exists"
        mp3_path_str = os.path.join(mock_download_dir_str, f"{item_id}_full_audio.mp3")

        # Configure os.path.exists to return True for mp3 and False for all others
        def path_exists_side_effect(path):
            if path == mp3_path_str:
                return True
            return False

        mock_os_path_exists.side_effect = path_exists_side_effect

        # Create the mp3 file to ensure it exists for real
        Path(mp3_path_str).touch()

        result_path = tool._ensure_full_audio_downloaded(item_id)

        assert result_path == mp3_path_str, (
            f"Expected {mp3_path_str}, got {result_path}"
        )

        # Ensure download was never called since file was found
        mock_abs_client.download_audio_file.assert_not_called()

    @patch("shutil.which", return_value="ffmpeg_path")
    @patch("absrefined.refinement_tool.chapter_refinement_tool.os.path.exists")
    def test_ensure_full_audio_downloaded_does_not_exist_success(
        self, mock_os_path_exists, mock_shutil_which, tmp_path
    ):
        """Test downloading a file that doesn't exist."""
        mock_abs_client = MagicMock(spec=AudiobookshelfClient)
        test_config = MOCK_TOOL_CONFIG.copy()
        mock_download_dir_str = str(tmp_path / "test_downloads")
        test_config["processing"]["download_path"] = mock_download_dir_str
        Path(mock_download_dir_str).mkdir(parents=True, exist_ok=True)
        tool = ChapterRefinementTool(config=test_config, client=mock_abs_client)
        tool.logger = logging.getLogger("TestLogger")

        item_id = "test_item_needs_download"
        # The prospective filename in _ensure_full_audio_downloaded uses item_id + _full_audio.m4a
        # The actual path passed to client.download_audio_file is this one.
        expected_download_path = os.path.join(
            mock_download_dir_str, f"{item_id}_full_audio.m4a"
        )

        # First, make sure os.path.exists returns False for all checks to trigger the download
        mock_os_path_exists.return_value = False

        # When download_audio_file is called, create the file and return its path
        # Update side_effect to accept debug_preserve_files
        def download_side_effect(item_id_arg, path_arg, debug_preserve_files=False):
            Path(path_arg).parent.mkdir(parents=True, exist_ok=True)
            Path(path_arg).touch()

            # After creating the file, modify os.path.exists to return True for this specific path
            # and False for others to simulate it now existing.
            def specific_path_exists(p):
                return p == expected_download_path

            mock_os_path_exists.side_effect = specific_path_exists
            return path_arg

        mock_abs_client.download_audio_file.side_effect = download_side_effect

        result_path = tool._ensure_full_audio_downloaded(item_id)

        assert result_path == expected_download_path
        mock_abs_client.download_audio_file.assert_called_once_with(
            item_id, expected_download_path, debug_preserve_files=False
        )
        # os.path.exists should have been called multiple times (initial checks, then after download)
        assert mock_os_path_exists.call_count > 1

    @patch("shutil.which", return_value="ffmpeg_path")
    @patch("absrefined.refinement_tool.chapter_refinement_tool.os.path.exists")
    def test_ensure_full_audio_downloaded_download_client_fails(
        self, mock_os_path_exists, mock_shutil_which, tmp_path
    ):
        """Test when the client fails to download a file."""
        mock_abs_client = MagicMock(spec=AudiobookshelfClient)
        test_config = MOCK_TOOL_CONFIG.copy()
        mock_download_dir_str = str(tmp_path / "test_downloads")
        test_config["processing"]["download_path"] = mock_download_dir_str
        Path(mock_download_dir_str).mkdir(parents=True, exist_ok=True)
        tool = ChapterRefinementTool(config=test_config, client=mock_abs_client)
        tool.logger = logging.getLogger("TestLogger")

        item_id = "test_item_client_fail"
        m4a_path_str = os.path.join(mock_download_dir_str, f"{item_id}_full_audio.m4a")

        # Mock client to simulate download failure (returns None)
        mock_abs_client.download_audio_file.return_value = None

        # Mock os.path.exists to return False for all paths
        def path_exists_side_effect(path):
            return False

        mock_os_path_exists.side_effect = path_exists_side_effect

        result_path = tool._ensure_full_audio_downloaded(item_id)
        assert result_path is None

        # Make sure client's download_audio_file was called
        mock_abs_client.download_audio_file.assert_called_once_with(
            item_id,
            m4a_path_str,
            debug_preserve_files=False,  # Added debug_preserve_files
        )

    @patch("shutil.which", return_value="ffmpeg_path")
    @patch("absrefined.refinement_tool.chapter_refinement_tool.os.path.exists")
    def test_ensure_full_audio_downloaded_file_vanishes_after_download(
        self, mock_os_path_exists, mock_shutil_which, tmp_path
    ):
        """Test when the file vanishes after download."""
        mock_abs_client = MagicMock(spec=AudiobookshelfClient)
        test_config = MOCK_TOOL_CONFIG.copy()
        mock_download_dir_str = str(tmp_path / "test_downloads")
        test_config["processing"]["download_path"] = mock_download_dir_str
        Path(mock_download_dir_str).mkdir(parents=True, exist_ok=True)
        tool = ChapterRefinementTool(config=test_config, client=mock_abs_client)
        tool.logger = logging.getLogger("TestLogger")

        item_id = "test_item_vanishes"
        m4a_path_str = os.path.join(mock_download_dir_str, f"{item_id}_full_audio.m4a")

        # Client download_audio_file returns the path, simulating success
        mock_abs_client.download_audio_file.return_value = m4a_path_str

        # Mock os.path.exists to return False for all paths
        # This simulates the file "disappearing" after the client reports successful download
        # because _ensure_full_audio_downloaded checks os.path.exists again.
        def path_exists_side_effect(path):
            return False

        mock_os_path_exists.side_effect = path_exists_side_effect

        result_path = tool._ensure_full_audio_downloaded(item_id)
        assert result_path is None

        # Make sure client's download_audio_file was called
        mock_abs_client.download_audio_file.assert_called_once_with(
            item_id,
            m4a_path_str,
            debug_preserve_files=False,  # Added debug_preserve_files
        )

    @patch("shutil.which", return_value="ffmpeg_path")
    @patch("subprocess.run")
    def test_get_audio_duration_success(self, mock_run, mock_shutil_which):
        """Test successful retrieval of audio duration."""
        mock_abs_client = MagicMock(spec=AudiobookshelfClient)
        tool = ChapterRefinementTool(config=MOCK_TOOL_CONFIG, client=mock_abs_client)
        tool.logger = logging.getLogger("TestLogger")

        # Mock subprocess.run to return a duration of 300.5 seconds
        mock_process_result = MagicMock()
        mock_process_result.stdout = "300.5\n"
        mock_run.return_value = mock_process_result

        audio_path = "test_audio.m4a"
        duration = tool._get_audio_duration(audio_path)

        assert duration == 300.5
        mock_run.assert_called_once()
        # Verify the ffprobe command was called correctly
        cmd_args = mock_run.call_args[0][0]
        assert "ffprobe" in cmd_args[0]
        assert "-show_entries" in cmd_args
        assert "format=duration" in cmd_args
        assert audio_path in cmd_args

    @patch("shutil.which", return_value="ffmpeg_path")
    @patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "ffprobe", stderr="Error message"),
    )
    def test_get_audio_duration_command_error(self, mock_run, mock_shutil_which):
        """Test audio duration retrieval when ffprobe command fails."""
        mock_abs_client = MagicMock(spec=AudiobookshelfClient)
        tool = ChapterRefinementTool(config=MOCK_TOOL_CONFIG, client=mock_abs_client)
        tool.logger = logging.getLogger("TestLogger")

        audio_path = "test_audio.m4a"
        duration = tool._get_audio_duration(audio_path)

        assert duration is None
        mock_run.assert_called_once()

    @patch("shutil.which", return_value="ffmpeg_path")
    @patch("subprocess.run", side_effect=FileNotFoundError("ffprobe not found"))
    def test_get_audio_duration_file_not_found(self, mock_run, mock_shutil_which):
        """Test audio duration retrieval when ffprobe executable is not found."""
        mock_abs_client = MagicMock(spec=AudiobookshelfClient)
        tool = ChapterRefinementTool(config=MOCK_TOOL_CONFIG, client=mock_abs_client)
        tool.logger = logging.getLogger("TestLogger")

        audio_path = "test_audio.m4a"
        duration = tool._get_audio_duration(audio_path)

        assert duration is None
        mock_run.assert_called_once()

    @patch("shutil.which", return_value="ffmpeg_path")
    @patch("subprocess.run")
    def test_get_audio_duration_value_error(self, mock_run, mock_shutil_which):
        """Test audio duration retrieval when ffprobe output is invalid."""
        mock_abs_client = MagicMock(spec=AudiobookshelfClient)
        tool = ChapterRefinementTool(config=MOCK_TOOL_CONFIG, client=mock_abs_client)
        tool.logger = logging.getLogger("TestLogger")

        # Mock subprocess.run to return invalid non-numeric output
        mock_process_result = MagicMock()
        mock_process_result.stdout = "invalid"
        mock_run.return_value = mock_process_result

        audio_path = "test_audio.m4a"
        duration = tool._get_audio_duration(audio_path)

        assert duration is None
        mock_run.assert_called_once()

    @patch("shutil.which", return_value="ffmpeg_path")
    @patch("subprocess.run")
    def test_extract_audio_segment_success(self, mock_run, mock_shutil_which):
        """Test successful extraction of audio segment."""
        mock_abs_client = MagicMock(spec=AudiobookshelfClient)
        tool = ChapterRefinementTool(config=MOCK_TOOL_CONFIG, client=mock_abs_client)
        tool.logger = logging.getLogger("TestLogger")

        # Mock successful subprocess execution
        mock_process_result = MagicMock()
        mock_process_result.stderr = ""
        mock_run.return_value = mock_process_result

        full_audio_path = "full_audio.m4a"
        chunk_audio_path = "chunk_audio.wav"
        result = tool._extract_audio_segment(
            full_audio_path=full_audio_path,
            window_start=10.0,
            window_end=20.0,
            chunk_audio_path=chunk_audio_path,
        )

        assert result is True
        mock_run.assert_called_once()
        # Verify the ffmpeg command was called correctly
        cmd_args = mock_run.call_args[0][0]
        assert "ffmpeg" in cmd_args[0]
        assert "-i" in cmd_args
        assert full_audio_path in cmd_args
        assert "-ss" in cmd_args
        assert "10.0" in cmd_args
        assert "-to" in cmd_args
        assert "20.0" in cmd_args
        assert chunk_audio_path in cmd_args

    @patch("shutil.which", return_value="ffmpeg_path")
    @patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "ffmpeg", stderr="Error message"),
    )
    def test_extract_audio_segment_command_error(self, mock_run, mock_shutil_which):
        """Test audio segment extraction when ffmpeg command fails."""
        mock_abs_client = MagicMock(spec=AudiobookshelfClient)
        tool = ChapterRefinementTool(config=MOCK_TOOL_CONFIG, client=mock_abs_client)
        tool.logger = logging.getLogger("TestLogger")

        full_audio_path = "full_audio.m4a"
        chunk_audio_path = "chunk_audio.wav"
        result = tool._extract_audio_segment(
            full_audio_path=full_audio_path,
            window_start=10.0,
            window_end=20.0,
            chunk_audio_path=chunk_audio_path,
        )

        assert result is False
        mock_run.assert_called_once()

    @patch("shutil.which", return_value="ffmpeg_path")
    def test_extract_audio_segment_no_chunk_path(self, mock_shutil_which):
        """Test audio segment extraction when no chunk path is provided."""
        mock_abs_client = MagicMock(spec=AudiobookshelfClient)
        tool = ChapterRefinementTool(config=MOCK_TOOL_CONFIG, client=mock_abs_client)
        tool.logger = logging.getLogger("TestLogger")

        full_audio_path = "full_audio.m4a"
        result = tool._extract_audio_segment(
            full_audio_path=full_audio_path,
            window_start=10.0,
            window_end=20.0,
            chunk_audio_path=None,
        )

        assert result is False
        # No subprocess call should happen with a None chunk_audio_path

    @patch("shutil.which", return_value="ffmpeg_path")
    @patch("subprocess.run", side_effect=Exception("Unexpected error"))
    def test_extract_audio_segment_unexpected_error(self, mock_run, mock_shutil_which):
        """Test audio segment extraction when an unexpected error occurs."""
        mock_abs_client = MagicMock(spec=AudiobookshelfClient)
        tool = ChapterRefinementTool(config=MOCK_TOOL_CONFIG, client=mock_abs_client)
        tool.logger = logging.getLogger("TestLogger")

        full_audio_path = "full_audio.m4a"
        chunk_audio_path = "chunk_audio.wav"
        result = tool._extract_audio_segment(
            full_audio_path=full_audio_path,
            window_start=10.0,
            window_end=20.0,
            chunk_audio_path=chunk_audio_path,
        )

        assert result is False
        mock_run.assert_called_once()

    @patch("shutil.which", return_value="ffmpeg_path")
    @patch("tempfile.NamedTemporaryFile")
    @patch("os.path.exists", return_value=True)
    @patch(
        "absrefined.refinement_tool.chapter_refinement_tool.tempfile.NamedTemporaryFile"
    )
    def test_process_chapters_successful_refinement(
        self,
        mock_tempfile,
        mock_path_exists,
        mock_named_temp_file,
        mock_shutil_which,
        tmp_path,
        abs_item_response,
        mock_transcript_data,
    ):
        """Test processing chapters with successful transcription and refinement."""
        # Set up mock temporary file
        mock_temp_file = MagicMock()
        mock_temp_file.name = str(tmp_path / "temp_chunk.wav")
        mock_tempfile.return_value.__enter__.return_value = mock_temp_file
        mock_named_temp_file.return_value.__enter__.return_value = mock_temp_file

        # Create mock client, transcriber, and refiner
        mock_abs_client = MagicMock(spec=AudiobookshelfClient)
        mock_transcriber = MagicMock(spec=AudioTranscriber)
        mock_refiner = MagicMock(spec=ChapterRefiner)

        # Configure test-specific config
        test_config = MOCK_TOOL_CONFIG.copy()
        test_config["processing"] = test_config["processing"].copy()
        test_config["processing"]["download_path"] = str(
            tmp_path / "successful_refinement_test"
        )
        test_config["logging"] = {"level": "DEBUG", "debug_files": True}

        # Create the tool with mocked components
        tool = ChapterRefinementTool(config=test_config, client=mock_abs_client)
        # Replace the tool's transcriber and refiner with our mocks
        tool.transcriber = mock_transcriber
        tool.refiner = mock_refiner
        tool.logger = logging.getLogger("TestLogger")

        # Mock the extract_audio_segment method to avoid actual filesystem operations
        tool._extract_audio_segment = MagicMock(return_value=True)

        # Set up test data
        item_id = "lib-item-123"
        original_chapters = abs_item_response["media"]["chapters"]
        full_audio_path = str(tmp_path / "test_audio.m4a")
        Path(full_audio_path).touch()  # Create empty file
        audio_duration = 3600.0  # 1 hour
        search_window_seconds = 30

        # Configure transcript mock data
        transcript_data = []
        for i in range(5):
            transcript_data.append(
                {
                    "start": 275.0 + i,  # Start at 275 seconds
                    "end": 276.0 + i,
                    "text": f"Segment {i} of transcript",
                    "words": [
                        {
                            "word": f"word{j}",
                            "start": 275.0 + i + (j * 0.2),
                            "end": 275.0 + i + (j * 0.2) + 0.2,
                            "probability": 0.9,
                        }
                        for j in range(5)
                    ],
                }
            )

        mock_transcriber.transcribe_audio.return_value = transcript_data

        # Configure refiner mock behavior - each chapter gets refined to a different timestamp
        refinement_offsets = {}
        usage_data = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

        # Prepare mock return values for each chapter
        for i, chapter in enumerate(original_chapters):
            # Refine each chapter by a different amount (add 1.0s to first, 2.0s to second, etc.)
            chapter_title = chapter.get("title", f"Chapter {i + 1}")
            refinement_offsets[chapter_title] = (
                i + 1.0,
                usage_data,
            )  # offset, usage_data tuple

        # Set up the mock behavior for refine_chapter_start_time
        def mock_refine_side_effect(*args, **kwargs):
            chapter_title = kwargs.get("chapter_title")
            # Return the prepared offset and usage data for this chapter title
            return refinement_offsets.get(chapter_title, (0.0, usage_data))

        mock_refiner.refine_chapter_start_time.side_effect = mock_refine_side_effect

        # Call the method under test
        result = tool.process_chapters(
            item_id=item_id,
            original_chapters=original_chapters,
            full_audio_path=full_audio_path,
            audio_duration=audio_duration,
            search_window_seconds=search_window_seconds,
            model_name_override=None,
        )

        # Verify results
        assert result is not None
        assert len(result) == len(original_chapters)

        # First chapter (index 0) should not have called extract_audio_segment or transcriber
        # Subsequent chapters should still have these methods called
        expected_extraction_calls = len(original_chapters) - 1  # Skip first chapter
        assert tool._extract_audio_segment.call_count == expected_extraction_calls
        assert mock_transcriber.transcribe_audio.call_count == expected_extraction_calls
        assert (
            mock_refiner.refine_chapter_start_time.call_count
            == expected_extraction_calls
        )

        # Check the results for each chapter
        for i, chapter_result in enumerate(result):
            original_chapter = original_chapters[i]
            chapter_title = original_chapter.get("title", f"Chapter {i + 1}")
            original_start = float(original_chapter["start"])

            if i == 0:
                # First chapter should always have refined_start = 0.0
                assert chapter_result["id"] == original_chapter["id"]
                assert chapter_result["title"] == chapter_title
                assert chapter_result["original_start"] == original_start
                assert chapter_result["refined_start"] == 0.0
                assert chapter_result["usage_data"] is None
                assert chapter_result["chunk_path"] is None
            else:
                # Calculate expected window start
                window_start = max(0.0, original_start - search_window_seconds / 2)

                # Calculate expected refined start time
                # window_start + refinement_offset
                expected_offset, _ = refinement_offsets[chapter_title]
                expected_refined_start = window_start + expected_offset

                assert chapter_result["id"] == original_chapter["id"]
                assert chapter_result["title"] == chapter_title
                assert chapter_result["original_start"] == original_start
                assert (
                    abs(chapter_result["refined_start"] - expected_refined_start) < 0.01
                )
                assert chapter_result["usage_data"] == usage_data

    @patch("shutil.which", return_value="ffmpeg_path")
    @patch("tempfile.NamedTemporaryFile")
    @patch("os.path.exists", return_value=True)
    def test_process_chapters_refiner_failure(
        self,
        mock_path_exists,
        mock_named_temp_file,
        mock_shutil_which,
        tmp_path,
        abs_item_response,
    ):
        """Test processing chapters when refinement fails (LLM returns None)."""
        # Set up mock temporary file
        mock_temp_file = MagicMock()
        mock_temp_file.name = str(tmp_path / "temp_chunk.wav")
        mock_named_temp_file.return_value.__enter__.return_value = mock_temp_file

        # Create mock client, transcriber, and refiner
        mock_abs_client = MagicMock(spec=AudiobookshelfClient)
        mock_transcriber = MagicMock(spec=AudioTranscriber)
        mock_refiner = MagicMock(spec=ChapterRefiner)

        # Configure test-specific config
        test_config = MOCK_TOOL_CONFIG.copy()
        test_config["processing"] = test_config["processing"].copy()
        test_config["processing"]["download_path"] = str(
            tmp_path / "refiner_failure_test"
        )

        # Create the tool with mocked components
        tool = ChapterRefinementTool(config=test_config, client=mock_abs_client)
        # Replace the tool's transcriber and refiner with our mocks
        tool.transcriber = mock_transcriber
        tool.refiner = mock_refiner
        tool.logger = logging.getLogger("TestLogger")

        # Mock extraction and transcription success
        tool._extract_audio_segment = MagicMock(return_value=True)

        # Configure transcript mock data - simple transcript
        transcript_data = [
            {
                "start": 290.0,
                "end": 300.0,
                "text": "Sample transcript text",
                "words": [
                    {
                        "word": "Sample",
                        "start": 290.0,
                        "end": 292.0,
                        "probability": 0.9,
                    },
                    {
                        "word": "transcript",
                        "start": 292.5,
                        "end": 295.0,
                        "probability": 0.9,
                    },
                    {"word": "text", "start": 295.5, "end": 300.0, "probability": 0.9},
                ],
            }
        ]
        mock_transcriber.transcribe_audio.return_value = transcript_data

        # Mock refiner to return None for offset (simulating LLM failure)
        # but return usage data (since the LLM was still called)
        usage_data = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        mock_refiner.refine_chapter_start_time.return_value = (None, usage_data)

        # Set up test data
        item_id = "lib-item-123"
        original_chapters = abs_item_response["media"]["chapters"]
        full_audio_path = str(tmp_path / "test_audio.m4a")
        Path(full_audio_path).touch()
        audio_duration = 3600.0
        search_window_seconds = 30

        # Call process_chapters
        result = tool.process_chapters(
            item_id=item_id,
            original_chapters=original_chapters,
            full_audio_path=full_audio_path,
            audio_duration=audio_duration,
            search_window_seconds=search_window_seconds,
            model_name_override=None,
        )

        # Verify results
        assert result is not None
        assert len(result) == len(original_chapters)

        # Verify functions were called the expected number of times (skip first chapter)
        expected_calls = len(original_chapters) - 1  # Skip first chapter
        assert tool._extract_audio_segment.call_count == expected_calls
        assert mock_transcriber.transcribe_audio.call_count == expected_calls
        assert mock_refiner.refine_chapter_start_time.call_count == expected_calls

        # Check the results for specific chapters
        # First chapter should have refined_start = 0.0 without processing
        first_chapter = result[0]
        assert first_chapter["id"] == original_chapters[0]["id"]
        assert first_chapter["refined_start"] == 0.0
        assert first_chapter["usage_data"] is None  # No usage for first chapter

        # Other chapters should have tried refinement but failed (refined_start is None)
        for i in range(1, len(result)):
            chapter_result = result[i]
            original_chapter = original_chapters[i]
            assert chapter_result["id"] == original_chapter["id"]
            assert (
                chapter_result["refined_start"] is None
            )  # No refinement because LLM returned None
            assert (
                chapter_result["usage_data"] == usage_data
            )  # Usage data should be present

    @patch("shutil.which", return_value="ffmpeg_path")
    @patch("tempfile.NamedTemporaryFile")
    @patch("os.path.exists", return_value=True)
    def test_process_chapters_missing_start_time(
        self, mock_path_exists, mock_named_temp_file, mock_shutil_which, tmp_path
    ):
        """Test processing chapters when a chapter is missing a start time."""
        # Set up mock temporary file
        mock_temp_file = MagicMock()
        mock_temp_file.name = str(tmp_path / "temp_chunk.wav")
        mock_named_temp_file.return_value.__enter__.return_value = mock_temp_file

        # Create mock client, transcriber, and refiner
        mock_abs_client = MagicMock(spec=AudiobookshelfClient)
        mock_transcriber = MagicMock(spec=AudioTranscriber)
        mock_refiner = MagicMock(spec=ChapterRefiner)

        # Configure test-specific config
        test_config = MOCK_TOOL_CONFIG.copy()
        test_config["processing"] = test_config["processing"].copy()
        test_config["processing"]["download_path"] = str(
            tmp_path / "missing_start_test"
        )

        # Create the tool with mocked components
        tool = ChapterRefinementTool(config=test_config, client=mock_abs_client)
        # Replace the tool's transcriber and refiner with our mocks
        tool.transcriber = mock_transcriber
        tool.refiner = mock_refiner
        tool.logger = logging.getLogger("TestLogger")

        # Mock successful audio extraction and transcription
        tool._extract_audio_segment = MagicMock(return_value=True)

        # Configure transcript mock data
        transcript_data = [
            {
                "start": 10.0,
                "end": 15.0,
                "text": "Sample text for testing.",
                "words": [
                    {"word": "Sample", "start": 10.0, "end": 11.0, "probability": 0.9},
                    {"word": "text", "start": 11.5, "end": 12.0, "probability": 0.9},
                    {"word": "for", "start": 12.5, "end": 13.0, "probability": 0.9},
                    {
                        "word": "testing.",
                        "start": 13.5,
                        "end": 15.0,
                        "probability": 0.9,
                    },
                ],
            }
        ]
        mock_transcriber.transcribe_audio.return_value = transcript_data

        # Mock refiner to return a refined timestamp
        usage_data = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        mock_refiner.refine_chapter_start_time.return_value = (5.0, usage_data)

        # Create test data with missing start times
        item_id = "missing-start-test"
        original_chapters = [
            {
                "id": "chapter-1",
                "title": "Chapter 1",
                "start": 0.0,  # First chapter typically starts at 0
                "end": 60.0,
            },
            {
                "id": "chapter-2",
                "title": "Chapter 2",
                # No start time for this chapter
                "end": 120.0,
            },
            {
                "id": "chapter-3",
                "title": "Chapter 3",
                "start": None,  # Explicit None
                "end": 180.0,
            },
        ]

        full_audio_path = str(tmp_path / "test_audio.m4a")
        Path(full_audio_path).touch()
        audio_duration = 180.0
        search_window_seconds = 30

        # Call the method under test
        result = tool.process_chapters(
            item_id=item_id,
            original_chapters=original_chapters,
            full_audio_path=full_audio_path,
            audio_duration=audio_duration,
            search_window_seconds=search_window_seconds,
            model_name_override=None,
        )

        # Verify results
        assert result is not None
        assert len(result) == len(original_chapters)

        # Check that we only processed chapters after the first one
        assert tool._extract_audio_segment.call_count == len(original_chapters) - 1
        assert (
            mock_transcriber.transcribe_audio.call_count == len(original_chapters) - 1
        )
        assert (
            mock_refiner.refine_chapter_start_time.call_count
            == len(original_chapters) - 1
        )

        # First chapter should have refined_start = 0.0 without processing
        assert result[0]["id"] == original_chapters[0]["id"]
        assert result[0]["original_start"] == 0.0
        assert result[0]["refined_start"] == 0.0
        assert result[0]["usage_data"] is None
        assert result[0]["chunk_path"] is None

        # Second chapter should have original_start defaulted to 0.0 (due to missing) but with refinement
        assert result[1]["id"] == original_chapters[1]["id"]
        assert result[1]["original_start"] == 0.0  # Default when missing
        assert result[1]["refined_start"] is not None  # Should have been refined

        # Third chapter should also have a default of 0.0 for explicit None start time
        assert result[2]["id"] == original_chapters[2]["id"]
        assert result[2]["original_start"] == 0.0  # Default for explicit None
        assert result[2]["refined_start"] is not None  # Should have been refined
