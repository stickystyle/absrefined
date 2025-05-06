import pytest
from unittest.mock import patch, MagicMock
import argparse
import sys
from pathlib import Path

from absrefined.main import main


class TestMain:
    """Tests for the main module."""

    @patch("sys.argv")
    @patch("absrefined.main.get_config")
    @patch("absrefined.main.extract_item_id_from_url")
    @patch("absrefined.main.AudiobookshelfClient")
    @patch("absrefined.main.ChapterRefinementTool")
    def test_main_dry_run(
        self, mock_tool, mock_client, mock_extract_id, mock_get_config, mock_argv
    ):
        """Test main function with dry run option."""
        # Setup argv mock
        mock_argv.__getitem__.side_effect = lambda i: [
            "absrefined",
            "lib-item-123",
            "--dry-run",
        ][i]
        mock_argv.__len__.return_value = 3

        # Setup ID extraction
        mock_extract_id.return_value = "lib-item-123"

        # Setup config
        mock_config = {
            "audiobookshelf": {"host": "http://abs.example.com", "api_key": "test_key"},
            "processing": {
                "download_path": "/tmp/downloads",
                "search_window_seconds": 20,
            },
            "refiner": {
                "model_name": "gpt-4-mini",
                "openai_api_key": "test_key",
                "openai_api_url": "http://api.example.com",
            },
            "logging": {"level": "INFO"},
        }
        mock_get_config.return_value = mock_config

        # Setup tool mock with processed item data
        mock_tool_instance = MagicMock()
        # Mock successful processing result - return a dict with success=True
        mock_tool_instance.process_item.return_value = {"success": True, "changes": 5}
        mock_tool.return_value = mock_tool_instance

        # Mock client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # Call the function
        with patch(
            "absrefined.main.argparse.ArgumentParser.parse_args"
        ) as mock_parse_args:
            # Create a mock return for the parsed arguments
            mock_args = MagicMock()
            mock_args.item_specifier = "lib-item-123"
            mock_args.config = Path("config.toml")
            mock_args.dry_run = True
            mock_args.debug = False
            mock_args.verbose = False
            mock_args.just_download = False
            mock_args.model = None
            mock_args.window = None
            mock_args.download_path = None
            mock_args.yes = False
            mock_parse_args.return_value = mock_args

            # The config path check
            with patch("pathlib.Path.is_file", return_value=True):
                # Run main
                result = main()

        # Verify that the key functions were called
        mock_get_config.assert_called_once()
        mock_client.assert_called_once()
        mock_tool.assert_called_once()

        # Check that process_item was called with the correct parameters
        mock_tool_instance.process_item.assert_called_once()
        call_args = mock_tool_instance.process_item.call_args
        assert call_args.kwargs["item_id"] == "lib-item-123"
        assert call_args.kwargs["search_window_seconds"] == 20
        assert call_args.kwargs["model_name_override"] is None
        assert call_args.kwargs["dry_run"] is True

        # Verify that the function ran successfully
        assert result == 0

    @patch("sys.argv")
    def test_main_config_error(self, mock_argv):
        """Test main function with config error."""
        # Setup argv mock
        mock_argv.__getitem__.side_effect = lambda i: ["absrefined", "lib-item-123"][i]
        mock_argv.__len__.return_value = 2

        # Mock config error
        with patch("absrefined.main.get_config") as mock_get_config:
            from absrefined.config import ConfigError

            mock_get_config.side_effect = ConfigError("Config file not found")

            # Mock config path checks
            with patch("pathlib.Path.is_file", return_value=True):
                with patch(
                    "pathlib.Path.exists", return_value=True
                ):  # For example config
                    # Call main function
                    result = main()

        # Verify error exit code
        assert result == 1
