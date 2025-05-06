import pytest
from absrefined.utils.timestamp import format_timestamp, parse_timestamp
from absrefined.utils.url_utils import extract_item_id_from_url


class TestUtils:
    """Tests for utility functions."""

    def test_format_timestamp(self):
        """Test formatting seconds as HH:MM:SS.mmm."""
        # Test zero seconds
        assert format_timestamp(0) == "00:00:0.000"

        # Test simple cases
        assert format_timestamp(61.5) == "00:01:1.500"
        assert format_timestamp(3661.123) == "01:01:1.123"

        # Test rounding
        assert format_timestamp(3661.1234) == "01:01:1.123"

        # Test large values
        assert format_timestamp(36000) == "10:00:0.000"

    def test_parse_timestamp(self):
        """Test parsing a timestamp in the format HH:MM:SS.mmm to seconds."""
        # Test simple cases
        assert parse_timestamp("5") == 5.0
        assert parse_timestamp("01:30") == 90.0
        assert parse_timestamp("01:30.5") == 90.5
        assert parse_timestamp("00:01:30") == 90.0
        assert parse_timestamp("01:01:01") == 3661.0
        assert parse_timestamp("01:01:01.5") == 3661.5

        # Test edge cases
        assert parse_timestamp("0") == 0.0
        assert parse_timestamp("00:00:00") == 0.0

    def test_extract_item_id_from_url(self):
        """Test extracting item ID from various URL formats."""
        # Standard URL format
        assert (
            extract_item_id_from_url("http://abs.example.com/item/lib-item-123")
            == "lib-item-123"
        )
        assert (
            extract_item_id_from_url(
                "https://abs.example.com/item/lib-item-123/details"
            )
            == "lib-item-123"
        )

        # With trailing slashes
        assert (
            extract_item_id_from_url("http://abs.example.com/item/lib-item-123/")
            == "lib-item-123"
        )

        # With query parameters
        assert (
            extract_item_id_from_url(
                "http://abs.example.com/item/lib-item-123?param=value"
            )
            == "lib-item-123"
        )

        # Different ID formats
        assert (
            extract_item_id_from_url(
                "http://abs.example.com/item/c1a2b3c4d5e6f7g8h9i0j1k2l3"
            )
            == "c1a2b3c4d5e6f7g8h9i0j1k2l3"
        )
        assert (
            extract_item_id_from_url("http://abs.example.com/item/custom_id_1234")
            == "custom_id_1234"
        )

        # Just the ID itself - if it matches expected patterns it will be returned
        assert extract_item_id_from_url("lib-item-123") == "lib-item-123"
        assert (
            extract_item_id_from_url("c1a2b3c4d5e6f7g8h9i0j1k2l3")
            == "c1a2b3c4d5e6f7g8h9i0j1k2l3"
        )
        assert (
            extract_item_id_from_url("invalid_id_1234") == "invalid_id_1234"
        )  # Matches the pattern

        # Invalid formats
        assert extract_item_id_from_url("http://abs.example.com/library/") == ""
        assert (
            extract_item_id_from_url("inv") == ""
        )  # Too short to be considered a valid ID
        assert extract_item_id_from_url("") == ""
