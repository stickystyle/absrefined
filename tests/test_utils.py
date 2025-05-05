import pytest
from absrefined.utils.timestamp import format_timestamp, parse_timestamp


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