from typing import Union


def format_timestamp(seconds: float) -> str:
    """
    Format seconds as HH:MM:SS.mmm.

    Args:
        seconds (float): Time in seconds

    Returns:
        str: Formatted timestamp
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:.3f}"


def parse_timestamp(timestamp: Union[str, int, float]) -> float:
    """
    Parse a timestamp in the format HH:MM:SS.mmm to seconds.

    Args:
        timestamp (Union[str, int, float]): Timestamp to parse

    Returns:
        float: Time in seconds
    """
    # Handle numeric values (int or float)
    if isinstance(timestamp, (int, float)):
        return float(timestamp)

    # Handle string timestamps
    if ":" not in timestamp:
        return float(timestamp)

    parts = timestamp.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        hours = 0
    else:
        hours, minutes, seconds = parts

    return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
