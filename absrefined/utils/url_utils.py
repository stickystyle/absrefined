import re


def extract_item_id_from_url(url: str) -> str:
    """
    Extracts the item ID from an Audiobookshelf item URL.
    Expected format: .../item/<item_id>/... or .../item/<item_id>
    Also handles URLs ending with /details or similar, and just the item ID itself if no slashes.

    Args:
        url (str): The URL string to parse.

    Returns:
        str: The extracted item ID, or an empty string if not found.
    """
    # Regex to find item ID like lib-item-xxxx, or generic IDs (alphanumeric, underscore, hyphen)
    # It looks for /item/ followed by the ID, optionally followed by / or end of string.
    # Handles common ID patterns like 'lib-item-...', CUIDs, or generic alphanumeric IDs.
    # Example patterns for IDs (more specific ones first):
    # lib_pattern = r"lib-[0-9a-f]{32}" # Example: lib-item-123 (using a more general form below)
    # cuid_pattern = r"c[a-z0-9]{24}"
    # generic_id_pattern = r"[a-zA-Z0-9_-]{7,}" # Reasonably long alphanumeric ID

    # Combined pattern for typical /item/ID scenarios
    # Allows for trailing slashes or segments like /details or query parameters
    path_match = re.search(
        r"/item/((?:lib-[a-zA-Z0-9_-]+)|(?:c[a-z0-9]{24})|(?:[a-zA-Z0-9_-]{7,}))(?:[/\\?].*|$)",
        url,
    )
    if path_match:
        return path_match.group(1)

    # If no /item/ structure, check if the URL itself is just a valid ID pattern and contains no slashes
    # This is to catch cases where just 'lib-item-123' is passed
    if "/" not in url:
        if re.fullmatch(
            r"(?:lib-[a-zA-Z0-9_-]+)|(?:c[a-z0-9]{24})|(?:[a-zA-Z0-9_-]{7,})", url
        ):
            return url

    return ""
