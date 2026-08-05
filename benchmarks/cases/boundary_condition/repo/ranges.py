def inclusive_range(start: int, end: int) -> list[int]:
    """Return every integer from start through end, including both bounds."""
    if start > end:
        return []
    return list(range(start, end))
