def parse_port(value: object) -> int:
    """Convert a value to a valid TCP port number."""
    try:
        port = int(value)
    except TypeError as exc:
        raise ValueError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    return port
