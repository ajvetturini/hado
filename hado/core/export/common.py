from pathlib import Path


def _resolve_output_path(filepath: str | Path, filename: str) -> Path:
    """Create the output directory and return the full output file path."""
    if isinstance(filepath, str):
        output_dir = Path(filepath)
    elif isinstance(filepath, Path):
        output_dir = filepath
    else:
        raise TypeError("filepath must be a str or Path")

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename
