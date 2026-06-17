#!/usr/bin/env python3
"""
Extract Python code from Quarto (.qmd) and Jupyter Notebook (.ipynb) files.
Usage:
    python extract_python.py <input_file> [output_file]

If output_file is omitted, the script saves a .py file with the same base name.
"""

import json
import re
import sys
from pathlib import Path


def extract_from_qmd(path: Path) -> str:
    """Extract Python code blocks from a Quarto (.qmd) file."""
    text = path.read_text(encoding="utf-8")
    # Match ```{python} ... ``` blocks (with optional trailing options)
    pattern = re.compile(r"```\{python[^}]*\}\n(.*?)```", re.DOTALL)
    blocks = pattern.findall(text)
    if not blocks:
        print(f"  Warning: no Python code blocks found in {path.name}")
    return "\n\n".join(blocks)


def extract_from_ipynb(path: Path) -> str:
    """Extract Python source cells from a Jupyter Notebook (.ipynb) file."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    blocks = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            # source can be a list of lines or a single string
            code = "".join(source) if isinstance(source, list) else source
            # Skip cells that are clearly non-Python (e.g. %%bash magic)
            if code.strip():
                blocks.append(code)
    if not blocks:
        print(f"  Warning: no code cells found in {path.name}")
    return "\n\n".join(blocks)


def extract(input_path: str, output_path: str | None = None) -> Path:
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")

    ext = src.suffix.lower()
    if ext == ".qmd":
        code = extract_from_qmd(src)
    elif ext == ".ipynb":
        code = extract_from_ipynb(src)
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Expected .qmd or .ipynb.")

    dest = Path(output_path) if output_path else src.with_suffix(".py")
    dest.write_text(code, encoding="utf-8")
    print(f"  Saved: {dest}")
    return dest


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    input_file = args[0]
    output_file = args[1] if len(args) > 1 else None
    try:
        extract(input_file, output_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
