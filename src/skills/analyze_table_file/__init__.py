import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

_DOWNLOADS_DIR = os.path.abspath(
    os.getenv(
        "AIDEN_DOWNLOADS_DIR",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "..",
            "downloads",
        ),
    )
)


def _resolve_data_path(file_path: str) -> str:
    normalized = file_path.strip()
    if not normalized:
        return normalized
    normalized = normalized.replace("/", os.sep)
    if os.path.isabs(normalized):
        if os.path.exists(normalized):
            return normalized
        return os.path.join(_DOWNLOADS_DIR, os.path.basename(normalized))
    return os.path.join(_DOWNLOADS_DIR, normalized)


import json

async def analyze_table_file(
    file_path: str,
    sheet_name: str = None,
    max_rows: int = 20,
    max_chars: int = 12000
) -> str:
    """
    Load a CSV/TSV/Excel file and return a concise summary using pandas.
    """
    try:
        resolved = _resolve_data_path(file_path)
        if not resolved:
            return json.dumps({
                "status": "error",
                "message": "Invalid file_path",
                "details": "The file_path must be a non-empty string."
            })
        if not os.path.exists(resolved):
            return json.dumps({
                "status": "error",
                "message": f"File not found at '{resolved}'",
                "suggestion": "Use manage_file_system 'list' on the downloads folder or parent directory to confirm the correct filename."
            })

        try:
            max_rows = int(max_rows)
            max_chars = int(max_chars)
        except ValueError:
            max_rows = 20
            max_chars = 12000

        if max_rows < 1:
            max_rows = 20
        if max_chars < 500:
            max_chars = 12000

        ext = os.path.splitext(resolved)[1].lower()
        if ext in {".csv", ".tsv"}:
            sep = "\t" if ext == ".tsv" else ","
            df = pd.read_csv(resolved, sep=sep, nrows=10000, low_memory=False)
        elif ext in {".xlsx", ".xls"}:
            df = pd.read_excel(resolved, sheet_name=sheet_name or 0, nrows=10000)
        else:
            return "Error: Unsupported file type. Use CSV, TSV, XLSX, or XLS."

        shape = df.shape
        head = df.head(max_rows)
        missing = df.isna().sum()
        missing_top = missing[missing > 0].sort_values(ascending=False).head(10)

        lines = [
            f"File: {os.path.basename(resolved)}",
            f"Shape: {shape[0]} rows x {shape[1]} columns",
            "Columns:",
            ", ".join([f"{c} ({df[c].dtype})" for c in df.columns]),
        ]
        if not missing_top.empty:
            lines.append("Missing values (top 10):")
            for col, count in missing_top.items():
                lines.append(f"  {col}: {int(count)}")

        lines.append("Preview:")
        lines.append(head.to_string(index=False))

        output = "\n".join(lines)
        is_truncated = False
        if len(output) > max_chars:
            output = output[:max_chars].rstrip() + "\n\n[TRUNCATED]"
            is_truncated = True

        logger.info(f"analyze_table_file: analyzed '{resolved}' ({shape[0]}x{shape[1]})")
        import json
        return json.dumps({
            "status": "success",
            "file": os.path.basename(resolved),
            "analysis": output,
            "truncated": is_truncated
        }, indent=2)
    except Exception as exc:
        logger.error(f"analyze_table_file failed: {exc}", exc_info=True)
        import json
        return json.dumps({
            "status": "error",
            "message": "Data analysis failed",
            "details": str(exc),
            "suggestion": "Check if the file is corrupted, or if the sheet_name is correct for Excel files."
        })
