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
            return "Error: file_path is empty."
        if not os.path.exists(resolved):
            return (
                f"Error: File not found at '{resolved}'. "
                "Use manage_file_system 'list' on the downloads folder to confirm the name."
            )
        if max_rows < 1:
            return "Error: max_rows must be at least 1."
        if max_chars < 500:
            return "Error: max_chars must be at least 500."

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
        if len(output) > max_chars:
            output = output[:max_chars].rstrip() + "\n\n[TRUNCATED]"

        logger.info(f"analyze_table_file: analyzed '{resolved}' ({shape[0]}x{shape[1]})")
        return output
    except Exception as exc:
        logger.error(f"analyze_table_file failed: {exc}", exc_info=True)
        return f"Error: Data analysis failed due to [{exc}]."
