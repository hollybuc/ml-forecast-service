"""File upload and processing utilities."""

import chardet
import pandas as pd
from pathlib import Path
from typing import Optional
from fastapi import UploadFile, HTTPException

from backend.core.config import settings


class FileHandler:
    """Handler for file uploads and data parsing."""

    SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet"}

    def __init__(self):
        """Initialize file handler."""
        self.upload_dir = Path(settings.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    async def save_upload_file(self, upload_file: UploadFile) -> Path:
        """Save uploaded file to disk.

        Args:
            upload_file: FastAPI UploadFile instance.

        Returns:
            Path to saved file.

        Raises:
            HTTPException: If file extension is not supported or file is too large.
        """
        # Validate file extension
        file_ext = Path(upload_file.filename).suffix.lower()
        if file_ext not in self.SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Supported: {', '.join(self.SUPPORTED_EXTENSIONS)}",
            )

        # Read file content
        content = await upload_file.read()

        # Check file size
        if len(content) > settings.max_upload_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_upload_size / (1024 * 1024):.0f}MB",
            )

        # Save file
        file_path = self.upload_dir / upload_file.filename
        with open(file_path, "wb") as f:
            f.write(content)

        return file_path

    def detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding.

        Args:
            file_path: Path to file.

        Returns:
            Detected encoding (e.g., 'utf-8', 'iso-8859-1').
        """
        with open(file_path, "rb") as f:
            raw_data = f.read(10000)  # Read first 10KB for detection
            result = chardet.detect(raw_data)
            return result["encoding"] or "utf-8"

    def detect_csv_delimiter(self, file_path: Path, encoding: str = "utf-8") -> str:
        """Detect CSV delimiter.

        Args:
            file_path: Path to CSV file.
            encoding: File encoding.

        Returns:
            Detected delimiter (e.g., ',', ';', '\t').
        """
        with open(file_path, "r", encoding=encoding) as f:
            first_line = f.readline()

        # Count occurrences of common delimiters
        delimiters = {",": 0, ";": 0, "\t": 0, "|": 0}
        for delimiter in delimiters:
            delimiters[delimiter] = first_line.count(delimiter)

        # Return delimiter with highest count
        return max(delimiters, key=delimiters.get)

    def parse_file(
        self,
        file_path: Path,
        encoding: Optional[str] = None,
        delimiter: Optional[str] = None,
    ) -> pd.DataFrame:
        """Parse file and return DataFrame.

        Args:
            file_path: Path to file.
            encoding: File encoding (auto-detected if None).
            delimiter: CSV delimiter (auto-detected if None).

        Returns:
            Pandas DataFrame.

        Raises:
            HTTPException: If file cannot be parsed.
        """
        file_ext = file_path.suffix.lower()

        try:
            if file_ext == ".csv":
                # Auto-detect encoding if not provided
                if encoding is None:
                    encoding = self.detect_encoding(file_path)

                # Auto-detect delimiter if not provided
                if delimiter is None:
                    delimiter = self.detect_csv_delimiter(file_path, encoding)

                df = pd.read_csv(file_path, encoding=encoding, sep=delimiter)

            elif file_ext in {".xlsx", ".xls"}:
                df = pd.read_excel(file_path)

            elif file_ext == ".parquet":
                df = pd.read_parquet(file_path)

            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file extension: {file_ext}",
                )

            # Basic validation
            if df.empty:
                raise HTTPException(status_code=400, detail="File is empty")

            if len(df.columns) < 2:
                raise HTTPException(
                    status_code=400,
                    detail="File must have at least 2 columns (date and target)",
                )

            if len(df) < 10:
                raise HTTPException(
                    status_code=400,
                    detail="File must have at least 10 rows",
                )

            return df

        except pd.errors.ParserError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to parse file: {str(e)}",
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing file: {str(e)}",
            )

    def get_file_info(self, file_path: Path) -> dict:
        """Get basic file information.

        Args:
            file_path: Path to file.

        Returns:
            Dictionary with file metadata.
        """
        stat = file_path.stat()
        return {
            "filename": file_path.name,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "extension": file_path.suffix.lower(),
        }

    def get_dataframe_preview(self, df: pd.DataFrame, n_rows: int = 10) -> dict:
        """Get DataFrame preview for frontend.

        Args:
            df: Pandas DataFrame.
            n_rows: Number of rows to include in preview.

        Returns:
            Dictionary with preview data.
        """
        return {
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "preview_data": df.head(n_rows)
            .replace([float("inf"), float("-inf")], None)
            .where(pd.notnull(df.head(n_rows)), None)
            .to_dict(orient="records"),
        }


# Global file handler instance
file_handler = FileHandler()


def get_file_handler() -> FileHandler:
    """Get file handler instance (for dependency injection).

    Returns:
        FileHandler instance.
    """
    return file_handler
