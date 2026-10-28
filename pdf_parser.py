import os
from typing import List
from pathlib import Path

import fitz  # PyMuPDF


class PDFParser:
    def __init__(self, upload_dir: str = None):
        self.upload_dir = Path(upload_dir or os.getenv("UPLOAD_DIR", "./data/uploads"))
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def parse_pdf(self, file_path: str) -> str:
        """Extract all text from a PDF file."""
        text = []
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text.append(page.get_text())
            doc.close()
            full_text = "\n\n".join(text)
            if len(full_text.strip()) < 200:
                raise Exception("PDF 解析结果过少，可能是扫描版 PDF（纯图片）。建议先使用 OCR 工具处理后再上传。")
            return full_text
        except Exception as e:
            if "扫描版" in str(e):
                raise e
            raise Exception(f"Failed to parse PDF: {str(e)}")

    def save_uploaded_file(self, uploaded_file) -> str:
        """Save uploaded file to disk and return the path."""
        file_path = self.upload_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return str(file_path)
