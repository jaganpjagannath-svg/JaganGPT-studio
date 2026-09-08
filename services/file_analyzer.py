import os
import csv
import json
import base64
from typing import Dict, Any

class FileAnalyzer:
    """Extracts text and metadata from documents, spreadsheets, code, and images."""

    @staticmethod
    def analyze_file(file_path: str, original_filename: str) -> Dict[str, Any]:
        ext = original_filename.rsplit('.', 1)[-1].lower() if '.' in original_filename else ''
        file_size = os.path.getsize(file_path)

        result = {
            'filename': original_filename,
            'extension': ext,
            'file_size': file_size,
            'extracted_text': '',
            'summary': '',
            'type_category': 'unknown',
            'is_image': False,
            'base64_data': None
        }

        try:
            # 1. PDF
            if ext == 'pdf':
                result['type_category'] = 'document'
                result['extracted_text'] = FileAnalyzer._extract_pdf(file_path)
                result['summary'] = f"PDF document ({len(result['extracted_text'].splitlines())} lines extracted)"

            # 2. Word (DOCX)
            elif ext in ['docx', 'doc']:
                result['type_category'] = 'document'
                result['extracted_text'] = FileAnalyzer._extract_docx(file_path)
                result['summary'] = f"Word document ({len(result['extracted_text'].splitlines())} paragraphs extracted)"

            # 3. Excel & CSV
            elif ext in ['xlsx', 'xls', 'csv']:
                result['type_category'] = 'spreadsheet'
                if ext == 'csv':
                    result['extracted_text'] = FileAnalyzer._extract_csv(file_path)
                else:
                    result['extracted_text'] = FileAnalyzer._extract_excel(file_path)
                result['summary'] = f"Spreadsheet data ({len(result['extracted_text'].splitlines())} rows extracted)"

            # 4. Text & Code
            elif ext in ['txt', 'py', 'js', 'html', 'css', 'json', 'sql', 'java', 'cpp', 'c', 'cs', 'ts', 'jsx', 'tsx', 'md', 'sh', 'bat']:
                result['type_category'] = 'code' if ext != 'txt' else 'text'
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    result['extracted_text'] = f.read(50000) # limit to 50KB for fast analysis
                result['summary'] = f"{ext.upper()} file ({len(result['extracted_text'].splitlines())} lines)"

            # 5. Images
            elif ext in ['jpg', 'jpeg', 'png', 'webp', 'gif']:
                result['type_category'] = 'image'
                result['is_image'] = True
                with open(file_path, 'rb') as img_file:
                    result['base64_data'] = base64.b64encode(img_file.read()).decode('utf-8')
                result['summary'] = f"Image file ({round(file_size / 1024, 1)} KB)"

            # 6. Audio / Video
            elif ext in ['mp3', 'wav', 'ogg', 'm4a', 'webm', 'mp4']:
                result['type_category'] = 'media'
                result['summary'] = f"{ext.upper()} media recording ({round(file_size / (1024*1024), 2)} MB)"

            else:
                result['summary'] = f"{ext.upper()} file uploaded"

        except Exception as e:
            result['summary'] = f"Uploaded file ({ext.upper()}) - note: {str(e)[:100]}"

        return result

    @staticmethod
    def _extract_pdf(file_path: str) -> str:
        import pypdf
        text_parts = []
        reader = pypdf.PdfReader(file_path)
        for i, page in enumerate(reader.pages[:25]): # Extract up to 25 pages
            page_text = page.extract_text()
            if page_text:
                text_parts.append(f"--- [Page {i+1}] ---\n{page_text}")
        return "\n\n".join(text_parts)

    @staticmethod
    def _extract_docx(file_path: str) -> str:
        import docx
        doc = docx.Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)

    @staticmethod
    def _extract_csv(file_path: str) -> str:
        lines = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i > 100: # First 100 rows preview
                    lines.append("... [additional rows truncated] ...")
                    break
                lines.append(", ".join(row))
        return "\n".join(lines)

    @staticmethod
    def _extract_excel(file_path: str) -> str:
        import openpyxl
        wb = openpyxl.load_workbook(file_path, data_only=True)
        lines = []
        for sheet_name in wb.sheetnames[:3]: # first 3 sheets
            sheet = wb[sheet_name]
            lines.append(f"=== Sheet: {sheet_name} ===")
            for r_idx, row in enumerate(sheet.iter_rows(values_only=True)):
                if r_idx > 50:
                    lines.append("... [rows truncated] ...")
                    break
                row_vals = [str(c) if c is not None else "" for c in row]
                if any(row_vals):
                    lines.append(" | ".join(row_vals))
        return "\n".join(lines)
