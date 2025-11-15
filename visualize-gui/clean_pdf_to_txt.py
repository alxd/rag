"""
Standalone script to clean PDF and convert to TXT with aggressive artifact removal.
Specifically designed to clean Irwin.pdf and similar PDFs with heavy formatting artifacts.
"""

import re
import sys
from pathlib import Path

try:
    import pypdf
except ImportError:
    print("Error: pypdf not installed. Install with: pip install pypdf")
    sys.exit(1)


def clean_text_aggressive(text: str) -> str:
    """
    Aggressively clean text to remove all formatting artifacts, page numbers, dates, etc.
    This is the same normalization used in the main code.
    """
    if not text:
        return ""
    
    # Remove XML-like tags (e.g., "<the companions>" -> "the companions")
    text = re.sub(r'<([^>]+)>', r'\1', text)
    
    # Remove section markers with brackets (e.g., "[c6]", "[c1]")
    text = re.sub(r'\[c\d+\]', '', text, flags=re.IGNORECASE)
    
    # Remove stray numbers in brackets (e.g., "[13]", "[5]")
    text = re.sub(r'\[\d+\]', '', text)
    
    # Remove section markers and special characters (e.g., "§4", "§6", "§3")
    text = re.sub(r'§\d+[a-z]?', '', text)
    
    # Remove standalone forward slashes with spaces (e.g., " / " -> " ")
    text = re.sub(r'\s*/\s*', ' ', text)
    
    # Remove number+letter patterns (e.g., "1095b", "1096a", "1095a")
    text = re.sub(r'\b\d{3,}[a-z]\b', '', text)
    
    # Remove page numbers and formatting artifacts
    # Patterns like: "10 15 20 25 30 351157b" (sequences of numbers)
    text = re.sub(r'\b\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+[a-z]?\b', '', text)
    # Shorter sequences: "10 15 20 25 30"
    text = re.sub(r'\b\d+\s+\d+\s+\d+\s+\d+\s+\d+\b', '', text)
    # Even shorter: "10 15 20" or "5 10 15 20"
    text = re.sub(r'\b\d+\s+\d+\s+\d+\s+\d+\b', '', text)
    # Very short: "5 10 15" or "10 15"
    text = re.sub(r'\b\d+\s+\d+\s+\d+\b', '', text)
    text = re.sub(r'\b\d+\s+\d+\b', '', text)
    
    # File names and paths: "DSHPC081-2_Body_p001-203.indd" or "DSHPC081-2_Body_p001-203. indd"
    text = re.sub(r'\b[A-Z0-9_-]+\.\s*(indd|pdf|txt|docx?)\b', '', text, flags=re.IGNORECASE)
    # Also handle without the dot before extension
    text = re.sub(r'\b[A-Z0-9_-]+\s+(indd|pdf|txt|docx?)\b', '', text, flags=re.IGNORECASE)
    
    # Dates in various formats: "22/06/19 3:15 PM" or "22/06/19 3: 14 PM" (with space in time)
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+\d{1,2}:\s*\d{2}\s*(AM|PM)?\b', '', text, flags=re.IGNORECASE)
    # Also handle dates without time: "22/06/19"
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '', text)
    
    # Remove time patterns followed by numbers/letters: ":14 PM 711125b" or ":14 PM 351127a5"
    text = re.sub(r':\s*\d{1,2}\s*(AM|PM)\s+\d+[a-z]?\d*[a-z]?\b', '', text, flags=re.IGNORECASE)
    # Remove numbers/letters followed by time patterns: "711125b :14 PM" or "351127a5 :14 PM"
    text = re.sub(r'\b\d+[a-z]?\d*[a-z]?\s+:\s*\d{1,2}\s*(AM|PM)\b', '', text, flags=re.IGNORECASE)
    # Remove standalone time patterns with spaces: ":14 PM" (when not part of a date)
    text = re.sub(r':\s*\d{1,2}\s*(AM|PM)\b', '', text, flags=re.IGNORECASE)
    
    # Book/chapter markers: "147Book VIII, Chapter 5" or "4Book I, Chapter 4"
    text = re.sub(r'\b\d+Book\s+[IVXLC]+\s*,\s*Chapter\s+\d+\b', '', text, flags=re.IGNORECASE)
    # Also handle without "Chapter": "4Book I"
    text = re.sub(r'\b\d+Book\s+[IVXLC]+\b', '', text, flags=re.IGNORECASE)
    
    # Standalone long number sequences (likely page numbers): "351157b", "41157", "251095a"
    text = re.sub(r'\b\d{5,}[a-z]?\b', '', text)
    
    # Remove patterns like "711125b" (6+ digits + letter) or "351127a5" (6+ digits + letter + digit)
    text = re.sub(r'\b\d{6,}[a-z]\d*[a-z]?\b', '', text)
    
    # Remove standalone single/double/triple digit numbers that are likely page numbers
    # Pattern: space, 1-3 digits, space (but not part of words or dates)
    text = re.sub(r'\s+\d{1,3}\s+', ' ', text)
    
    # Remove numbers immediately after punctuation: "species.10" -> "species."
    text = re.sub(r'([.,;:!?])\d{1,3}(?=\s|$|[A-Za-z])', r'\1', text)
    
    # Remove numbers at start of sentences/paragraphs: "10 Base people" -> "Base people"
    text = re.sub(r'^\s*\d+\s+', '', text, flags=re.MULTILINE)
    
    # Remove numbers at end of lines (likely page numbers)
    text = re.sub(r'\s+\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Remove numbers attached directly to words (e.g., "nothing5" -> "nothing", "word10" -> "word")
    # Pattern: word ending in letter + 1-3 digits + (optional hyphen + word or end)
    text = re.sub(r'([a-zA-Z])\d{1,3}(?=-|$|\s)', r'\1', text)
    
    # Remove very short words (1-2 letters) after hyphens that are likely formatting artifacts
    # Pattern: word-hyphen-very-short-word at end or before punctuation
    # This handles cases like "nothing5-if" -> "nothing-if" -> "nothing"
    text = re.sub(r'([a-zA-Z]+)-\s*([a-zA-Z]{1,2})(?=[\s.,;:!?]|$)', r'\1', text)
    
    # Remove standalone numbers before words (even if after punctuation): "species.10 Base" -> "species. Base"
    text = re.sub(r'([.,;:!?])\s*\d{1,3}\s+([A-Za-z])', r'\1 \2', text)
    
    # Replace ligatures (common in PDFs): ﬁ → fi, ﬂ → fl, etc.
    ligature_map = {
        'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
        'æ': 'ae', 'œ': 'oe', 'Æ': 'AE', 'Œ': 'OE',
    }
    for ligature, replacement in ligature_map.items():
        text = text.replace(ligature, replacement)
    
    # Replace em dashes (—) and en dashes (–) with hyphens for consistency
    text = text.replace('—', '-').replace('–', '-')
    
    # First, handle hyphenation at line breaks (e.g., "properly-\nspeaking" -> "properly speaking")
    # Remove hyphens that are followed by newline/whitespace and a lowercase letter
    text = re.sub(r'-\s+([a-z])', r'\1', text)
    # Also handle hyphens at end of line followed by newline
    text = re.sub(r'-\s*\n\s*([a-z])', r'\1', text)
    
    # Remove ALL newlines, carriage returns, and other line breaks - make everything continuous
    text = re.sub(r'[\r\n]+', ' ', text)
    
    # Handle hyphens that incorrectly split words (e.g., "thought-involuntary" -> "thought involuntary")
    # Also handle line break hyphens like "gen-eral" -> "general"
    # Replace hyphens between two words (both sides have letters) with a space
    text = re.sub(r'([a-zA-Z]+)-([a-zA-Z]+)', r'\1 \2', text)
    
    # Normalize hyphen spacing (handle "evil for evil-and" vs "evil for evil—and")
    # Replace hyphens with spaces around them with just hyphen (no spaces)
    text = re.sub(r'\s*-\s*', '-', text)
    
    # Handle common word splitting issues
    # "cana not" -> "cannot", "can not" -> "cannot"
    text = re.sub(r'\bcana\s+not\b', 'cannot', text, flags=re.IGNORECASE)
    text = re.sub(r'\bcan\s+not\b', 'cannot', text, flags=re.IGNORECASE)
    
    # Replace all remaining whitespace (spaces, tabs) with single space
    normalized = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    normalized = normalized.strip()
    
    return normalized


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file"""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""


def clean_pdf_to_txt(pdf_path: str, output_txt_path: str = None):
    """
    Clean PDF and convert to TXT with aggressive artifact removal.
    
    Args:
        pdf_path: Path to input PDF file
        output_txt_path: Path to output TXT file (default: same name as PDF with .txt extension)
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return False
    
    if output_txt_path is None:
        output_txt_path = pdf_path.with_suffix('.txt')
    else:
        output_txt_path = Path(output_txt_path)
    
    print(f"Extracting text from: {pdf_path.name}")
    raw_text = extract_text_from_pdf(str(pdf_path))
    
    if not raw_text:
        print("Error: No text extracted from PDF")
        return False
    
    print(f"Raw text length: {len(raw_text)} characters")
    print(f"Cleaning text...")
    
    # Clean the text
    cleaned_text = clean_text_aggressive(raw_text)
    
    print(f"Cleaned text length: {len(cleaned_text)} characters")
    print(f"Removed: {len(raw_text) - len(cleaned_text)} characters ({100 * (len(raw_text) - len(cleaned_text)) / len(raw_text):.1f}%)")
    
    # Write cleaned text to file
    try:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        print(f"Cleaned text saved to: {output_txt_path}")
        return True
    except Exception as e:
        print(f"Error writing cleaned text: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_pdf_to_txt.py <pdf_path> [output_txt_path]")
        print("\nExample:")
        print("  python clean_pdf_to_txt.py Irwin.pdf")
        print("  python clean_pdf_to_txt.py Irwin.pdf Irwin_cleaned.txt")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = clean_pdf_to_txt(pdf_path, output_path)
    sys.exit(0 if success else 1)

