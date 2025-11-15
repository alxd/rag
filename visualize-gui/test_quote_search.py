"""
Standalone script to test and improve quote search algorithm.
Extracts quotes from DOCX file and searches them in corresponding TXT files.
"""

import os
import re
import unicodedata
from pathlib import Path
from typing import List, Tuple, Dict, Optional


class QuoteSearchTester:
    """Test and improve quote search algorithm iteratively"""
    
    def __init__(self):
        self.stats = {
            'total_quotes': 0,
            'found': 0,
            'not_found': 0,
            'by_column': {}
        }
    
    def normalize_text_for_search(self, text: str) -> str:
        """
        Normalize text by removing all extra whitespace, newlines, and normalizing spaces.
        Also handles em dashes, hyphens, word splitting, and PDF/book artifacts.
        Should be used for BOTH source text (cache) and quotes (search).
        """
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
        
        # Remove number+letter patterns (e.g., "1095b", "1096a", "1095a", "251095a")
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
        
        # Handle split words (e.g., "gen eral" -> "general", "in volun tary" -> "involuntary")
        # Pattern: word ending in consonant(s) + space + word starting with vowel(s) that could be one word
        # This is a heuristic - merge if first part ends with consonant and second starts with vowel/letter
        # We'll be conservative and only merge if both parts are short (likely split)
        text = re.sub(r'\b([a-z]{2,4})([bcdfghjklmnpqrstvwxyz])\s+([aeiou][a-z]{2,8})\b', 
                     lambda m: m.group(1) + m.group(2) + m.group(3) if len(m.group(1) + m.group(2) + m.group(3)) <= 12 else m.group(0),
                     text, flags=re.IGNORECASE)
        
        # More aggressive: handle common split patterns
        # "gen eral", "gen erally", "in volun tary", "volun tary", etc.
        common_splits = [
            (r'\bgen\s+eral(ly)?\b', lambda m: 'general' + (m.group(1) if m.group(1) else '')),
            (r'\bin\s+volun\s*tary\b', 'involuntary'),
            (r'\bvolun\s*tary\b', 'voluntary'),
            (r'\bcom\s+pan\s*ions?\b', lambda m: 'companion' + ('s' if m.group(0).endswith('s') else '')),
            (r'\bcom\s+pan\s*ion\b', 'companion'),
        ]
        for pattern, replacement in common_splits:
            if callable(replacement):
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            else:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Replace all remaining whitespace (spaces, tabs) with single space
        normalized = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        normalized = normalized.strip()
        return normalized
    
    def normalize_text_aggressive(self, text: str) -> str:
        """
        More aggressive normalization for flexible matching.
        Handles em dashes, hyphens, word splitting, and punctuation differences.
        """
        # First normalize whitespace (this handles line breaks, hyphens, em dashes, word splitting)
        normalized = self.normalize_text_for_search(text)
        
        # Normalize semicolons to periods for matching (both are sentence separators)
        # This handles cases where quote has "at study" but source has "at study;"
        normalized = re.sub(r';\s*', '. ', normalized)
        
        # Normalize spacing around punctuation - ensure single space after punctuation
        normalized = re.sub(r'\s*([,.:!?])\s*', r'\1 ', normalized)
        
        # Remove trailing punctuation that might differ (periods, commas, semicolons)
        # Do this AFTER normalizing spacing to avoid issues
        normalized = normalized.rstrip('.,;:!?')
        
        # Remove extra spaces that might remain
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized.strip()
    
    def normalize_pdf_text(self, text: str) -> str:
        """Enhanced normalization specifically for PDF-extracted text (also useful for TXT)"""
        # Step 1: Normalize Unicode characters (handles ligatures, smart quotes, etc.)
        text = unicodedata.normalize('NFKD', text)
        
        # Step 2: Replace common PDF artifacts
        ligature_map = {
            'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
            '–': '-', '—': '-',  # En/em dashes to hyphen
            ''': "'", ''': "'",  # Smart quotes to straight quotes
            '"': '"', '"': '"',  # Smart double quotes
            '…': '...',  # Ellipsis
        }
        for old, new in ligature_map.items():
            text = text.replace(old, new)
        
        # Step 3: Handle hyphenation at line breaks more aggressively
        text = re.sub(r'([a-zA-Z])-\s+([a-z])', r'\1\2', text)
        text = re.sub(r'([a-zA-Z])-\s*\n\s*([a-z])', r'\1\2', text)
        text = re.sub(r'([a-zA-Z])-\s*\r\s*([a-z])', r'\1\2', text)
        
        # Step 4: Fix common spacing issues
        text = re.sub(r'\s+([,.;:!?])', r'\1', text)
        text = re.sub(r'([,.;:!?])([^\s\'"])', r'\1 \2', text)
        
        # Step 5: Handle line breaks and normalize whitespace
        text = re.sub(r'[\r\n]+', ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Step 6: Fix word concatenation issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Step 7: Remove zero-width spaces and other invisible characters
        text = re.sub(r'[\u200b-\u200d\ufeff]', '', text)
        
        # Step 8: Final whitespace normalization
        normalized = text.strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized
    
    def fuzzy_match_quote_in_normalized(self, quote_text: str, normalized_source: str) -> tuple[bool, str]:
        """
        Match quote against normalized source data using efficient substring matching
        Returns: (found: bool, matched_text: str) - matched_text is the actual text from source that matched
        """
        if not normalized_source:
            return False, ""
        
        # Clean quote text - remove "Quote X:" prefix and source info
        if quote_text.startswith("Quote "):
            parts = quote_text.split(":", 1)
            if len(parts) > 1:
                quote_text = parts[1].strip()
        
        # Normalize the quote text
        # First remove XML-like tags from quote (e.g., "<rather>" -> "rather")
        # This must happen BEFORE normalization to ensure tags are removed
        quote_text_original = quote_text
        quote_text = re.sub(r'<([^>]+)>', r'\1', quote_text)
        
        # Also remove XML tags that might have been normalized differently
        quote_text = re.sub(r'&lt;([^&]+)&gt;', r'\1', quote_text)  # Handle HTML entities
        
        clean_quote = self.normalize_text_for_search(quote_text)
        clean_quote = self.clean_source_info_from_quote(clean_quote)
        
        if not clean_quote or len(clean_quote) < 10:
            return False, ""
        
        # Strategy 1: Exact normalized match (fastest and most accurate)
        idx = normalized_source.find(clean_quote)
        if idx != -1:
            matched_text = self._extract_matched_text_with_context(normalized_source, idx, len(clean_quote), quote_text=clean_quote, context_chars=100)
            return True, matched_text
        
        # Strategy 2: Case-insensitive exact match
        clean_quote_lower = clean_quote.lower()
        normalized_source_lower = normalized_source.lower()
        idx = normalized_source_lower.find(clean_quote_lower)
        if idx != -1:
            matched_text = self._extract_matched_text_with_context(normalized_source, idx, len(clean_quote), quote_text=clean_quote, context_chars=100)
            return True, matched_text
        
        # Strategy 3: Try removing common words and matching significant words only
        # Extract significant words (3+ chars) and try to find them in sequence
        quote_words = self._extract_significant_words(clean_quote)
        if len(quote_words) >= 5:  # Need at least 5 significant words
            # Try to find a substring that contains these words in order
            match_result = self._find_words_in_sequence_fast(quote_words, normalized_source)
            if match_result:
                start_idx, end_idx = match_result
                source_words = normalized_source.split()
                # Extend to show at least as many words as the quote, or more
                quote_word_count = len(clean_quote.split())
                extend_words = max(quote_word_count, end_idx - start_idx + 10)
                context_start = max(0, start_idx - 5)
                context_end = min(len(source_words), start_idx + extend_words + 5)
                matched_words = source_words[context_start:context_end]
                matched_text = ' '.join(matched_words)
                return True, matched_text
            
            # Strategy 3b: Try matching without first few words (handles quotes with extra words at start)
            # Try skipping first 1-3 words and see if the rest matches
            if len(quote_words) >= 6:  # Need enough words to skip some
                for skip_count in range(1, min(4, len(quote_words) - 4)):  # Skip 1-3 words, keep at least 4
                    remaining_words = quote_words[skip_count:]
                    if len(remaining_words) >= 4:
                        match_result = self._find_words_in_sequence_fast(remaining_words, normalized_source)
                        if match_result:
                            start_idx, end_idx = match_result
                            source_words = normalized_source.split()
                            quote_word_count = len(clean_quote.split())
                            extend_words = max(quote_word_count, end_idx - start_idx + 10)
                            context_start = max(0, start_idx - 5)
                            context_end = min(len(source_words), start_idx + extend_words + 5)
                            matched_words = source_words[context_start:context_end]
                            matched_text = ' '.join(matched_words)
                            return True, matched_text
        
        # Strategy 4: Try aggressive normalization (handles punctuation differences)
        # Use clean_quote instead of quote_text to ensure consistent normalization
        agg_quote = self.normalize_text_aggressive(clean_quote)
        agg_source = self.normalize_text_aggressive(normalized_source)
        agg_quote_lower = agg_quote.lower()
        agg_source_lower = agg_source.lower()
        
        idx = agg_source_lower.find(agg_quote_lower)
        if idx != -1:
            # Map back to original source
            quote_words_agg = self._extract_significant_words(agg_quote)
            if len(quote_words_agg) >= 5:
                match_result = self._find_words_in_sequence_fast(quote_words_agg, normalized_source)
                if match_result:
                    start_idx, end_idx = match_result
                    source_words = normalized_source.split()
                    # Extend to show at least as many words as the quote, or more
                    quote_word_count = len(clean_quote.split())
                    extend_words = max(quote_word_count, end_idx - start_idx + 10)
                    context_start = max(0, start_idx - 5)
                    context_end = min(len(source_words), start_idx + extend_words + 5)
                    matched_words = source_words[context_start:context_end]
                    matched_text = ' '.join(matched_words)
                    return True, matched_text
        
        return False, ""
    
    def _extract_significant_words(self, text: str) -> list:
        """Extract significant words from text (filter out very short/common words)"""
        words = []
        for w in text.split():
            cleaned = w.strip('.,;:!?()[]{}"\'').lower()
            # Only include words with 3+ characters (filters out "a", "an", "the", "is", etc.)
            if len(cleaned) >= 3:
                words.append(cleaned)
        return words
    
    def _find_words_in_sequence_fast(self, quote_words: list, source_text: str) -> tuple:
        """
        Fast algorithm to find quote words in sequence in source text.
        Returns (start_word_idx, end_word_idx) or None if not found.
        Uses a single pass through the source text - much faster than nested loops.
        Handles split words (e.g., "gen eral" matches "general").
        """
        if not quote_words or len(quote_words) < 3:
            return None
        
        source_words = source_text.split()
        if len(source_words) < len(quote_words):
            return None
        
        # Build a map of word -> list of positions for fast lookup
        # Also check for split words by combining consecutive words
        word_positions = {}
        for idx, word in enumerate(source_words):
            cleaned = word.strip('.,;:!?()[]{}"\'').lower()
            if len(cleaned) >= 3:  # Only significant words
                if cleaned not in word_positions:
                    word_positions[cleaned] = []
                word_positions[cleaned].append(idx)
            
            # Check if this word combined with next word matches any quote word
            if idx + 1 < len(source_words):
                next_word = source_words[idx + 1].strip('.,;:!?()[]{}"\'').lower()
                combined = cleaned + next_word
                if len(combined) >= 4:  # Only for reasonable combined lengths
                    if combined not in word_positions:
                        word_positions[combined] = []
                    word_positions[combined].append(idx)  # Store position of first word
        
        # Try to find quote words in sequence
        # Start from each position of the first quote word
        first_word = quote_words[0]
        if first_word not in word_positions:
            return None
        
        for start_pos in word_positions[first_word]:
            current_pos = start_pos
            matched_positions = [start_pos]
            
            # Try to find remaining words in sequence
            for word_idx in range(1, len(quote_words)):
                word = quote_words[word_idx]
                
                # Try exact match first
                found = False
                if word in word_positions:
                    for pos in word_positions[word]:
                        if pos > current_pos and pos <= current_pos + 10:  # Allow up to 10 words gap
                            matched_positions.append(pos)
                            current_pos = pos
                            found = True
                            break
                
                # If not found, try matching against split words (combined consecutive words)
                if not found:
                    # Check if we can match by combining source words
                    for check_pos in range(current_pos + 1, min(current_pos + 3, len(source_words))):
                        # Try combining 2 words
                        if check_pos < len(source_words):
                            w1 = source_words[check_pos].strip('.,;:!?()[]{}"\'').lower()
                            if check_pos + 1 < len(source_words):
                                w2 = source_words[check_pos + 1].strip('.,;:!?()[]{}"\'').lower()
                                combined = w1 + w2
                                if combined == word:
                                    matched_positions.append(check_pos)
                                    current_pos = check_pos + 1  # Skip next word since we used it
                                    found = True
                                    break
                        if found:
                            break
                
                if not found:
                    break
            
            # If we found all words in a reasonable sequence
            if len(matched_positions) == len(quote_words):
                span = matched_positions[-1] - matched_positions[0]
                # Check if span is reasonable (not too spread out)
                if span <= len(quote_words) * 3:
                    return (matched_positions[0], matched_positions[-1])
            
            # Also allow partial matches if we found most words (e.g., 80%+) and there are extra words at the end
            # This handles cases where the source has extra words like "nothing if" when quote has "nothing."
            if len(matched_positions) >= int(len(quote_words) * 0.8) and len(matched_positions) >= 5:
                span = matched_positions[-1] - matched_positions[0]
                # Allow slightly more spread for partial matches
                if span <= len(quote_words) * 4:
                    return (matched_positions[0], matched_positions[-1])
        
        return None
    
    def _extract_matched_text_with_context(self, source: str, start_idx: int, length: int, quote_text: str = "", context_chars: int = 100) -> str:
        """
        Extract matched text with context before and after.
        Extends to show at least as many words as the quote, or more.
        """
        if start_idx == -1:
            return ""
        
        # Calculate how much to extend - show at least as many words as the quote, or more
        quote_word_count = len(quote_text.split()) if quote_text else 0
        # Extend by at least the quote length, plus some context
        extend_chars = max(length, quote_word_count * 8)  # ~8 chars per word average
        
        # Add context before
        context_start = max(0, start_idx - context_chars)
        # Extend after to show more of the quote or at least the quote length
        context_end = min(len(source), start_idx + extend_chars + context_chars)
        
        matched_text = source[context_start:context_end]
        return matched_text.strip()
    
    def _highlight_match_in_text(self, matched_text: str, quote_text: str) -> str:
        """
        Highlight the matching portion (quote) in the matched text using ANSI color codes.
        Returns text with green highlighting for console output.
        """
        if not matched_text or not quote_text:
            return matched_text
        
        # Normalize both for comparison
        matched_lower = matched_text.lower()
        quote_lower = quote_text.lower()
        
        # Try to find the quote in the matched text
        idx = matched_lower.find(quote_lower)
        if idx != -1:
            # Found exact match - highlight it
            before = matched_text[:idx]
            match_portion = matched_text[idx:idx+len(quote_text)]
            after = matched_text[idx+len(quote_text):]
            
            # Use ANSI color codes for green (works in most terminals)
            GREEN = '\033[92m'  # Bright green
            RESET = '\033[0m'   # Reset color
            
            return f"{before}{GREEN}{match_portion}{RESET}{after}"
        else:
            # Try to find significant words from quote in matched text
            quote_words = self._extract_significant_words(quote_text)
            if len(quote_words) >= 3:
                # Find where these words appear and highlight them
                words_to_highlight = set(quote_words)
                matched_words = matched_text.split()
                highlighted_words = []
                
                for word in matched_words:
                    cleaned = word.strip('.,;:!?()[]{}"\'').lower()
                    if cleaned in words_to_highlight:
                        GREEN = '\033[92m'
                        RESET = '\033[0m'
                        highlighted_words.append(f"{GREEN}{word}{RESET}")
                    else:
                        highlighted_words.append(word)
                
                return ' '.join(highlighted_words)
        
        return matched_text
    
    def _highlight_match_in_docx(self, matched_text: str, quote_text: str, paragraph) -> bool:
        """
        Highlight the matching portion (quote) in the matched text for DOCX output.
        Returns True if highlighting was applied, False otherwise.
        """
        if not matched_text or not quote_text:
            return False
        
        # Import required modules
        from docx.shared import RGBColor, Inches
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
        
        # Normalize both for comparison
        matched_lower = matched_text.lower()
        quote_lower = quote_text.lower()
        
        # Try to find the quote in the matched text
        idx = matched_lower.find(quote_lower)
        if idx != -1:
            # Found exact match - highlight it
            before = matched_text[:idx]
            match_portion = matched_text[idx:idx+len(quote_text)]
            after = matched_text[idx+len(quote_text):]
            
            # Add text with highlighting
            if before:
                run_before = paragraph.add_run(f"  → Matched: {before}")
                run_before.font.size = Inches(0.08)
                run_before.font.italic = True
                run_before.font.color.rgb = RGBColor(0, 0, 0)  # Black
            
            # Highlight the matching portion in green
            run_match = paragraph.add_run(match_portion)
            run_match.font.size = Inches(0.08)
            run_match.font.italic = True
            run_match.font.color.rgb = RGBColor(255, 255, 255)  # White text
            rPr = run_match._element.get_or_add_rPr()
            shd = OxmlElement('w:shd')
            shd.set(qn('w:fill'), '00FF00')  # Bright green
            shd.set(qn('w:val'), 'clear')
            rPr.append(shd)
            
            if after:
                run_after = paragraph.add_run(f"{after[:100]}{'...' if len(after) > 100 else ''}")
                run_after.font.size = Inches(0.08)
                run_after.font.italic = True
                run_after.font.color.rgb = RGBColor(0, 0, 0)  # Black
            
            return True
        
        return False
    
    def clean_source_info_from_quote(self, quote: str) -> str:
        """Remove source information and page numbers from quote text"""
        import re
        # Remove patterns like "Sources: Source 1: Temp=0.5, Top-p=0.95, Top-k=50, BM25=0.0 (BM25 Sweep)"
        # Remove "Sources:" and everything after it
        quote = re.sub(r'\s*Sources?:.*$', '', quote, flags=re.IGNORECASE | re.MULTILINE)
        # Remove "Source X:" patterns
        quote = re.sub(r'\s*Source\s+\d+:\s*.*$', '', quote, flags=re.IGNORECASE | re.MULTILINE)
        # Remove parameter patterns like "Temp=0.5, Top-p=0.95, Top-k=50, BM25=0.0"
        quote = re.sub(r'\s*Temp=[\d.]+[,\s]*Top-p=[\d.]+[,\s]*Top-k=\d+[,\s]*BM25=[\d.]+.*$', '', quote, flags=re.IGNORECASE)
        # Remove "(BM25 Sweep)" or similar patterns
        quote = re.sub(r'\s*\([^)]*Sweep[^)]*\)', '', quote, flags=re.IGNORECASE)
        
        # Remove page numbers in parentheses or brackets at the end of the quote
        # Patterns like: [p.92], [p. 92], (p.92), (p. 92), [page 92], (page 92), etc.
        quote = re.sub(r'\s*\[p\.?\s*\d+\]', '', quote, flags=re.IGNORECASE)
        quote = re.sub(r'\s*\(p\.?\s*\d+\)', '', quote, flags=re.IGNORECASE)
        quote = re.sub(r'\s*\[page\s+\d+\]', '', quote, flags=re.IGNORECASE)
        quote = re.sub(r'\s*\(page\s+\d+\)', '', quote, flags=re.IGNORECASE)
        # Also handle patterns like [92], (92) at the end (likely page numbers)
        quote = re.sub(r'\s*\[\d+\]\s*$', '', quote)
        quote = re.sub(r'\s*\(\d+\)\s*$', '', quote)
        
        return quote.strip()
    
    def load_and_normalize_txt(self, txt_path: str) -> Optional[str]:
        """Load and normalize a TXT file - MATCH MAIN CODE EXACTLY"""
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            # Use the same normalization as main code for TXT files
            # Main code uses _normalize_text_for_search() for TXT, not PDF normalization
            return self.normalize_text_for_search(content)
        except Exception as e:
            print(f"Error loading TXT file {txt_path}: {e}")
            return None
    
    def extract_quotes_from_docx(self, docx_path: str) -> Dict[str, List[str]]:
        """
        Extract quotes from DOCX file.
        Returns a dictionary mapping column names to lists of quotes.
        """
        try:
            import docx
            doc = docx.Document(docx_path)
            quotes_by_column = {}
            
            print(f"\nExtracting quotes from {os.path.basename(docx_path)}")
            print(f"Found {len(doc.tables)} tables")
            
            for table_idx, table in enumerate(doc.tables):
                if len(table.rows) == 0:
                    continue
                
                # Try to identify column headers
                header_row = table.rows[0]
                headers = [cell.text.strip() for cell in header_row.cells]
                print(f"\nTable {table_idx + 1} headers: {headers}")
                
                # For Table 1 (index 0), extract from all columns except the first (which is usually "Concept")
                # For other tables, look for "Concept Citations" and "Quotes" columns
                quote_column_indices = []
                
                if table_idx == 0:
                    # Table 1: Skip first column (usually "Concept"), take all others (book names like "Chase", "Ross")
                    for idx in range(1, len(headers)):
                        if headers[idx].strip():  # Only add non-empty headers
                            quote_column_indices.append(idx)
                            print(f"  Found quote column at index {idx}: '{headers[idx]}'")
                else:
                    # Other tables: Look for "Concept Citations" or "Quotes"
                    for idx, header in enumerate(headers):
                        header_lower = header.lower().strip()
                        if header_lower == "concept citations" or header_lower == "quotes":
                            quote_column_indices.append(idx)
                            print(f"  Found target column at index {idx}: '{header}'")
                
                # Extract quotes from identified columns
                for col_idx in quote_column_indices:
                    if col_idx < len(headers):
                        column_name = headers[col_idx] if col_idx < len(headers) else f"Column_{col_idx}"
                        if column_name not in quotes_by_column:
                            quotes_by_column[column_name] = []
                        
                        # Extract quotes from this column
                        # Match main code behavior: quotes in cells can be in separate paragraphs
                        # OR multiple quotes can be in the same paragraph/text separated by "Quote X:" patterns
                        for row_idx, row in enumerate(table.rows[1:], start=1):  # Skip header
                            if col_idx < len(row.cells):
                                cell = row.cells[col_idx]
                                
                                # Get all text from the cell (combine all paragraphs)
                                cell_text = cell.text.strip()
                                
                                if not cell_text or cell_text in ["No quotes found", "No citation available"]:
                                    continue
                                
                                # Split cell text by "Quote X:" pattern to extract all quotes
                                # Pattern: "Quote " followed by digits and ":"
                                import re
                                
                                # Find all "Quote X:" patterns in the text
                                quote_pattern = r'(Quote\s+\d+\s*:)'
                                matches = list(re.finditer(quote_pattern, cell_text, re.IGNORECASE))
                                
                                if matches:
                                    # Extract each quote starting from each "Quote X:" marker
                                    for i, match in enumerate(matches):
                                        start_pos = match.start()
                                        # Find the end: either next "Quote X:" or end of text
                                        if i + 1 < len(matches):
                                            end_pos = matches[i + 1].start()
                                            quote_text = cell_text[start_pos:end_pos].strip()
                                        else:
                                            quote_text = cell_text[start_pos:].strip()
                                        
                                        # Filter out invalid quotes
                                        quote_stripped = quote_text.strip()
                                        if (not quote_stripped or 
                                            quote_stripped == "No quotes found" or 
                                            quote_stripped == "No citation available" or
                                            (quote_stripped.isdigit() and len(quote_stripped) <= 3)):
                                            continue
                                        
                                        # Only add quotes that have "Quote X:" format
                                        if quote_stripped.startswith("Quote ") and ":" in quote_stripped:
                                            quotes_by_column[column_name].append(quote_stripped)
                                else:
                                    # No "Quote X:" pattern found, check if it's a valid quote anyway
                                    # (for backward compatibility with quotes without prefix)
                                    quote_stripped = cell_text.strip()
                                    if (quote_stripped and 
                                        quote_stripped not in ["No quotes found", "No citation available"] and
                                        not (quote_stripped.isdigit() and len(quote_stripped) <= 3)):
                                        # Only add if it looks like a quote (has reasonable length)
                                        if len(quote_stripped) > 10:
                                            quotes_by_column[column_name].append(quote_stripped)
            
            # Print summary
            for col_name, quotes in quotes_by_column.items():
                print(f"\nColumn '{col_name}': {len(quotes)} quotes extracted")
            
            return quotes_by_column
            
        except Exception as e:
            print(f"Error extracting quotes from DOCX: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def find_txt_file(self, base_dir: str, column_name: str) -> Optional[str]:
        """Find the corresponding TXT file for a column name"""
        # Try different patterns: column_name_search.TXT, column_name.TXT, etc.
        patterns = [
            f"{column_name}_search.TXT",
            f"{column_name}_search.txt",
            f"{column_name}.TXT",
            f"{column_name}.txt",
        ]
        
        for pattern in patterns:
            txt_path = os.path.join(base_dir, pattern)
            if os.path.exists(txt_path):
                return txt_path
        
        # Also try case-insensitive search
        for file in os.listdir(base_dir):
            if file.upper().endswith('.TXT') or file.upper().endswith('.txt'):
                file_base = os.path.splitext(file)[0]
                if column_name.lower() in file_base.lower() or file_base.lower() in column_name.lower():
                    return os.path.join(base_dir, file)
        
        return None
    
    def test_quotes(self, docx_path: str, base_dir: Optional[str] = None) -> Dict:
        """
        Main test function: Extract quotes from DOCX and search in TXT files.
        
        Args:
            docx_path: Path to the DOCX file with quotes
            base_dir: Directory containing the TXT files (defaults to same dir as DOCX)
        
        Returns:
            Dictionary with test results
        """
        if base_dir is None:
            base_dir = os.path.dirname(docx_path)
        
        # Extract quotes from DOCX
        quotes_by_column = self.extract_quotes_from_docx(docx_path)
        
        if not quotes_by_column:
            print("No quotes extracted from DOCX file!")
            return {
                'total_quotes': 0,
                'found': 0,
                'not_found': 0,
                'by_column': {},
                'details': []
            }
        
        results = {
            'total_quotes': 0,
            'found': 0,
            'not_found': 0,
            'by_column': {},
            'details': []
        }
        
        # Test each column's quotes
        for column_name, quotes in quotes_by_column.items():
            print(f"\n{'='*80}")
            print(f"Testing quotes from column: '{column_name}'")
            print(f"{'='*80}")
            
            # Find corresponding TXT file
            txt_path = self.find_txt_file(base_dir, column_name)
            if not txt_path:
                print(f"  WARNING: Could not find TXT file for column '{column_name}'")
                print(f"  Searched in: {base_dir}")
                continue
            
            print(f"  Using TXT file: {os.path.basename(txt_path)}")
            
            # Load and normalize TXT file
            normalized_text = self.load_and_normalize_txt(txt_path)
            if not normalized_text:
                print(f"  ERROR: Could not load TXT file")
                continue
            
            print(f"  Loaded {len(normalized_text)} characters")
            
            # Initialize column stats - will be updated as we process quotes
            column_stats = {
                'total': 0,  # Will count as we process
                'found': 0,
                'not_found': 0,
                'not_found_quotes': []
            }
            
            # Test each quote
            for quote_idx, quote in enumerate(quotes, 1):
                # Only process quotes that start with "Quote X:" format
                quote_trimmed = quote.strip()
                
                # Check if quote is empty or "Not found"
                if not quote_trimmed or quote_trimmed.lower() in ["not found", "no quotes found", "no citation available"]:
                    # Don't add to details - skip silently
                    continue
                
                # Check if quote has "Quote X:" format - this is required
                if not quote_trimmed.startswith("Quote "):
                    # Don't add to details - skip silently
                    continue
                
                # Extract the actual quote text
                # IMPORTANT: The main code passes the full quote text (including "Quote X:" prefix) to _fuzzy_match_quote_in_normalized
                # Strategy 6 in the main code handles stripping the prefix, so we should pass it the same way
                parts = quote_trimmed.split(":", 1)
                if len(parts) < 2:
                    # No colon found, skip
                    continue
                
                quote_after_colon = parts[1].strip()
                if not quote_after_colon:
                    # Empty after colon, skip
                    continue
                
                # Strip source info from the quote (source info would never be in original text)
                clean_quote = self.clean_source_info_from_quote(quote_after_colon)
                
                # This is a valid quote to search
                results['total_quotes'] += 1
                column_stats['total'] += 1  # Count valid quotes as we process them
                
                # IMPORTANT: Pass the quote WITH "Quote X:" prefix to match main code behavior
                # The main code's _fuzzy_match_quote_in_normalized has Strategy 6 that handles this
                # But we need to reconstruct it with the cleaned quote text (without source info)
                quote_to_search = f"Quote {parts[0].split()[-1] if parts[0].split() else '1'}: {clean_quote}"
                
                # However, the main code might also pass quotes without prefix, so let's try both
                # But first try with prefix (matching main code's Strategy 6 behavior)
                found, matched_text = self.fuzzy_match_quote_in_normalized(quote_to_search, normalized_text)
                
                # If not found with prefix, try without prefix (just the clean quote)
                if not found:
                    found, matched_text = self.fuzzy_match_quote_in_normalized(clean_quote, normalized_text)
                
                if found:
                    results['found'] += 1
                    column_stats['found'] += 1
                    print(f"  ✓ [{quote_idx}/{len(quotes)}] FOUND")
                    if matched_text:
                        # Show matched text (no highlighting)
                        matched_snippet = matched_text[:200] + "..." if len(matched_text) > 200 else matched_text
                        print(f"      Matched: {matched_snippet}")
                else:
                    results['not_found'] += 1
                    column_stats['not_found'] += 1
                    column_stats['not_found_quotes'].append(clean_quote)
                    print(f"  ✗ [{quote_idx}/{len(quotes)}] NOT FOUND")
                
                results['details'].append({
                    'column': column_name,
                    'quote': clean_quote,
                    'found': found,
                    'matched_text': matched_text if found else "",
                    'quote_text_for_highlight': clean_quote if found else "",  # Store for highlighting
                    'status': 'FOUND' if found else 'NOT FOUND',
                    'original_quote': quote_trimmed  # Keep original for matching in DOCX
                })
            
            results['by_column'][column_name] = column_stats
            
            # Print column summary with debug info
            found_pct = (column_stats['found'] / column_stats['total'] * 100) if column_stats['total'] > 0 else 0
            print(f"\n  Column '{column_name}' Summary:")
            print(f"    Total quotes extracted: {len(quotes)}")
            print(f"    Valid quotes (with 'Quote X:' format): {column_stats['total']}")
            print(f"    Found: {column_stats['found']}/{column_stats['total']} ({found_pct:.1f}%)")
            print(f"    Not Found: {column_stats['not_found']}/{column_stats['total']} ({100-found_pct:.1f}%)")
        
        # Print overall summary with stats table
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY")
        print(f"{'='*80}")
        print(f"Total quotes tested: {results['total_quotes']}")
        print(f"Found: {results['found']}")
        print(f"Not found: {results['not_found']}")
        if results['total_quotes'] > 0:
            overall_pct = (results['found'] / results['total_quotes'] * 100)
            print(f"Success rate: {overall_pct:.1f}%")
        
        # Print stats table similar to original format
        print(f"\n{'='*80}")
        print("STATISTICS BY BOOK/FOLDER")
        print(f"{'='*80}")
        print(f"{'Book/Folder':<30} {'Found':<10} {'Not Found':<12} {'Found %':<10}")
        print("-" * 80)
        
        for column_name, stats in sorted(results['by_column'].items()):
            found_count = stats['found']
            not_found_count = stats['not_found']
            total_count = stats['total']
            
            if total_count > 0:
                found_percentage = (found_count / total_count) * 100
            else:
                found_percentage = 0.0
            
            print(f"{column_name:<30} {found_count:<10} {not_found_count:<12} {found_percentage:.1f}%")
        
        # Overall row
        if results['total_quotes'] > 0:
            overall_pct = (results['found'] / results['total_quotes'] * 100)
            print("-" * 80)
            print(f"{'TOTAL':<30} {results['found']:<10} {results['not_found']:<12} {overall_pct:.1f}%")
        
        return results


def main():
    """Main entry point for testing"""
    import sys
    
    # Default paths
    default_docx = r"rag5_clean_embedding_1024\Results\Mistral\aggregated_words_20251115_orig.docx"
    default_base_dir = r"rag5_clean_embedding_1024\Results\Mistral"
    
    # Get paths from command line or use defaults
    if len(sys.argv) > 1:
        docx_path = sys.argv[1]
    else:
        # Try to find the file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        docx_path = os.path.join(script_dir, default_docx)
        if not os.path.exists(docx_path):
            # Try relative to workspace root
            workspace_root = os.path.dirname(script_dir)
            docx_path = os.path.join(workspace_root, default_docx)
    
    if len(sys.argv) > 2:
        base_dir = sys.argv[2]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(script_dir, default_base_dir)
        if not os.path.exists(base_dir):
            workspace_root = os.path.dirname(script_dir)
            base_dir = os.path.join(workspace_root, default_base_dir)
    
    if not os.path.exists(docx_path):
        print(f"ERROR: DOCX file not found: {docx_path}")
        print(f"\nUsage: python test_quote_search.py [docx_path] [base_dir]")
        return
    
    print(f"DOCX file: {docx_path}")
    print(f"Base directory: {base_dir}")
    
    # Run tests
    tester = QuoteSearchTester()
    results = tester.test_quotes(docx_path, base_dir)
    
    # Save results to file with stats table
    output_file = os.path.join(base_dir, "quote_search_test_results.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("QUOTE SEARCH TEST RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"DOCX file: {docx_path}\n")
        f.write(f"Base directory: {base_dir}\n\n")
        f.write(f"Total quotes: {results['total_quotes']}\n")
        f.write(f"Found: {results['found']}\n")
        f.write(f"Not found: {results['not_found']}\n")
        if results['total_quotes'] > 0:
            f.write(f"Success rate: {results['found'] / results['total_quotes'] * 100:.1f}%\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Write stats table
        f.write("STATISTICS BY BOOK/FOLDER\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Book/Folder':<30} {'Found':<10} {'Not Found':<12} {'Found %':<10}\n")
        f.write("-" * 80 + "\n")
        
        for column_name, stats in sorted(results['by_column'].items()):
            found_count = stats['found']
            not_found_count = stats['not_found']
            total_count = stats['total']
            
            if total_count > 0:
                found_percentage = (found_count / total_count) * 100
            else:
                found_percentage = 0.0
            
            f.write(f"{column_name:<30} {found_count:<10} {not_found_count:<12} {found_percentage:.1f}%\n")
        
        # Overall row
        if results['total_quotes'] > 0:
            overall_pct = (results['found'] / results['total_quotes'] * 100)
            f.write("-" * 80 + "\n")
            f.write(f"{'TOTAL':<30} {results['found']:<10} {results['not_found']:<12} {overall_pct:.1f}%\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Write all quotes with their status (FOUND/NOT FOUND/N/A)
        f.write("\n" + "=" * 80 + "\n\n")
        f.write("QUOTE DETAILS:\n")
        f.write("=" * 80 + "\n")
        
        for column_name in sorted(results['by_column'].keys()):
            f.write(f"\nColumn: {column_name}\n")
            f.write("-" * 80 + "\n")
            
            for detail in results['details']:
                if detail['column'] == column_name:
                    # Clean source information from quote before writing
                    clean_quote = tester.clean_source_info_from_quote(detail['quote'])
                    status = detail.get('status', 'FOUND' if detail.get('found') else 'NOT FOUND')
                    if status == 'N/A' or detail.get('found') is None:
                        f.write(f"N/A: {clean_quote}\n")
                    elif detail.get('found'):
                        f.write(f"FOUND: {clean_quote}\n")
                        # Add matched text from source
                        matched_text = detail.get('matched_text', '')
                        if matched_text:
                            f.write(f"  → Matched in source: {matched_text}\n")
                    else:
                        f.write(f"NOT FOUND: {clean_quote}\n")
    
    print(f"\nResults saved to: {output_file}")
    
    # Create DOCX output with colored quotes
    docx_output_file = os.path.join(base_dir, "quote_search_test_results.docx")
    create_docx_output(docx_path, results, tester, docx_output_file)
    print(f"DOCX results saved to: {docx_output_file}")


def create_docx_output(docx_path: str, results: Dict, tester, output_path: str):
    """Create a DOCX file with the same structure as input, with colored quotes and stats table"""
    try:
        import docx
        from docx.shared import RGBColor, Inches
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
        
        # Read the original DOCX to get structure
        doc_input = docx.Document(docx_path)
        
        # Create new DOCX
        doc_output = docx.Document()
        
        # Only process Table 1 (quotes table) - skip other tables
        if len(doc_input.tables) > 0:
            table_input = doc_input.tables[0]
            
            if len(table_input.rows) > 0:
                # Get headers
                header_row = table_input.rows[0]
                headers = [cell.text.strip() for cell in header_row.cells]
                
                # Create new table with same structure
                table_output = doc_output.add_table(rows=1, cols=len(headers))
                table_output.style = 'Light Grid Accent 1'
                
                # Add headers
                header_cells = table_output.rows[0].cells
                for i, header in enumerate(headers):
                    header_cells[i].text = header
                    # Make header bold
                    for para in header_cells[i].paragraphs:
                        for run in para.runs:
                            run.bold = True
                
                # Create a mapping of quote text to found status
                # Use multiple matching strategies to handle variations
                quote_status_map = {}
                for detail in results['details']:
                    if detail.get('found') is not None:
                        original_quote = detail['quote']
                        original_with_prefix = detail.get('original_quote', '')
                        clean_quote = tester.clean_source_info_from_quote(original_quote)
                        
                        # Store multiple versions for flexible matching
                        quote_status_map[clean_quote.lower().strip()] = detail['found']
                        quote_status_map[original_quote.lower().strip()] = detail['found']
                        if original_with_prefix:
                            # Also try matching with "Quote X:" prefix removed
                            if original_with_prefix.startswith("Quote ") and ":" in original_with_prefix:
                                parts = original_with_prefix.split(":", 1)
                                if len(parts) > 1:
                                    quote_after_colon = parts[1].strip()
                                    clean_after_colon = tester.clean_source_info_from_quote(quote_after_colon)
                                    quote_status_map[clean_after_colon.lower().strip()] = detail['found']
                                    quote_status_map[quote_after_colon.lower().strip()] = detail['found']
                
                # Add data rows
                for row_idx, row_input in enumerate(table_input.rows[1:], start=1):
                    row_output = table_output.add_row()
                    
                    # Copy first column (Concept)
                    if len(row_input.cells) > 0:
                        row_output.cells[0].text = row_input.cells[0].text
                    
                    # Process quote columns (skip first column)
                    for col_idx in range(1, len(headers)):
                        if col_idx < len(row_input.cells):
                            cell_input = row_input.cells[col_idx]
                            cell_output = row_output.cells[col_idx]
                            
                            # Clear output cell
                            cell_output.text = ""
                            
                            # Get all text from input cell
                            cell_text = cell_input.text.strip()
                            
                            if not cell_text or cell_text in ["No quotes found", "No citation available"]:
                                cell_output.text = cell_text
                                continue
                            
                            # Extract all quotes from the cell
                            import re
                            quote_pattern = r'(Quote\s+\d+\s*:)'
                            matches = list(re.finditer(quote_pattern, cell_text, re.IGNORECASE))
                            
                            if matches:
                                # Extract each quote
                                for i, match in enumerate(matches):
                                    start_pos = match.start()
                                    if i + 1 < len(matches):
                                        end_pos = matches[i + 1].start()
                                        quote_text = cell_text[start_pos:end_pos].strip()
                                    else:
                                        quote_text = cell_text[start_pos:].strip()
                                    
                                    # Extract clean quote (remove "Quote X:" prefix and source info)
                                    if quote_text.startswith("Quote ") and ":" in quote_text:
                                        parts = quote_text.split(":", 1)
                                        if len(parts) > 1:
                                            quote_after_colon = parts[1].strip()
                                            # Strip source info
                                            clean_quote = tester.clean_source_info_from_quote(quote_after_colon)
                                            
                                            # Try multiple matching strategies
                                            found = None
                                            # Try 1: Clean quote (with source info stripped)
                                            found = quote_status_map.get(clean_quote.lower().strip(), None)
                                            # Try 2: Original quote after colon (before cleaning)
                                            if found is None:
                                                found = quote_status_map.get(quote_after_colon.lower().strip(), None)
                                            # Try 3: Full quote text with prefix
                                            if found is None:
                                                found = quote_status_map.get(quote_text.lower().strip(), None)
                                            # Try 4: Fuzzy match - check if any stored quote is similar
                                            if found is None:
                                                # Try substring matching
                                                for stored_quote, stored_found in quote_status_map.items():
                                                    if clean_quote.lower().strip() in stored_quote or stored_quote in clean_quote.lower().strip():
                                                        if len(clean_quote) > 20 and len(stored_quote) > 20:  # Only for substantial matches
                                                            found = stored_found
                                                            break
                                            
                                            # Add paragraph for this quote
                                            if i > 0:
                                                para = cell_output.add_paragraph()
                                            else:
                                                para = cell_output.paragraphs[0]
                                            
                                            # Add the quote text with "Quote X:" prefix but without source info
                                            # Preserve the original quote number from the input
                                            quote_num_match = re.search(r'Quote\s+(\d+)', quote_text, re.IGNORECASE)
                                            if quote_num_match:
                                                quote_num = quote_num_match.group(1)
                                                quote_with_prefix = f"Quote {quote_num}: {clean_quote}"
                                            else:
                                                quote_with_prefix = f"Quote {i+1}: {clean_quote}"
                                            
                                            # Find the matched text for this quote
                                            matched_text = ""
                                            for detail in results['details']:
                                                if detail['column'] == headers[col_idx] and detail.get('original_quote', '').strip() == quote_text.strip():
                                                    matched_text = detail.get('matched_text', '')
                                                    break
                                            
                                            run = para.add_run(quote_with_prefix)
                                            run.font.size = Inches(0.09)
                                            
                                            # Color based on found status
                                            if found is True:
                                                run.font.color.rgb = RGBColor(255, 255, 255)  # White text
                                                rPr = run._element.get_or_add_rPr()
                                                shd = OxmlElement('w:shd')
                                                shd.set(qn('w:fill'), '006400')  # Dark green
                                                shd.set(qn('w:val'), 'clear')
                                                rPr.append(shd)
                                                
                                                # Add matched text below the quote if available
                                                if matched_text:
                                                    para_matched = cell_output.add_paragraph()
                                                    # Just show matched text without highlighting
                                                    run_matched = para_matched.add_run(f"  → Matched: {matched_text[:200]}{'...' if len(matched_text) > 200 else ''}")
                                                    run_matched.font.size = Inches(0.08)
                                                    run_matched.font.italic = True
                                                    run_matched.font.color.rgb = RGBColor(0, 100, 0)  # Dark green
                                            elif found is False:
                                                run.font.color.rgb = RGBColor(255, 255, 255)  # White text
                                                rPr = run._element.get_or_add_rPr()
                                                shd = OxmlElement('w:shd')
                                                shd.set(qn('w:fill'), '8B0000')  # Dark red
                                                shd.set(qn('w:val'), 'clear')
                                                rPr.append(shd)
                            else:
                                # No "Quote X:" pattern, just copy the text (strip source info)
                                clean_text = tester.clean_source_info_from_quote(cell_text)
                                cell_output.text = clean_text
        
        # Add stats table with found/not found results
        doc_output.add_heading("Quote Verification Statistics", level=1)
        
        stats_table = doc_output.add_table(rows=1, cols=4)
        stats_table.style = 'Light Grid Accent 1'
        
        # Add header row
        header_cells = stats_table.rows[0].cells
        header_cells[0].text = "Book/Folder"
        header_cells[1].text = "Found"
        header_cells[2].text = "Not Found"
        header_cells[3].text = "Found %"
        
        # Make headers bold
        for cell in header_cells:
            for para in cell.paragraphs:
                for run in para.runs:
                    run.bold = True
        
        # Add data rows
        for column_name, stats in sorted(results['by_column'].items()):
            found_count = stats['found']
            not_found_count = stats['not_found']
            total_count = stats['total']
            
            if total_count > 0:
                found_percentage = (found_count / total_count) * 100
            else:
                found_percentage = 0.0
            
            row = stats_table.add_row()
            row.cells[0].text = column_name
            row.cells[1].text = str(found_count)
            row.cells[2].text = str(not_found_count)
            row.cells[3].text = f"{found_percentage:.1f}%"
            
            # Color code the percentage cell
            found_para = row.cells[3].paragraphs[0]
            found_run = found_para.runs[0] if found_para.runs else found_para.add_run(row.cells[3].text)
            if found_percentage >= 80:
                found_run.font.color.rgb = RGBColor(0, 100, 0)  # Dark green
            elif found_percentage >= 50:
                found_run.font.color.rgb = RGBColor(184, 134, 11)  # Dark goldenrod
            else:
                found_run.font.color.rgb = RGBColor(139, 0, 0)  # Dark red
        
        # Add total row
        if results['total_quotes'] > 0:
            overall_pct = (results['found'] / results['total_quotes'] * 100)
            row = stats_table.add_row()
            row.cells[0].text = "TOTAL"
            row.cells[1].text = str(results['found'])
            row.cells[2].text = str(results['not_found'])
            row.cells[3].text = f"{overall_pct:.1f}%"
            
            # Make total row bold
            for cell in row.cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        run.bold = True
        
        # Add original stats table from DOCX for comparison (Table 2)
        if len(doc_input.tables) > 1:
            doc_output.add_heading("Original Statistics (for comparison)", level=1)
            
            table_input = doc_input.tables[1]  # Table 2 is the stats table
            
            if len(table_input.rows) > 0:
                # Get headers
                header_row = table_input.rows[0]
                headers = [cell.text.strip() for cell in header_row.cells]
                
                # Create new table with same structure
                table_output = doc_output.add_table(rows=len(table_input.rows), cols=len(headers))
                table_output.style = 'Light Grid Accent 1'
                
                # Copy all rows
                for row_idx, row_input in enumerate(table_input.rows):
                    row_output = table_output.rows[row_idx]
                    for col_idx, cell_input in enumerate(row_input.cells):
                        if col_idx < len(row_output.cells):
                            row_output.cells[col_idx].text = cell_input.text
                            
                            # Copy formatting (bold for headers)
                            if row_idx == 0:
                                for para in row_output.cells[col_idx].paragraphs:
                                    for run in para.runs:
                                        run.bold = True
                            
                            # Copy color formatting if present (for percentage cells)
                            if col_idx > 0 and row_idx > 0:  # Skip header row and first column
                                try:
                                    # Try to copy font color from original
                                    if cell_input.paragraphs and cell_input.paragraphs[0].runs:
                                        original_run = cell_input.paragraphs[0].runs[0]
                                        if original_run.font.color and original_run.font.color.rgb:
                                            output_run = row_output.cells[col_idx].paragraphs[0].runs[0] if row_output.cells[col_idx].paragraphs[0].runs else row_output.cells[col_idx].paragraphs[0].add_run(row_output.cells[col_idx].text)
                                            output_run.font.color.rgb = original_run.font.color.rgb
                                except:
                                    pass  # If color copying fails, just use default
        
        # Save the document
        doc_output.save(output_path)
        
    except Exception as e:
        print(f"Error creating DOCX output: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

