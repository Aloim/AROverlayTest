"""
Text processor for DesktopWebcamHandtracker voice-to-text.

Provides autocorrect and basic grammar fixes for transcribed text
using MIT-licensed libraries only.
"""

import re
from pathlib import Path
from typing import Optional

try:
    from spellchecker import SpellChecker
except ImportError as e:
    raise ImportError(
        "pyspellchecker is required. Install with: pip install pyspellchecker"
    ) from e

try:
    from symspellpy import SymSpell, Verbosity
    import symspellpy
    HAS_SYMSPELL = True
except ImportError as e:
    HAS_SYMSPELL = False
    SymSpell = None
    Verbosity = None
    symspellpy = None

from logger import get_logger

logger = get_logger("TextProcessor")


class TextProcessor:
    """
    Text processing for voice recognition output.

    Provides spelling correction and basic grammar fixes to improve
    transcription quality.
    """

    # Punctuation word substitutions (spoken word -> symbol)
    # Order matters: longer phrases must come before shorter ones
    PUNCTUATION_SUBSTITUTIONS = [
        # Multi-word phrases first
        (r'\bquestion\s+mark\b', '?'),
        (r'\bexclamation\s+point\b', '!'),
        (r'\bexclamation\s+mark\b', '!'),
        # Single words
        (r'\bdot\b', '.'),
        (r'\bperiod\b', '.'),
        (r'\bslash\b', '/'),
        (r'\bforward\s+slash\b', '/'),
        (r'\bbackslash\b', '\\\\'),
        (r'\bback\s+slash\b', '\\\\'),
        (r'\bat\b', '@'),
        (r'\bat\s+sign\b', '@'),
        (r'\bdash\b', '-'),
        (r'\bhyphen\b', '-'),
        (r'\bunderscore\b', '_'),
        (r'\bcomma\b', ','),
        (r'\bcolon\b', ':'),
        (r'\bsemicolon\b', ';'),
        (r'\bsemi\s+colon\b', ';'),
        # Additional useful ones
        (r'\bhashtag\b', '#'),
        (r'\bhash\b', '#'),
        (r'\bampersand\b', '&'),
        (r'\band\s+sign\b', '&'),
        (r'\basterisk\b', '*'),
        (r'\bstar\b', '*'),
        (r'\bplus\b', '+'),
        (r'\bplus\s+sign\b', '+'),
        (r'\bequals\b', '='),
        (r'\bequal\s+sign\b', '='),
        (r'\bpercent\b', '%'),
        (r'\bpercent\s+sign\b', '%'),
        (r'\bdollar\b', '$'),
        (r'\bdollar\s+sign\b', '$'),
        (r'\bopen\s+paren\b', '('),
        (r'\bclose\s+paren\b', ')'),
        (r'\bopen\s+parenthesis\b', '('),
        (r'\bclose\s+parenthesis\b', ')'),
        (r'\bopen\s+bracket\b', '['),
        (r'\bclose\s+bracket\b', ']'),
        (r'\bopen\s+brace\b', '{'),
        (r'\bclose\s+brace\b', '}'),
        (r'\bnew\s+line\b', '\n'),
        (r'\btab\b', '\t'),
    ]

    def __init__(
        self,
        enable_autocorrect: bool = True,
        enable_grammar_fix: bool = True,
        enable_punctuation_substitution: bool = True
    ):
        """
        Initialize text processor.

        Args:
            enable_autocorrect: Enable spelling correction.
            enable_grammar_fix: Enable basic grammar fixes.
            enable_punctuation_substitution: Enable spoken punctuation word substitution
                                            (e.g., "dot" -> ".", "at" -> "@").
        """
        self.enable_autocorrect = enable_autocorrect
        self.enable_grammar_fix = enable_grammar_fix
        self.enable_punctuation_substitution = enable_punctuation_substitution

        # Initialize spellcheckers
        if enable_autocorrect:
            # PySpellChecker for context-aware corrections
            self._spell_checker = SpellChecker()

            # SymSpell for fast fuzzy matching (optional - gracefully degrade if unavailable)
            self._sym_spell = None
            if HAS_SYMSPELL and symspellpy is not None:
                try:
                    self._sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
                    # Load frequency dictionary from symspellpy package directory
                    symspell_dir = Path(symspellpy.__file__).parent
                    dictionary_path = symspell_dir / "frequency_dictionary_en_82_765.txt"
                    if dictionary_path.exists():
                        self._sym_spell.load_dictionary(
                            str(dictionary_path),
                            term_index=0,
                            count_index=1
                        )
                        logger.debug(f"Loaded {self._sym_spell.word_count} words into SymSpell")
                    else:
                        logger.warning(f"SymSpell dictionary not found at {dictionary_path}")
                        self._sym_spell = None
                except Exception as e:
                    logger.warning(f"Failed to initialize SymSpell: {e}")
                    self._sym_spell = None
        else:
            self._spell_checker = None
            self._sym_spell = None

        logger.info(
            f"TextProcessor initialized (autocorrect={enable_autocorrect}, "
            f"grammar={enable_grammar_fix}, punctuation={enable_punctuation_substitution})"
        )

    def process(self, text: str) -> str:
        """
        Process text with enabled corrections.

        Args:
            text: Raw transcribed text.

        Returns:
            Processed text with corrections applied.
        """
        if not text:
            return text

        # Apply punctuation substitution FIRST (before autocorrect)
        # This prevents "dot" from being "corrected" to something else
        if self.enable_punctuation_substitution:
            text = self.substitute_punctuation(text)

        # Apply autocorrect if enabled
        if self.enable_autocorrect:
            text = self.autocorrect(text)

        # Apply grammar fixes if enabled
        if self.enable_grammar_fix:
            text = self.fix_grammar(text)

        return text

    def substitute_punctuation(self, text: str) -> str:
        """
        Substitute spoken punctuation words with actual symbols.

        Converts words like "dot", "slash", "at", "comma" into their
        corresponding ASCII symbols (., /, @, ,).

        Args:
            text: Text with spoken punctuation words.

        Returns:
            Text with punctuation symbols substituted.
        """
        if not text:
            return text

        original = text
        for pattern, symbol in self.PUNCTUATION_SUBSTITUTIONS:
            text = re.sub(pattern, symbol, text, flags=re.IGNORECASE)

        # Clean up spaces around symbols that typically appear without spaces
        # For URL/email symbols (@, /, ., _), remove spaces on BOTH sides
        text = re.sub(r'\s*(@|/|_)\s*', r'\1', text)

        # For dot, only collapse spaces if it looks like a URL/email (surrounded by alphanumeric)
        # e.g., "gmail . com" -> "gmail.com" but "Hello . World" stays "Hello. World"
        text = re.sub(r'(\w)\s*\.\s*(\w)', r'\1.\2', text)

        # For punctuation at end of words, remove space before
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)

        # For dash/hyphen between words, remove spaces on both sides
        text = re.sub(r'\s*-\s*', '-', text)

        if text != original:
            logger.debug(f"Punctuation substitution: '{original}' -> '{text}'")

        return text

    def autocorrect(self, text: str) -> str:
        """
        Apply spelling corrections to text.

        Uses both PySpellChecker and SymSpell for robust correction.
        Preserves proper nouns and technical terms.

        Args:
            text: Text to correct.

        Returns:
            Corrected text.
        """
        if not self._spell_checker or not self._sym_spell:
            return text

        # Split into words while preserving whitespace and punctuation
        words = re.findall(r'\b\w+\b|\W+', text)
        corrected_words = []

        for token in words:
            # Skip non-word tokens (whitespace, punctuation)
            if not re.match(r'\w+', token):
                corrected_words.append(token)
                continue

            # Skip if word is capitalized (likely proper noun)
            if token[0].isupper() and len(token) > 1:
                corrected_words.append(token)
                continue

            # Skip short words (often correct)
            if len(token) <= 2:
                corrected_words.append(token)
                continue

            # Check if word is correct
            if self._spell_checker.known([token.lower()]):
                corrected_words.append(token)
                continue

            # Try SymSpell first (faster for obvious typos)
            suggestions = self._sym_spell.lookup(
                token.lower(),
                Verbosity.CLOSEST,
                max_edit_distance=2
            )

            if suggestions and suggestions[0].distance <= 2:
                corrected = suggestions[0].term
                # Preserve original casing
                if token[0].isupper():
                    corrected = corrected.capitalize()
                logger.debug(f"Autocorrect: '{token}' -> '{corrected}'")
                corrected_words.append(corrected)
            else:
                # Fallback to PySpellChecker
                correction = self._spell_checker.correction(token.lower())
                if correction and correction != token.lower():
                    # Preserve original casing
                    if token[0].isupper():
                        correction = correction.capitalize()
                    logger.debug(f"Autocorrect: '{token}' -> '{correction}'")
                    corrected_words.append(correction)
                else:
                    # Keep original if no good correction found
                    corrected_words.append(token)

        return ''.join(corrected_words)

    def fix_grammar(self, text: str) -> str:
        """
        Apply basic grammar fixes to text.

        Handles:
        - Capitalization of first word
        - Sentence ending punctuation
        - Common grammar patterns
        - Contraction fixes

        Note: This is NOT a full grammar checker (LanguageTool is GPL).
        Only handles common voice transcription issues.

        Args:
            text: Text to fix.

        Returns:
            Text with grammar fixes applied.
        """
        if not text:
            return text

        # Trim whitespace
        text = text.strip()

        # Capitalize first letter
        if text and text[0].islower():
            text = text[0].upper() + text[1:]

        # Fix common contractions that ASR might miss
        contractions = {
            r'\bim\b': "I'm",
            r'\bive\b': "I've",
            r'\bill\b': "I'll",
            r'\bwont\b': "won't",
            r'\bcant\b': "can't",
            r'\bdont\b': "don't",
            r'\bdidnt\b': "didn't",
            r'\bisnt\b': "isn't",
            r'\barent\b': "aren't",
            r'\bwasnt\b': "wasn't",
            r'\bwerent\b': "weren't",
            r'\bhavent\b': "haven't",
            r'\bhasnt\b': "hasn't",
            r'\bhadnt\b': "hadn't",
            r'\bwouldnt\b': "wouldn't",
            r'\bcouldnt\b': "couldn't",
            r'\bshouldnt\b': "shouldn't",
        }

        for pattern, replacement in contractions.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Fix "i" -> "I" (pronoun)
        text = re.sub(r'\bi\b', 'I', text)

        # Add period at end if missing punctuation
        if text and text[-1] not in '.!?':
            text += '.'

        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,!?;:])(\w)', r'\1 \2', text)  # Add space after punctuation

        # Fix multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Capitalize after sentence endings
        def capitalize_after_period(match):
            return match.group(1) + match.group(2).upper()

        text = re.sub(r'([.!?]\s+)([a-z])', capitalize_after_period, text)

        return text

    def set_autocorrect(self, enabled: bool) -> None:
        """
        Enable or disable autocorrect.

        Args:
            enabled: Whether to enable autocorrect.
        """
        self.enable_autocorrect = enabled
        logger.debug(f"Autocorrect {'enabled' if enabled else 'disabled'}")

    def set_grammar_fix(self, enabled: bool) -> None:
        """
        Enable or disable grammar fixes.

        Args:
            enabled: Whether to enable grammar fixes.
        """
        self.enable_grammar_fix = enabled
        logger.debug(f"Grammar fix {'enabled' if enabled else 'disabled'}")

    def set_punctuation_substitution(self, enabled: bool) -> None:
        """
        Enable or disable punctuation word substitution.

        Args:
            enabled: Whether to enable punctuation substitution.
        """
        self.enable_punctuation_substitution = enabled
        logger.debug(f"Punctuation substitution {'enabled' if enabled else 'disabled'}")


def test_text_processor():
    """Simple test function for text processor."""
    processor = TextProcessor()

    print("Text Processor Test:")
    print("=" * 50)

    # Test punctuation substitution
    print("\n--- Punctuation Substitution ---")
    punctuation_tests = [
        "my email is john at gmail dot com",  # -> john@gmail.com
        "visit example dot com slash home",  # -> example.com/home
        "what is this question mark",  # -> what is this?
        "wow exclamation point",  # -> wow!
        "use dash to separate",  # -> use - to separate
        "my underscore var",  # -> my_var
        "first comma second comma third",  # -> first, second, third
        "hello dot dot dot world",  # -> hello... world
    ]

    for original in punctuation_tests:
        result = processor.substitute_punctuation(original)
        print(f"  '{original}'")
        print(f"  -> '{result}'")
        print()

    # Test spelling corrections
    print("\n--- Spelling Corrections ---")
    spelling_tests = [
        "this is a tst",  # Spelling error
        "im going to the stor",  # Contraction + spelling
        "hello wrld",  # Spelling error
        "i cant beleive it",  # Pronoun + contraction + spelling
    ]

    for original in spelling_tests:
        corrected = processor.process(original)
        print(f"  '{original}'")
        print(f"  -> '{corrected}'")
        print()


if __name__ == "__main__":
    test_text_processor()
