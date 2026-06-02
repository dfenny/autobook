import re
from typing import List

_ABBREVS = {
    r"\bMr\.": "Mister",
    r"\bMrs\.": "Missus",
    r"\bMs\.": "Miss",
    r"\bDr\.": "Doctor",
    r"\bProf\.": "Professor",
    r"\bSt\.": "Saint",
    r"\bvs\.": "versus",
    r"\betc\.": "et cetera",
    r"\be\.g\.": "for example",
    r"\bi\.e\.": "that is",
    r"\bJr\.": "Junior",
    r"\bSr\.": "Senior",
    r"\bNo\.": "Number",
    r"\bVol\.": "Volume",
    r"\bCh\.": "Chapter",
    r"\bFig\.": "Figure",
    r"\bpp\.": "pages",
    r"\bp\.": "page",
}

_ROMAN = re.compile(
    r"\b(M{0,4})(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b"
)


def clean_text(text: str) -> str:
    text = _normalize_whitespace(text)
    text = _fix_unicode(text)
    text = _expand_abbreviations(text)
    text = _clean_punctuation(text)
    return text.strip()


def split_into_chunks(text: str, max_words: int = 400) -> List[str]:
    """Split text into sentence-boundary-respecting chunks."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current: List[str] = []
    current_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_count + word_count > max_words and current:
            chunks.append(" ".join(current))
            current = []
            current_count = 0
        current.append(sentence)
        current_count += word_count

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _fix_unicode(text: str) -> str:
    replacements = {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": ", ",  # en-dash → pause
        "—": ", ",  # em-dash → pause
        "…": "...",
        " ": " ",
        "­": "",    # soft hyphen
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    return text


def _expand_abbreviations(text: str) -> str:
    for pattern, replacement in _ABBREVS.items():
        text = re.sub(pattern, replacement, text)
    return text


def _clean_punctuation(text: str) -> str:
    # Remove stray asterisks, underscores used for markdown emphasis
    text = re.sub(r"[*_]{1,3}(.+?)[*_]{1,3}", r"\1", text)
    # Collapse multiple punctuation
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    return text
