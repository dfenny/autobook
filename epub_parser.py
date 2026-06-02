import re
from dataclasses import dataclass
from typing import List

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

# Matches xmlns declarations and namespace prefixes in tag/attribute names so
# the XML parser sees clean, unprefixed element names (e.g. <ns0:p> → <p>).
_RE_XMLNS = re.compile(rb'\s+xmlns(?::\w+)?="[^"]*"')
_RE_NS_PREFIX = re.compile(rb"<(/?)[\w.-]+:([\w.-]+)")


def _strip_namespaces(content: bytes) -> bytes:
    content = _RE_XMLNS.sub(b"", content)
    content = _RE_NS_PREFIX.sub(rb"<\1\2", content)
    return content


@dataclass
class Chapter:
    index: int
    title: str
    text: str


def parse_epub(epub_path: str) -> List[Chapter]:
    book = epub.read_epub(epub_path, options={"ignore_ncx": False})
    toc_map = _build_toc_map(book)
    chapters = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(_strip_namespaces(item.get_content()), "lxml-xml")
        text = _extract_text(soup)

        if len(text.strip()) < 150:
            continue

        file_name = item.get_name().split("/")[-1]
        title = toc_map.get(file_name) or toc_map.get(item.get_name())
        if not title:
            title = _heading_title(soup) or f"Chapter {len(chapters) + 1}"

        chapters.append(Chapter(index=len(chapters), title=title, text=text))

    return chapters


def _build_toc_map(book: epub.EpubBook) -> dict:
    mapping: dict = {}

    def walk(items):
        for item in items:
            if isinstance(item, epub.Link):
                href = item.href.split("#")[0].split("/")[-1]
                if href and item.title:
                    mapping[href] = item.title
            elif isinstance(item, tuple):
                section, children = item
                if isinstance(section, epub.Section) and section.href:
                    href = section.href.split("#")[0].split("/")[-1]
                    if href and section.title:
                        mapping[href] = section.title
                walk(children)

    walk(book.toc)
    return mapping


def _extract_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "aside", "nav", "figure", "figcaption"]):
        tag.decompose()

    lines = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote"]):
        text = el.get_text(" ", strip=True)
        if text:
            lines.append(text)

    return "\n\n".join(lines)


def _heading_title(soup: BeautifulSoup) -> str | None:
    for tag in ["h1", "h2", "h3"]:
        el = soup.find(tag)
        if el:
            title = el.get_text(" ", strip=True)
            if title and len(title) < 120:
                return title
    return None
