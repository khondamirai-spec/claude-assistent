"""Cheap language hint (uz / ru / en) for the latest incoming message.

This is only a hint for Claude; Claude makes the final decision itself.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

# Apostrophe variants people use for o' / g' in Latin Uzbek.
APOSTROPHES = "'ʻʼ’‘`"

_UZ_CYRILLIC_LETTERS = set("ўқғҳЎҚҒҲ")

_UZ_CYRILLIC_WORDS = {
    "ассалому", "алайкум", "салом", "рахмат", "раҳмат", "яхши", "яхшими", "нима", "нега",
    "қалесан", "калесан", "қалайсиз", "қаерда", "каерда", "қаердасан", "йўқ", "йук",
    "бугун", "эртага", "керак", "хоп", "майли", "ҳозир", "хозир", "кейин", "мен", "сен",
    "сиз", "биз", "қачон", "качон", "борми", "билан", "учун", "эмас", "ака", "опа",
}

_UZ_LATIN_WORDS = {
    "assalomu", "alaykum", "salom", "rahmat", "raxmat", "yaxshi", "yaxshimi", "nima", "nega",
    "qalesan", "qalaysan", "qalaysiz", "qayerda", "qayerdasan", "qayerdasiz", "bugun",
    "ertaga", "kerak", "xop", "bopti", "zor", "hozir", "keyin", "men", "sen", "siz",
    "qachon", "yoq", "bormi", "bilan", "uchun", "emas", "opa", "uka", "juda",
}

_UZ_SUFFIXES = (
    "misan", "misiz", "mizmi", "yapsan", "yapman", "yapti", "yapmiz", "asizmi", "ganman", "dingmi",
)

_UZ_CYRILLIC_SUFFIXES = (
    "мисан", "мисиз", "мизми", "япсан", "япман", "япти", "япмиз", "асизми", "ганман", "дингми",
)

# o' / g' followed by a letter, but not the English "dog's" or "o'clock".
_UZ_APOSTROPHE_RE = re.compile(
    rf"[og][{APOSTROPHES}](?!s\b)(?!clock\b)(?=[a-z])",
    re.IGNORECASE,
)
_APOSTROPHE_STRIP_RE = re.compile(f"[{APOSTROPHES}]")
_LATIN_WORD_RE = re.compile(r"[a-z]+")
_CYRILLIC_WORD_RE = re.compile(r"[Ѐ-ӿ]+")


def _is_cyrillic(ch: str) -> bool:
    return "Ѐ" <= ch <= "ӿ"


def _is_latin(ch: str) -> bool:
    return ch.isalpha() and unicodedata.name(ch, "").startswith("LATIN")


def detect_language(text: str) -> Optional[str]:
    """Return "uz", "ru", "en", or None when the text has no letters."""
    if not text:
        return None
    cyrillic = sum(1 for ch in text if _is_cyrillic(ch))
    latin = sum(1 for ch in text if _is_latin(ch))
    if cyrillic == 0 and latin == 0:
        return None

    lowered = text.lower()
    if cyrillic > latin:
        if any(ch in _UZ_CYRILLIC_LETTERS for ch in text):
            return "uz"
        for word in _CYRILLIC_WORD_RE.findall(lowered):
            if word in _UZ_CYRILLIC_WORDS or _has_suffix(word, _UZ_CYRILLIC_SUFFIXES):
                return "uz"
        return "ru"

    if _UZ_APOSTROPHE_RE.search(text):
        return "uz"
    for word in _LATIN_WORD_RE.findall(_APOSTROPHE_STRIP_RE.sub("", lowered)):
        if word in _UZ_LATIN_WORDS or _has_suffix(word, _UZ_SUFFIXES):
            return "uz"
    return "en"


def _has_suffix(word: str, suffixes) -> bool:
    return any(len(word) > len(suffix) and word.endswith(suffix) for suffix in suffixes)


LANGUAGE_NAMES = {"uz": "Uzbek", "ru": "Russian", "en": "English"}
