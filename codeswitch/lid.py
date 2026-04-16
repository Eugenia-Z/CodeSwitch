"""Language Identification (LID) system built on XLM-R + heuristics."""
from __future__ import annotations
import re
import unicodedata
from typing import List, Tuple

import torch
from transformers import pipeline


def is_language_neutral_content(text: str) -> bool:
    """Return True if text contains no alphabetic characters (numbers, punct, etc.)."""
    if not text or not text.strip():
        return True
    return not any(unicodedata.category(c).startswith("L") for c in text.strip())


class ProductionLID:
    """Multi-tier word-level LID: unicode → lexicon → n-gram → transformer."""

    def __init__(
        self,
        model_name: str = "papluca/xlm-roberta-base-language-detection",
        device: int | None = None,
        batch_size: int = 32,
    ) -> None:
        print("Loading Transformer-based LID model...")
        self.device = device if device is not None else (
            0 if torch.cuda.is_available() else -1
        )
        self.lid_model = pipeline(
            "text-classification",
            model=model_name,
            device=self.device,
            batch_size=batch_size,
        )

        self.lang_map: dict[str, str] = {
            "en": "English",  "zh": "Chinese",   "ja": "Japanese",
            "ar": "Arabic",   "hi": "Hindi",     "vi": "Vietnamese",
            "ru": "Russian",  "fr": "French",    "de": "German",
            "es": "Spanish",  "it": "Italian",   "ko": "Korean",
            "ms": "Malay",    "tl": "Filipino",
        }

        self.unicode_ranges: dict[str, list[tuple[int, int]]] = {
            "Japanese": [(0x3040, 0x309F), (0x30A0, 0x30FF)],
            "Korean":   [(0xAC00, 0xD7AF)],
            "Arabic":   [(0x0600, 0x06FF)],
            "Hindi":    [(0x0900, 0x097F)],
            "Russian":  [(0x0400, 0x04FF)],
            "Chinese":  [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
        }

        self.lexicons: dict[str, frozenset] = {
            "Vietnamese": frozenset([
                "có", "là", "không", "và", "của", "cho", "trong", "với", "được", "này",
                "các", "một", "những", "khi", "nhưng", "hay", "để", "từ", "đã", "cũng",
                "phải", "sẽ", "nếu", "vì", "hoặc", "đến", "ra", "về", "năm", "người",
            ]),
            "Cantonese": frozenset([
                "嘅", "咗", "唔", "係", "喺", "咁", "哋", "嚟", "噉", "嘢",
                "啲", "咩", "冇", "佢", "乜", "嗰", "吖", "嘞", "噃", "啩", "咯", "㗎",
            ]),
            "German": frozenset([
                "ich", "nicht", "ist", "das", "die", "der", "und", "ein", "zu", "mit",
                "sich", "auf", "auch", "dass", "aber", "noch", "wird", "beim", "vom",
                "haben", "sein", "war", "sind", "wurde", "wenn", "durch", "nach", "oder",
            ]),
            "French": frozenset([
                "je", "ne", "pas", "est", "les", "des", "une", "que", "qui", "dans",
                "avec", "sur", "pour", "par", "mais", "ont", "être", "très", "bien",
                "nous", "vous", "ils", "elle", "tout", "plus", "cette", "comme", "aux",
            ]),
            "Malay": frozenset([
                "dan", "yang", "ini", "itu", "di", "dengan", "untuk", "tidak", "ada",
                "dari", "dalam", "akan", "pada", "juga", "saya", "sudah", "bisa",
                "mereka", "seperti", "tetapi", "atau", "hanya", "oleh", "karena",
            ]),
            "Filipino": frozenset([
                "ang", "mga", "sa", "na", "ng", "ko", "ka", "niya", "siya", "ito",
                "ako", "mo", "namin", "nila", "pero", "kasi", "talaga", "lang",
                "din", "dito", "yung", "parang", "dapat", "pwede", "gusto",
            ]),
            "Korean": frozenset([
                "이", "그", "저", "것", "수", "등", "들", "및", "을", "를",
                "에", "의", "가", "으로", "하다", "있다", "되다", "없다",
            ]),
            "Spanish": frozenset([
                "el", "la", "los", "las", "un", "una", "que", "en", "es", "por",
                "con", "para", "como", "pero", "más", "este", "ya", "todo", "esta",
                "ser", "también", "fue", "había", "muy", "puede", "todos", "así",
                "nos", "cuando", "algo", "entre", "sin", "sobre", "tiene", "donde",
            ]),
        }

        self.ngram_features: dict[str, frozenset] = {
            "Vietnamese": frozenset(
                "àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
            ),
            "German":   frozenset("äöüßÄÖÜ"),
            "French":   frozenset("çœæÇŒÆ"),
            "Spanish":  frozenset("ñ¿¡Ñ"),
        }

        self._word_cache: dict[str, str] = {}
        print(f"✓ LID model loaded on {'GPU' if self.device >= 0 else 'CPU'}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _unicode_check(self, text: str) -> str | None:
        for ch in text:
            cp = ord(ch)
            for lang, ranges in self.unicode_ranges.items():
                for lo, hi in ranges:
                    if lo <= cp <= hi:
                        return lang
        return None

    def _ngram_check(self, text: str, lang1: str, lang2: str) -> str | None:
        for lang, char_set in self.ngram_features.items():
            if lang not in (lang1, lang2):
                continue
            if any(c in char_set for c in text):
                return lang
        return None

    def _transformer_detect_single(self, text: str, lang1: str, lang2: str) -> str:
        try:
            results = self.lid_model(text, top_k=3)
            for r in results:
                label    = r["label"]
                detected = self.lang_map.get(label)
                if label == "zh":
                    detected = "Cantonese" if "Cantonese" in (lang1, lang2) else "Chinese"
                if detected in (lang1, lang2):
                    return detected
        except Exception:
            pass
        return lang1

    def _batch_transformer_detect(
        self, texts: list[str], lang1: str, lang2: str
    ) -> list[str]:
        if not texts:
            return []
        unique     = list(dict.fromkeys(texts))
        result_map: dict[str, str] = {}
        try:
            batch_results = self.lid_model(unique, top_k=3, batch_size=32)
            for word, top3 in zip(unique, batch_results):
                resolved = lang1
                for r in top3:
                    label    = r["label"]
                    detected = self.lang_map.get(label)
                    if label == "zh":
                        detected = "Cantonese" if "Cantonese" in (lang1, lang2) else "Chinese"
                    if detected in (lang1, lang2):
                        resolved = detected
                        break
                result_map[word] = resolved
        except Exception:
            result_map = {w: lang1 for w in unique}
        return [result_map[t] for t in texts]

    def _cache_set(self, text: str, lang: str) -> None:
        if len(self._word_cache) >= 100_000:
            self._word_cache.clear()
        self._word_cache[text] = lang

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_language(self, text: str, lang1: str, lang2: str) -> str:
        """Detect language of a single string, constrained to {lang1, lang2, 'neutral'}."""
        clean = re.sub(r"[^\w\s]", "", text).strip()
        if not clean or is_language_neutral_content(clean):
            return "neutral"
        hit = self._unicode_check(clean)
        if hit and hit in (lang1, lang2):
            return hit
        word_lower = clean.lower().split()[0]
        for lang, lexicon in self.lexicons.items():
            if lang not in (lang1, lang2):
                continue
            if word_lower in lexicon or any(c in lexicon for c in clean):
                return lang
        hit = self._ngram_check(clean, lang1, lang2)
        if hit:
            return hit
        cached = self._word_cache.get(clean)
        if cached is not None:
            return cached
        result = self._transformer_detect_single(clean, lang1, lang2)
        self._cache_set(clean, result)
        return result

    def word_level_lid(
        self, text: str, lang1: str, lang2: str
    ) -> Tuple[List[str], List[str]]:
        """Return (words, per-word language tags) for the input text."""
        words                          = text.split()
        lids:        list[str | None]  = []
        l4_positions: list[int]        = []
        l4_texts:     list[str]        = []

        for i, word in enumerate(words):
            clean = re.sub(r"[^\w]", "", word)
            if not clean or is_language_neutral_content(clean):
                lids.append("neutral")
                continue
            hit = self._unicode_check(clean)
            if hit and hit in (lang1, lang2):
                lids.append(hit)
                continue
            hit_lang   = None
            word_lower = clean.lower()
            for lang, lexicon in self.lexicons.items():
                if lang not in (lang1, lang2):
                    continue
                if word_lower in lexicon or any(c in lexicon for c in clean):
                    hit_lang = lang
                    break
            if hit_lang:
                lids.append(hit_lang)
                continue
            hit = self._ngram_check(clean, lang1, lang2)
            if hit:
                lids.append(hit)
                continue
            cached = self._word_cache.get(clean)
            if cached is not None:
                lids.append(cached)
                continue
            # defer to transformer batch
            lids.append(None)
            l4_positions.append(i)
            l4_texts.append(clean)

        if l4_texts:
            detected = self._batch_transformer_detect(l4_texts, lang1, lang2)
            for pos, lang, clean in zip(l4_positions, detected, l4_texts):
                lids[pos] = lang
                self._cache_set(clean, lang)

        lids = [l if l is not None else lang1 for l in lids]
        return words, lids  # type: ignore[return-value]
