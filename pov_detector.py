"""
Multi-POV character detection and consistent voice assignment.

When --pov is passed, each chapter is analysed to identify the point-of-view
character and their gender. Voices are assigned from gendered pools and
persisted in voices.json so characters stay consistent across runs and
interrupted sessions.

Detection order:
  1. Claude API  (claude-haiku, title + first 300 words) — accurate for any name
  2. Heuristic   (title parsing + gender-guesser)         — fallback, no API needed

A character manifest (YAML or JSON) can be provided via --characters to
pre-define names, genders, and optional voice overrides. This eliminates
gender guessing and constrains detection to the known cast.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Voice pools
# ---------------------------------------------------------------------------

_KOKORO_FEMALE   = ["af_bella", "af_nicole", "af_sarah", "af_sky", "bf_emma", "bf_isabella"]
_KOKORO_MALE     = ["am_adam", "am_michael", "bm_george", "bm_lewis"]
_KOKORO_NARRATOR = "af_heart"

_EDGE_FEMALE     = ["en-US-JennyNeural", "en-US-AriaNeural", "en-GB-SoniaNeural", "en-AU-NatashaNeural"]
_EDGE_MALE       = ["en-US-GuyNeural", "en-US-DavisNeural", "en-GB-RyanNeural", "en-AU-WilliamNeural"]
_EDGE_NARRATOR   = "en-US-JennyNeural"

_VOICE_MAP_FILE  = "voices.json"

# Capitalised words that appear in chapter titles but are never character names
_TITLE_STOPWORDS = frozenset({
    "chapter", "part", "book", "interlude", "prologue", "epilogue", "postlude",
    "section", "the", "a", "an", "and", "of", "in", "to", "from",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "preface", "introduction", "foreword", "afterword", "appendix",
    "note", "notes", "acknowledgments", "acknowledgements", "contents",
    "illustrations", "endnote", "copyright", "fiction", "years", "ago",
    "thousand", "hundred", "million", "half",
})


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> list[dict[str, Any]]:
    """
    Load a character manifest from a YAML or JSON file.

    Accepted formats:

      # characters.yaml
      characters:
        - name: Kaladin
          gender: male
          voice: am_adam   # optional
        - name: Shallan
          gender: female

      # or a bare list (no wrapper key)
      - name: Kaladin
        gender: male

    JSON is also accepted; the same structures apply.
    An optional top-level "narrator" key sets the narrator voice:
      narrator: af_heart
    """
    raw = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml
        data = yaml.safe_load(raw)
    else:
        data = json.loads(raw)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("characters", [])
    raise ValueError(f"Manifest must be a list or a dict with a 'characters' key, got {type(data)}")


def _narrator_from_manifest(path: Path) -> str | None:
    """Return the optional top-level 'narrator' voice from a manifest file."""
    raw = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml
        data = yaml.safe_load(raw)
    else:
        data = json.loads(raw)
    if isinstance(data, dict):
        return data.get("narrator") or None
    return None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CharacterInfo:
    name: str
    gender: str   # "male" | "female" | "unknown"
    voice: str


class VoiceMap:
    """
    Maintains the character → voice assignment for one book.

    Persisted as voices.json in the output directory so sessions can be
    interrupted and resumed without re-detecting or re-assigning.
    """

    def __init__(self, engine: str = "kokoro", narrator_voice: str | None = None):
        self.engine          = engine
        self.female_pool     = _KOKORO_FEMALE   if engine == "kokoro" else _EDGE_FEMALE
        self.male_pool       = _KOKORO_MALE     if engine == "kokoro" else _EDGE_MALE
        self._narrator_voice = narrator_voice or (
            _KOKORO_NARRATOR if engine == "kokoro" else _EDGE_NARRATOR
        )
        self.characters: dict[str, CharacterInfo] = {}
        self._female_idx = 0
        self._male_idx   = 0

    @property
    def narrator_voice(self) -> str:
        return self._narrator_voice

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: Path) -> None:
        data = {
            "engine":         self.engine,
            "narrator_voice": self._narrator_voice,
            "_female_idx":    self._female_idx,
            "_male_idx":      self._male_idx,
            "characters": {
                name: {"gender": c.gender, "voice": c.voice}
                for name, c in self.characters.items()
            },
        }
        (output_dir / _VOICE_MAP_FILE).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, output_dir: Path, engine: str = "kokoro", narrator_voice: str | None = None) -> "VoiceMap":
        path = output_dir / _VOICE_MAP_FILE
        if not path.exists():
            return cls(engine=engine, narrator_voice=narrator_voice)
        try:
            data = json.loads(path.read_text())
            if data.get("engine") != engine:
                print(
                    f"  Warning: voices.json was created with engine={data.get('engine')!r} "
                    f"but current engine is {engine!r}. Starting fresh."
                )
                return cls(engine=engine, narrator_voice=narrator_voice)
            vm = cls(engine=engine, narrator_voice=data.get("narrator_voice") or narrator_voice)
            vm._female_idx = data.get("_female_idx", 0)
            vm._male_idx   = data.get("_male_idx", 0)
            for name, cd in data.get("characters", {}).items():
                vm.characters[name] = CharacterInfo(
                    name=name, gender=cd["gender"], voice=cd["voice"]
                )
            return vm
        except (json.JSONDecodeError, KeyError):
            return cls(engine=engine, narrator_voice=narrator_voice)

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def populate_from_manifest(self, manifest: list[dict[str, Any]]) -> None:
        """
        Pre-populate characters from a manifest. Manifest always wins over
        whatever was previously loaded from voices.json:
          - Explicit voice  → stored as-is, pool indices not touched.
          - Gender only     → existing voice kept if already assigned;
                              otherwise auto-assigned from the gendered pool.
        Characters discovered dynamically during narration (not in the
        manifest) are still assigned from the remaining pool slots.
        """
        for entry in manifest:
            name = entry.get("name", "").strip().title()
            if not name:
                continue
            gender = entry.get("gender", "unknown").lower()
            if gender not in ("male", "female", "unknown"):
                gender = "unknown"
            explicit_voice: str | None = entry.get("voice")

            if explicit_voice:
                # Explicit voice always overrides whatever is in voices.json.
                self.characters[name] = CharacterInfo(
                    name=name, gender=gender, voice=explicit_voice
                )
            elif name in self.characters:
                # Character already has a voice from a previous run; just
                # refresh gender in case the manifest corrects an "unknown".
                if self.characters[name].gender == "unknown" and gender != "unknown":
                    self.characters[name].gender = gender
            else:
                self._assign_voice(name, gender)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(
        self,
        chapter_title: str,
        chapter_text: str,
        api_key: str | None = None,
    ) -> tuple[str | None, str, str]:
        """
        Detect the POV character for this chapter and return their voice.

        Returns (character_name, gender, voice).
        character_name is None for omniscient / narrator chapters.
        """
        known_names = list(self.characters.keys()) or None

        if api_key:
            name, gender = _detect_with_claude(chapter_title, chapter_text, api_key, known_names)
        else:
            name, gender = _detect_with_heuristic(chapter_title, chapter_text, known_names)

        if not name:
            return None, "unknown", self._narrator_voice

        # Normalise to Title Case for consistent map keys
        name = name.strip().title()

        if name not in self.characters:
            voice = self._assign_voice(name, gender)
        else:
            existing = self.characters[name]
            # Upgrade "unknown" gender if we now know better
            if existing.gender == "unknown" and gender != "unknown":
                existing.gender = gender
            voice = existing.voice

        return name, self.characters[name].gender, voice

    def _assign_voice(self, name: str, gender: str) -> str:
        if gender == "female":
            if self._female_idx >= len(self.female_pool):
                print("  Warning: more female POV characters than available voices; cycling.")
            voice = self.female_pool[self._female_idx % len(self.female_pool)]
            self._female_idx += 1
        elif gender == "male":
            if self._male_idx >= len(self.male_pool):
                print("  Warning: more male POV characters than available voices; cycling.")
            voice = self.male_pool[self._male_idx % len(self.male_pool)]
            self._male_idx += 1
        else:
            voice = self._narrator_voice

        self.characters[name] = CharacterInfo(name=name, gender=gender, voice=voice)
        return voice

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        if not self.characters:
            return
        print("\n  POV voice assignments:")
        print(f"    {'Narrator':<22s} {'—':<10s} → {self._narrator_voice}")
        for name, c in self.characters.items():
            print(f"    {name:<22s} {c.gender:<10s} → {c.voice}")


# ---------------------------------------------------------------------------
# Detection backends
# ---------------------------------------------------------------------------

def _detect_with_claude(
    title: str,
    text: str,
    api_key: str,
    known_names: list[str] | None = None,
) -> tuple[str | None, str]:
    """Use Claude Haiku to identify POV character and gender."""
    try:
        import anthropic
    except ImportError:
        print("  Warning: anthropic package not installed. Falling back to heuristic detection.")
        return _detect_with_heuristic(title, text, known_names)

    opening = " ".join(text.split()[:300])
    client  = anthropic.Anthropic(api_key=api_key)

    if known_names:
        cast_line = (
            "The known POV characters in this book are: "
            + ", ".join(known_names)
            + ". Identify which of these is the POV character, or return null "
              "if none of them are (e.g. an omniscient narrator chapter). "
              "If the character is in the known list, use their name exactly as given."
        )
        gender_instruction = (
            'Omit "gender" or set it to "unknown" for known characters '
            "(their gender is already recorded)."
        )
    else:
        cast_line = ""
        gender_instruction = 'Set "gender" to "male", "female", or "unknown".'

    system_text = (
        "You identify the point-of-view character in fiction chapters. "
        f"{cast_line} "
        f'Respond ONLY with a JSON object: {{"character": "Name or null", "gender": "male|female|unknown"}}. '
        f"{gender_instruction} "
        "Use null for omniscient or narrator chapters with no single POV character. "
        "Never include explanation, only the JSON."
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=80,
            system=[{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": f'Chapter title: "{title}"\n\nOpening:\n{opening}'}],
        )
        raw    = response.content[0].text.strip()
        result = json.loads(raw)
        name   = result.get("character") or None
        gender = result.get("gender", "unknown")
        if gender not in ("male", "female", "unknown"):
            gender = "unknown"
        return name, gender
    except Exception as exc:
        print(f"  Warning: Claude API error ({exc}). Falling back to heuristic.")
        return _detect_with_heuristic(title, text, known_names)


def _detect_with_heuristic(
    title: str,
    text: str,
    known_names: list[str] | None = None,
) -> tuple[str | None, str]:
    """
    When known_names are provided, check whether any character's name tokens
    appear in the chapter title — this is fast and highly accurate for books
    that label chapters by character name.

    Falls back to open-ended title parsing when no known names are given.
    Gender is only guessed for newly-discovered characters (not in known_names).
    """
    if known_names:
        match = _match_known_name(title, known_names)
        if match:
            return match, "unknown"  # gender already stored from manifest

    name = _name_from_title(title)
    if not name:
        return None, "unknown"
    gender = _guess_gender(name.split()[0])
    return name, gender


def _match_known_name(title: str, known_names: list[str]) -> str | None:
    """
    Return the canonical character name if any of their name tokens appear
    in the chapter title (case-insensitive word match).

    "Kaladin Stormblessed" matches a title containing "Kaladin".
    """
    title_tokens = set(re.findall(r"[a-zA-Z''-]+", title.lower()))
    for name in known_names:
        name_tokens = set(re.findall(r"[a-zA-Z''-]+", name.lower()))
        if name_tokens & title_tokens:
            return name
    return None


def _name_from_title(title: str) -> str | None:
    """
    Strip the chapter number prefix then return the first capitalised
    non-stopword token, if any.

    Examples:
      "Chapter 5 — Kaladin"      → "Kaladin"
      "Shallan"                  → "Shallan"
      "Interlude 3"              → None
      "Twenty-Six Years Ago"     → None  (all stopwords)
    """
    # Remove leading "Chapter/Part/Book N [— | : | .]" pattern
    cleaned = re.sub(
        r"^(chapter|part|book|interlude|section)\s+[\w.]+\s*[-–—:]*\s*",
        "",
        title,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^\d[\d.,]*\s*[-–—:.]*\s*", "", cleaned).strip()

    for word in cleaned.split():
        token = re.sub(r"[^\w'-]", "", word)
        if (
            token
            and token[0].isupper()
            and token.lower() not in _TITLE_STOPWORDS
            and not token.isdigit()
        ):
            return token
    return None


def _guess_gender(first_name: str) -> str:
    try:
        import gender_guesser.detector as gd
        result = gd.Detector(case_sensitive=False).get_gender(first_name)
        if result in ("male", "mostly_male"):
            return "male"
        if result in ("female", "mostly_female"):
            return "female"
    except ImportError:
        pass
    return "unknown"
