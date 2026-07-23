"""Language detection for the output-drift metric.

A thin wrapper over lingua. Detection is restricted to an experiment's own
languages so the detector never guesses one it was never asked about, and each
built detector is cached by its language set. A minimum relative distance keeps
the detector honest on short, ambiguous answers: when the top two languages are
too close it returns no language rather than a confident-looking guess.
"""

from functools import cache

from lingua import IsoCode639_1, Language, LanguageDetector, LanguageDetectorBuilder

_MINIMUM_RELATIVE_DISTANCE = 0.25


def primary_subtag(lang: str) -> str:
    """Return the lowercase primary subtag of a BCP-47 code, 'pt-BR' -> 'pt'."""
    return lang.split("-")[0].lower()


def _to_language(lang: str) -> Language:
    """Map a BCP-47 code to a lingua Language, or raise if it is unsupported."""
    iso = getattr(IsoCode639_1, primary_subtag(lang).upper(), None)
    if iso is None:
        raise ValueError(f"Language not supported by the detector: {lang}")
    return Language.from_iso_code_639_1(iso)


@cache
def _detector(languages: tuple[str, ...]) -> LanguageDetector | None:
    """Build a detector over the languages, or None if fewer than two remain."""
    mapped = {_to_language(lang) for lang in languages}
    if len(mapped) < 2:
        return None
    return (
        LanguageDetectorBuilder.from_languages(*mapped)
        .with_minimum_relative_distance(_MINIMUM_RELATIVE_DISTANCE)
        .build()
    )


def detect_language(text: str, languages: tuple[str, ...]) -> str | None:
    """Detect a text's language among languages, as its BCP-47 primary subtag.

    Returns None when detection is unavailable (fewer than two languages) or the
    detector is not confident enough to choose one.
    """
    detector = _detector(languages)
    if detector is None:
        return None
    detected = detector.detect_language_of(text)
    if detected is None:
        return None
    return detected.iso_code_639_1.name.lower()
