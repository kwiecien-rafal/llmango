"""What color every series in experiment 001 is drawn in, which is its language."""

SERIES_COLORS = {
    "en": "#0072B2",
    "pl": "#D55E00",
    "ja": "#009E73",
}


def language_color(lang: str) -> str:
    """Color one series by its language, refusing a language with no color declared."""
    return SERIES_COLORS[lang]
