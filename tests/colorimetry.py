"""Colour maths the palette tests measure with, rather than eyeball."""

import math

# Machado, Oliveira and Fernandes (2009) transforms at severity 1.0, linear RGB.
CVD_TRANSFORMS = {
    "protan": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deutan": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
}


def linear(color: str) -> tuple[float, float, float]:
    """One hex colour as linear-light RGB, which every measure below starts from."""
    digits = color.lstrip("#")
    channels = [int(digits[index : index + 2], 16) / 255 for index in (0, 2, 4)]
    expanded = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]

    return (expanded[0], expanded[1], expanded[2])


def oklab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """Linear RGB in OKLab, the space every distance here is measured in."""
    red, green, blue = rgb
    long = math.cbrt(0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue)
    medium = math.cbrt(0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue)
    short = math.cbrt(0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue)

    return (
        0.2104542553 * long + 0.7936177850 * medium - 0.0040720468 * short,
        1.9779984951 * long - 2.4285922050 * medium + 0.4505937099 * short,
        0.0259040371 * long + 0.7827717662 * medium - 0.8086757660 * short,
    )


def lightness_and_chroma(color: str) -> tuple[float, float]:
    """One colour's OKLCH lightness and chroma, the two the bands are set on."""
    lightness, green_red, blue_yellow = oklab(linear(color))

    return lightness, math.hypot(green_red, blue_yellow)


def simulate(color: str, deficiency: str) -> tuple[float, float, float]:
    """One colour as a protanope or deuteranope sees it, in linear RGB."""
    rgb = linear(color)
    rows = CVD_TRANSFORMS[deficiency]

    mixed = [
        sum(weight * channel for weight, channel in zip(row, rgb, strict=True))
        for row in rows
    ]

    return tuple(min(max(channel, 0.0), 1.0) for channel in mixed)  # type: ignore[return-value]


def delta_e(first: str, second: str, deficiency: str | None = None) -> float:
    """Euclidean OKLab distance times 100, simulated when a deficiency is named."""
    left = oklab(simulate(first, deficiency) if deficiency else linear(first))
    right = oklab(simulate(second, deficiency) if deficiency else linear(second))

    return 100 * math.dist(left, right)


def contrast(color: str, surface: str) -> float:
    """The WCAG contrast ratio between one colour and the page it is drawn on."""
    ratios = sorted(_relative_luminance(value) for value in (color, surface))

    return (ratios[1] + 0.05) / (ratios[0] + 0.05)


def _relative_luminance(color: str) -> float:
    """The WCAG relative luminance of one colour."""
    red, green, blue = linear(color)

    return 0.2126 * red + 0.7152 * green + 0.0722 * blue
