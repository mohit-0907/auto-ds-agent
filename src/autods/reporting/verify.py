"""Check every number in the LLM's prose against the facts sheet.

The rule: a number in the prose must equal some number in the facts, up to
honest rounding. If the fact is 0.8815, then "0.882" and "0.88" pass but "0.885"
fails; if it is $28,285, then "$28.3k" passes. Small counts (0-10, as in "5
folds" or "3 attempts") are always allowed -- otherwise every sentence that
counts something would be flagged.
"""

from __future__ import annotations

import re

# Optional sign and $, an integer part (plain or with thousands commas), an
# optional decimal part, an optional k/M multiplier. The lookbehind skips digits
# glued to letters (F1, R2, gpt-6) so names are not read as numbers.
_NUMBER = re.compile(r"(?<![\w.])(-?)\$?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?:\s?([kKmM])\b)?(?![\w])")
_MULTIPLIER = {None: 1, "k": 1_000, "K": 1_000, "m": 1_000_000, "M": 1_000_000}
SMALL_COUNT_MAX = 10


def _numbers(text: str) -> list[tuple[str, float, int, int]]:
    """(as written, value in units shown, decimals shown, multiplier) for each number."""
    found = []
    for match in _NUMBER.finditer(text):
        sign, integer, fraction, suffix = match.groups()
        shown = float(sign + integer.replace(",", "") + (fraction or ""))
        decimals = len(fraction) - 1 if fraction else 0
        found.append((match.group(0), shown, decimals, _MULTIPLIER[suffix]))
    return found


def _fact_values(facts: str) -> list[float]:
    return [shown * multiplier for _, shown, _, multiplier in _numbers(facts)]


def _matches(shown: float, decimals: int, multiplier: int, fact: float) -> bool:
    """True if `shown` is `fact` rounded to the precision the prose used."""
    tolerance = 0.5 * 10**-decimals + 1e-9
    return abs(fact / multiplier - shown) <= tolerance


def unverified_numbers(prose: str, facts: str) -> list[str]:
    """Every number in the prose that no fact accounts for.

    Args:
        prose: The LLM-written narrative.
        facts: The facts sheet the LLM was given.

    Returns:
        The offending numbers as written, in order, without duplicates. Empty
        when every number checks out.
    """
    values = _fact_values(facts)
    bad: list[str] = []
    for written, shown, decimals, multiplier in _numbers(prose):
        if multiplier == 1 and decimals == 0 and 0 <= shown <= SMALL_COUNT_MAX:
            continue
        if any(_matches(shown, decimals, multiplier, fact) for fact in values):
            continue
        if written.strip() not in bad:
            bad.append(written.strip())
    return bad
