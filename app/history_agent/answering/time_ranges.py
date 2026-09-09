"""Deterministic parsing for open-ended year constraints."""

from __future__ import annotations

import re
from dataclasses import dataclass

RELATIVE_YEAR_PATTERN = re.compile(
    r"(?P<year>\d{4})\s*年?\s*(?P<inclusive>及)?\s*"
    r"(?P<relation>之前|以前|之后|以后|前|后)"
)
YEAR_TOKEN = re.compile(r"(?<!\d)\d{4}(?!\d)")


@dataclass(frozen=True)
class RelativeYearRange:
    raw: str
    cutoff: int
    start: int
    end: int
    inclusive: bool


def parse_relative_year_range(
    question: str, lower: int, upper: int
) -> RelativeYearRange | None:
    """Resolve one unambiguous “before/after YEAR” expression to corpus bounds."""

    matches = list(RELATIVE_YEAR_PATTERN.finditer(question))
    if len(matches) != 1 or YEAR_TOKEN.search(RELATIVE_YEAR_PATTERN.sub("", question)):
        return None
    match = matches[0]
    cutoff = int(match["year"])
    inclusive = match["inclusive"] is not None
    if match["relation"] in {"之前", "以前", "前"}:
        start, end = lower, min(upper, cutoff if inclusive else cutoff - 1)
    else:
        start, end = max(lower, cutoff if inclusive else cutoff + 1), upper
    return RelativeYearRange(match.group(), cutoff, start, end, inclusive)
