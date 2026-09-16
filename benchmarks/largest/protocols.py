"""LargeST temporal holdout protocol identifiers."""

from __future__ import annotations

PROTOCOLS: tuple[str, ...] = ("eval_2019", "eval_2020", "eval_2021")
DEFAULT_PROTOCOL: str = "eval_2019"
DEFAULT_PROTOCOLS_CSV: str = ",".join(PROTOCOLS)
