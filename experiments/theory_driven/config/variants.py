"""Stable stage arm identities; policy suffixes never change the base arm."""
import re

POLICIES = {"strict", "majority", "veto", "half_no_regress"}


def parse_arm_variant(value: str) -> tuple[str, str | None]:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*(?:@[a-z_]+)?", value):
        raise ValueError(f"Invalid arm variant: {value}")
    base, separator, policy = value.partition("@")
    if separator and (base not in {"B", "M"} or policy not in POLICIES):
        raise ValueError(f"Invalid policy variant: {value}")
    return base, policy if separator else None
