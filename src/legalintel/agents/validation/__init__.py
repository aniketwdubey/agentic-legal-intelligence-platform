"""Validation package: rule-based citation verification."""

from legalintel.agents.validation.schema import ClaimVerdict, ValidationReport
from legalintel.agents.validation.validator import validate

__all__ = ["validate", "ValidationReport", "ClaimVerdict"]
