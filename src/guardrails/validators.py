"""
Guardrails: Input/Output Validation and Clinical Safety
Implements safety controls for a healthcare AI system.

Module 11: Guardrails
- Input validation: query sanitization, PHI detection, length limits
- Output validation: confidence checks, citation requirements, clinical disclaimers
- Prompt injection defense
"""
import re
import logging
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field, field_validator

from config.settings import MAX_QUERY_LENGTH, MIN_CONFIDENCE_THRESHOLD, BLOCKED_PATTERNS

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCKED = "blocked"


@dataclass
class ValidationResult:
    """Result of a guardrail check."""
    passed: bool
    risk_level: RiskLevel = RiskLevel.LOW
    issues: list[str] = field(default_factory=list)
    sanitized_input: Optional[str] = None
    blocked_reason: Optional[str] = None

    def to_dict(self):
        return {
            "passed": self.passed,
            "risk_level": self.risk_level.value,
            "issues": self.issues,
            "blocked_reason": self.blocked_reason,
        }


# === Pydantic Schema-Based Guardrails (Module 11: Schema-Based Guardrails) ===

class QueryRequest(BaseModel):
    """Validated query input schema."""
    query: str = Field(..., min_length=3, max_length=MAX_QUERY_LENGTH)
    drug_name: Optional[str] = Field(None, max_length=200)
    section_type: Optional[str] = Field(None, max_length=100)
    therapeutic_area: Optional[str] = Field(None, max_length=200)
    enable_rewrite: bool = True

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        # Strip excessive whitespace
        v = re.sub(r"\s+", " ", v.strip())
        if len(v) < 3:
            raise ValueError("Query must be at least 3 characters")
        return v


class ComparisonRequest(BaseModel):
    """Validated comparison request schema."""
    query: str = Field(..., min_length=3, max_length=MAX_QUERY_LENGTH)
    drug_names: list[str] = Field(..., min_length=2, max_length=10)

    @field_validator("drug_names")
    @classmethod
    def validate_drug_names(cls, v):
        if len(v) < 2:
            raise ValueError("Need at least 2 drugs to compare")
        return [name.strip() for name in v]


class RAGResponseSchema(BaseModel):
    """Schema for validated RAG output."""
    answer: str
    citations: list[dict]
    confidence: float = Field(ge=0.0, le=1.0)
    disclaimer: str = ""


# === Input Guardrails ===

class InputGuardrails:
    """
    Pre-processing guardrails for incoming queries.
    
    Checks:
    1. PHI detection (SSN, DOB patterns)
    2. Prompt injection defense
    3. Content filtering
    4. Length and format validation
    """

    # Prompt injection patterns
    INJECTION_PATTERNS = [
        r"(?i)ignore\s+(all\s+)?previous\s+instructions",
        r"(?i)disregard\s+(all\s+)?above",
        r"(?i)you\s+are\s+now\s+",
        r"(?i)forget\s+(everything|all)",
        r"(?i)system\s*:\s*",
        r"(?i)jailbreak",
        r"(?i)pretend\s+you",
        r"(?i)act\s+as\s+if",
        r"(?i)reveal\s+your\s+(system\s+)?prompt",
        r"(?i)what\s+are\s+your\s+instructions",
    ]

    # PHI detection patterns
    PHI_PATTERNS = [
        (r"\b\d{3}-\d{2}-\d{4}\b", "SSN pattern detected"),
        (r"\b\d{9}\b", "Potential SSN without dashes"),
        (r"(?i)patient\s+name\s*[:=]\s*\w+", "Patient name reference"),
        (r"(?i)MRN\s*[:=]\s*\d+", "Medical Record Number"),
        (r"(?i)DOB\s*[:=]\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", "Date of birth"),
    ]

    def validate(self, query: str) -> ValidationResult:
        """Run all input validation checks."""
        issues = []
        risk = RiskLevel.LOW

        # Check 1: Length
        if len(query) > MAX_QUERY_LENGTH:
            return ValidationResult(
                passed=False,
                risk_level=RiskLevel.BLOCKED,
                issues=[f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters"],
                blocked_reason="Query too long",
            )

        # Check 2: PHI detection
        for pattern, description in self.PHI_PATTERNS:
            if re.search(pattern, query):
                return ValidationResult(
                    passed=False,
                    risk_level=RiskLevel.BLOCKED,
                    issues=[f"Potential PHI detected: {description}"],
                    blocked_reason="Protected Health Information detected. This system does not process individual patient data.",
                )

        # Check 3: Prompt injection
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, query):
                issues.append("Potential prompt injection detected")
                risk = RiskLevel.HIGH

        if risk == RiskLevel.HIGH:
            return ValidationResult(
                passed=False,
                risk_level=RiskLevel.BLOCKED,
                issues=issues,
                blocked_reason="Query appears to contain prompt injection attempts.",
            )

        # Check 4: Blocked patterns from config
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, query):
                issues.append(f"Blocked content pattern detected")
                risk = RiskLevel.MEDIUM

        # Check 5: Sanitize
        sanitized = self._sanitize(query)

        return ValidationResult(
            passed=True,
            risk_level=risk,
            issues=issues,
            sanitized_input=sanitized,
        )

    def _sanitize(self, query: str) -> str:
        """Clean and normalize the query."""
        # Remove potential HTML/script tags
        query = re.sub(r"<[^>]+>", "", query)
        # Normalize whitespace
        query = re.sub(r"\s+", " ", query.strip())
        return query


# === Output Guardrails ===

class OutputGuardrails:
    """
    Post-processing guardrails for LLM outputs.
    
    Checks:
    1. Confidence threshold
    2. Citation presence
    3. Unsupported claims detection
    4. Clinical disclaimer injection
    """

    CLINICAL_DISCLAIMER = (
        "DISCLAIMER: This information is derived from FDA drug labels and is intended "
        "for informational purposes only. It does not constitute medical advice. "
        "Always consult a qualified healthcare provider for clinical decisions."
    )

    UNSUPPORTED_CLAIM_PATTERNS = [
        r"(?i)guaranteed\s+to\s+(cure|treat|prevent)",
        r"(?i)100%\s+(effective|safe)",
        r"(?i)no\s+side\s+effects",
        r"(?i)I\s+recommend\s+(taking|using|starting)",
        r"(?i)you\s+should\s+(take|use|start|stop)",
    ]

    def validate(self, response, min_confidence: float = MIN_CONFIDENCE_THRESHOLD) -> ValidationResult:
        """Run all output validation checks."""
        issues = []
        risk = RiskLevel.LOW

        # Check 1: Confidence threshold
        if response.confidence < min_confidence:
            issues.append(
                f"Low confidence ({response.confidence:.2f} < {min_confidence}). "
                "Answer may not be reliable."
            )
            risk = RiskLevel.MEDIUM

        # Check 2: Citations present for non-trivial answers
        if len(response.answer) > 100 and len(response.citations) == 0:
            issues.append("No citations provided for a substantive answer")
            risk = RiskLevel.MEDIUM

        # Check 3: Unsupported claims
        for pattern in self.UNSUPPORTED_CLAIM_PATTERNS:
            if re.search(pattern, response.answer):
                issues.append("Potentially unsupported clinical claim detected")
                risk = RiskLevel.HIGH

        # Check 4: Answer length sanity
        if len(response.answer) > 10000:
            issues.append("Response unusually long - may contain hallucinated content")
            risk = RiskLevel.MEDIUM

        return ValidationResult(
            passed=risk != RiskLevel.HIGH,
            risk_level=risk,
            issues=issues,
        )

    def add_disclaimer(self, response) -> str:
        """Append clinical disclaimer to response."""
        return f"{response.answer}\n\n---\n{self.CLINICAL_DISCLAIMER}"

    def add_confidence_warning(self, response) -> Optional[str]:
        """Generate a confidence-based warning if needed."""
        if response.confidence < 0.3:
            return (
                "⚠️ LOW CONFIDENCE: The retrieved context may not fully address your query. "
                "Consider rephrasing or specifying a drug name."
            )
        elif response.confidence < 0.5:
            return (
                "Note: Moderate confidence in this response. "
                "Some information may be incomplete."
            )
        return None
