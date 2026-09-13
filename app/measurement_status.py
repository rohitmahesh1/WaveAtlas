from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Optional, Tuple


MeasurementStatus = Literal["invalid", "review", "accepted"]

INVALID_STATUS: MeasurementStatus = "invalid"
REVIEW_STATUS: MeasurementStatus = "review"
ACCEPTED_STATUS: MeasurementStatus = "accepted"

EVIDENCE_RULE_NOT_CALIBRATED = "evidence_rule_not_calibrated"
EVIDENCE_RULE_NOT_PASSED = "evidence_rule_not_passed"
FALLBACK_PEAK = "fallback_peak"


@dataclass(frozen=True)
class MeasurementAssessment:
    """Separate numerical validity from scientific-evidence acceptance."""

    estimator_valid: bool
    measurement_status: MeasurementStatus
    status_reasons: Tuple[str, ...]
    evidence_rule_version: Optional[str]

    def metadata(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["status_reasons"] = list(self.status_reasons)
        return payload


def assess_standard_measurement(
    *,
    estimator_valid: bool,
    failure_reason: Optional[str] = None,
    fallback_candidate: bool = False,
    evidence_accepted: Optional[bool] = None,
    evidence_rule_version: Optional[str] = None,
) -> MeasurementAssessment:
    """Classify a Standard measurement without inventing an evidence threshold.

    ``estimator_valid`` reports whether the numerical calculation is admissible.
    Scientific acceptance is a separate state and requires a named, calibrated
    evidence rule. Until such a rule is supplied, computable measurements remain
    available with ``measurement_status == "review"``.
    """

    if not estimator_valid:
        reasons = _unique_reasons(failure_reason, FALLBACK_PEAK if fallback_candidate else None)
        return MeasurementAssessment(
            estimator_valid=False,
            measurement_status=INVALID_STATUS,
            status_reasons=reasons or ("estimator_invalid",),
            evidence_rule_version=evidence_rule_version,
        )

    if fallback_candidate:
        return MeasurementAssessment(
            estimator_valid=False,
            measurement_status=REVIEW_STATUS,
            status_reasons=(FALLBACK_PEAK,),
            evidence_rule_version=evidence_rule_version,
        )

    if evidence_accepted is True:
        if not evidence_rule_version:
            raise ValueError("Accepted measurements require an evidence_rule_version")
        return MeasurementAssessment(
            estimator_valid=True,
            measurement_status=ACCEPTED_STATUS,
            status_reasons=(),
            evidence_rule_version=evidence_rule_version,
        )

    if evidence_accepted is False:
        if not evidence_rule_version:
            raise ValueError("Evaluated measurements require an evidence_rule_version")
        reasons = _unique_reasons(failure_reason, EVIDENCE_RULE_NOT_PASSED)
        return MeasurementAssessment(
            estimator_valid=True,
            measurement_status=REVIEW_STATUS,
            status_reasons=reasons,
            evidence_rule_version=evidence_rule_version,
        )

    return MeasurementAssessment(
        estimator_valid=True,
        measurement_status=REVIEW_STATUS,
        status_reasons=(EVIDENCE_RULE_NOT_CALIBRATED,),
        evidence_rule_version=None,
    )


def _unique_reasons(*reasons: Optional[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(reason for reason in reasons if reason))
