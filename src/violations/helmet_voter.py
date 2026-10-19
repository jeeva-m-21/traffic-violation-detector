"""Temporal voting and persistence tracker for motorcycle helmet compliance.

This module aggregates multi-frame observations for each rider track ID to eliminate
transient false positives caused by motion blur, occlusions, or camera angle shifts.
"""

from typing import Dict, Tuple, List, Any
from collections import defaultdict


class HelmetViolationVoter:
    """Accumulates helmet compliance predictions across video frames per track ID."""

    def __init__(self, min_observations: int = 4, violation_threshold: float = 0.6):
        """Initializes the temporal voter.

        Args:
            min_observations: Minimum number of observed frames before deciding violation status.
            violation_threshold: Fraction of no-helmet observations needed to trigger a violation alert.
        """
        self.min_observations = min_observations
        self.violation_threshold = violation_threshold

        # track_id -> list of (is_helmet: bool, conf: float)
        self.history: Dict[int, List[Tuple[bool, float]]] = defaultdict(list)

    def add_observation(self, track_id: int, is_helmet: bool, confidence: float) -> None:
        """Records an observation for a given rider track ID."""
        self.history[track_id].append((is_helmet, float(confidence)))

    def is_violation(self, track_id: int) -> bool:
        """Determines if a rider track ID is confirmed to be in violation (no helmet).

        Requires at least `min_observations` and at least `violation_threshold`
        fraction of observations showing no helmet.
        """
        is_viol, _ = self.get_violation_status(track_id)
        return is_viol

    def get_violation_status(self, track_id: int) -> Tuple[bool, float]:
        """Returns the violation decision and weighted confidence for a rider track ID.

        Returns:
            Tuple of (is_violation: bool, mean_confidence: float).
        """
        obs = self.history.get(track_id, [])
        if len(obs) < self.min_observations:
            return (False, 0.0)

        no_helmet_confs = [conf for (is_h, conf) in obs if not is_h]
        helmet_confs = [conf for (is_h, conf) in obs if is_h]

        total = len(obs)
        no_helmet_ratio = len(no_helmet_confs) / float(total)

        if no_helmet_ratio >= self.violation_threshold:
            mean_conf = sum(no_helmet_confs) / len(no_helmet_confs) if no_helmet_confs else 0.0
            return (True, float(mean_conf))
        else:
            mean_conf = sum(helmet_confs) / len(helmet_confs) if helmet_confs else 0.0
            return (False, float(mean_conf))

    def get_stats(self) -> Dict[str, Any]:
        """Calculates global compliance statistics across all tracked riders."""
        total_riders = len(self.history)
        violations = 0
        compliant = 0
        pending = 0

        for track_id, obs in self.history.items():
            if len(obs) < self.min_observations:
                pending += 1
            else:
                is_viol, _ = self.get_violation_status(track_id)
                if is_viol:
                    violations += 1
                else:
                    compliant += 1

        resolved = violations + compliant
        compliance_rate = (compliant / float(resolved)) if resolved > 0 else 1.0

        return {
            "total_riders": total_riders,
            "violations": violations,
            "compliant": compliant,
            "pending": pending,
            "compliance_rate": float(compliance_rate),
        }
