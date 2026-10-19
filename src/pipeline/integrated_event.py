"""Integrated Traffic Violation Event Lifecycle Engine.

Coordinates the end-to-end lifecycle of a traffic violation:
1. Detects motorcycle rider helmet infractions.
2. Triggers targeted license plate scanning on the associated vehicle.
3. Links the extracted license plate text and confidence to the violation incident.
4. Stores snapshot evidence paths and provides structured serialization.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import time


@dataclass
class ViolationIncident:
    """Represents a discrete traffic violation event linked to a vehicle and offender."""

    incident_id: str
    rider_id: int
    motorcycle_id: Optional[int]
    timestamp_sec: float
    last_seen_sec: float = field(default=0.0)
    violation_type: str = "NO_HELMET"
    helmet_confidence: float = 0.0
    plate_number: Optional[str] = None
    plate_confidence: float = 0.0
    status: str = "PENDING_PLATE"  # "PENDING_PLATE", "CONFIRMED", "UNRESOLVED_PLATE"
    snapshot_path: Optional[str] = None
    sample_bbox: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.last_seen_sec == 0.0:
            self.last_seen_sec = self.timestamp_sec

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the incident to a JSON-compatible dictionary."""
        return asdict(self)


class ViolationEventManager:
    """Manages active and confirmed traffic violation incidents."""

    def __init__(self) -> None:
        self.incidents: Dict[str, ViolationIncident] = {}
        # rider_id -> incident_id mapping
        self.rider_to_incident: Dict[int, str] = {}
        # motorcycle_id -> set of incident_ids
        self.moto_to_incidents: Dict[int, List[str]] = {}
        self._next_id = 1

    def trigger_violation(
        self,
        rider_id: int,
        motorcycle_id: Optional[int],
        timestamp_sec: float,
        helmet_confidence: float,
        bbox: Optional[List[int]] = None,
    ) -> ViolationIncident:
        """Triggers or updates a violation incident for a given rider.

        If an incident for the rider already exists, updates the timestamp and confidence.
        Otherwise, assigns a new unique incident ID.
        """
        if rider_id in self.rider_to_incident:
            inc_id = self.rider_to_incident[rider_id]
            inc = self.incidents[inc_id]
            inc.last_seen_sec = round(float(timestamp_sec), 2)
            inc.helmet_confidence = max(inc.helmet_confidence, round(float(helmet_confidence), 3))
            if bbox:
                inc.sample_bbox = bbox
            return inc

        inc_id = f"INC-{self._next_id:03d}"
        self._next_id += 1

        inc = ViolationIncident(
            incident_id=inc_id,
            rider_id=rider_id,
            motorcycle_id=motorcycle_id,
            timestamp_sec=round(float(timestamp_sec), 2),
            last_seen_sec=round(float(timestamp_sec), 2),
            violation_type="NO_HELMET",
            helmet_confidence=round(float(helmet_confidence), 3),
            status="PENDING_PLATE",
            sample_bbox=bbox or [],
        )

        self.incidents[inc_id] = inc
        self.rider_to_incident[rider_id] = inc_id

        if motorcycle_id is not None:
            if motorcycle_id not in self.moto_to_incidents:
                self.moto_to_incidents[motorcycle_id] = []
            self.moto_to_incidents[motorcycle_id].append(inc_id)

        return inc

    def attach_plate(
        self,
        motorcycle_id: int,
        plate_string: str,
        plate_confidence: float,
        snapshot_path: Optional[str] = None,
    ) -> bool:
        """Attaches an extracted license plate to all pending incidents for a motorcycle."""
        inc_ids = self.moto_to_incidents.get(motorcycle_id, [])
        if not inc_ids:
            return False

        updated = False
        for inc_id in inc_ids:
            inc = self.incidents[inc_id]
            inc.plate_number = plate_string
            inc.plate_confidence = round(float(plate_confidence), 3)
            inc.status = "CONFIRMED"
            if snapshot_path:
                inc.snapshot_path = snapshot_path
            updated = True

        return updated

    def get_incident_by_rider(self, rider_id: int) -> Optional[ViolationIncident]:
        """Retrieves incident for a rider ID, if exists."""
        inc_id = self.rider_to_incident.get(rider_id)
        return self.incidents.get(inc_id) if inc_id else None

    def get_all_incidents(self) -> List[ViolationIncident]:
        """Returns all recorded incidents sorted chronologically."""
        return sorted(self.incidents.values(), key=lambda x: x.timestamp_sec)

    def get_confirmed_incidents(self) -> List[ViolationIncident]:
        """Returns incidents that have successfully resolved license plates."""
        return [inc for inc in self.incidents.values() if inc.status == "CONFIRMED"]

    def export_summary(self) -> Dict[str, Any]:
        """Exports aggregate metrics and all incident records."""
        all_inc = self.get_all_incidents()
        confirmed = self.get_confirmed_incidents()
        return {
            "total_incidents": len(all_inc),
            "confirmed_with_plate": len(confirmed),
            "pending_or_unresolved_plate": len(all_inc) - len(confirmed),
            "incidents": [inc.to_dict() for inc in all_inc],
        }
