from collections import defaultdict
from typing import Dict, List, Optional, Tuple

class TemporalPlateVoter:
    """Aggregates plate readings across multiple video frames per vehicle track ID."""

    def __init__(self, min_confidence: float = 0.35, min_votes: int = 2):
        self.min_confidence = min_confidence
        self.min_votes = min_votes
        # track_id -> dict of plate_string -> total_weighted_score
        self.votes: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.vote_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def add_observation(self, track_id: int, plate_string: Optional[str], confidence: float) -> None:
        """Records a single plate observation for a tracked vehicle."""
        if not plate_string or confidence < self.min_confidence:
            return

        # Weight by confidence score
        self.votes[track_id][plate_string] += confidence
        self.vote_counts[track_id][plate_string] += 1

    def resolve_plate(self, track_id: int) -> Tuple[Optional[str], float, int]:
        """
        Resolves the consensus license plate string for track_id using weighted voting.
        Returns: (plate_string, average_confidence, vote_count)
        """
        if track_id not in self.votes or not self.votes[track_id]:
            return None, 0.0, 0

        candidates = self.votes[track_id]
        best_plate = max(candidates.keys(), key=lambda p: candidates[p])
        count = self.vote_counts[track_id][best_plate]
        avg_conf = candidates[best_plate] / count

        return best_plate, round(avg_conf, 3), count
