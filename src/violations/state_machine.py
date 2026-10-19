"""Temporal Hysteresis & State Machine for Wrong-Way Enforcement.

Maintains sliding-window observation history per track to eliminate
single-frame detection noise and enforce strict K-of-N persistence before
confirming infractions.
"""

from collections import defaultdict, deque
from typing import Dict, Optional
from src.violations.kinematics import TrackKinematics, ViolationState


class WrongWayStateMachine:
    """Manages temporal state transitions with K-of-N hysteresis window."""

    def __init__(
        self,
        window_size: int = 10,
        min_violation_count: int = 8,
    ) -> None:
        self.window_size = window_size
        self.min_violation_count = min_violation_count

        # track_id -> deque of candidate ViolationStates
        self.track_windows: Dict[int, deque] = defaultdict(lambda: deque(maxlen=window_size))
        # track_id -> confirmed state
        self.confirmed_states: Dict[int, ViolationState] = {}

    def update(self, kin: TrackKinematics) -> ViolationState:
        """Updates track history with current frame kinematics and evaluates confirmed state."""
        t_id = kin.track_id
        inst_state = kin.violation_state

        window = self.track_windows[t_id]
        window.append(inst_state)

        # Count occurrences in window
        wrong_way_count = sum(1 for s in window if s == ViolationState.CONFIRMED_WRONG_WAY)
        reversing_count = sum(1 for s in window if s == ViolationState.REVERSING_VIOLATION)
        stationary_count = sum(1 for s in window if s == ViolationState.STATIONARY_HAZARD)
        overtaking_count = sum(1 for s in window if s == ViolationState.LEGAL_OVERTAKING)

        # Evaluation
        if wrong_way_count >= self.min_violation_count:
            final_state = ViolationState.CONFIRMED_WRONG_WAY
        elif reversing_count >= self.min_violation_count:
            final_state = ViolationState.REVERSING_VIOLATION
        elif stationary_count >= self.min_violation_count:
            final_state = ViolationState.STATIONARY_HAZARD
        elif overtaking_count >= (len(window) // 2):
            final_state = ViolationState.LEGAL_OVERTAKING
        elif not kin.in_ego_corridor:
            final_state = ViolationState.NORMAL_FLOW
        else:
            final_state = self.confirmed_states.get(t_id, inst_state)

        self.confirmed_states[t_id] = final_state
        return final_state

    def get_state(self, track_id: int) -> ViolationState:
        """Retrieves current confirmed state for a track."""
        return self.confirmed_states.get(track_id, ViolationState.NORMAL_FLOW)
