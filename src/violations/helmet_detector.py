"""Motorcycle rider association and head ROI extraction module.

This module provides geometric functions to associate detected pedestrians/persons
with detected motorcycles (identifying riders and pillion passengers), and extracts
the upper head region of interest (ROI) for helmet classification.
"""

from typing import List, Dict, Tuple, Optional, Any


def extract_head_roi(
    person_bbox: Tuple[int, int, int, int],
    head_ratio: float = 0.3,
    img_width: Optional[int] = None,
    img_height: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    """Extracts the upper bounding box region corresponding to a rider's head.

    Args:
        person_bbox: (x1, y1, x2, y2) bounding box of detected person.
        head_ratio: Fraction of the person's height from the top to include (default 0.3).
        img_width: Optional image width to clamp coordinates within bounds.
        img_height: Optional image height to clamp coordinates within bounds.

    Returns:
        (hx1, hy1, hx2, hy2) bounding box of the head region.
    """
    x1, y1, x2, y2 = person_bbox
    height = max(0, y2 - y1)
    head_height = int(height * head_ratio)

    hx1 = x1
    hy1 = y1
    hx2 = x2
    hy2 = y1 + head_height

    if img_width is not None:
        hx1 = max(0, min(img_width, hx1))
        hx2 = max(0, min(img_width, hx2))
    if img_height is not None:
        hy1 = max(0, min(img_height, hy1))
        hy2 = max(0, min(img_height, hy2))

    return (hx1, hy1, hx2, hy2)


def compute_rider_motorcycle_overlap(
    person_bbox: Tuple[int, int, int, int],
    moto_bbox: Tuple[int, int, int, int],
) -> float:
    """Computes geometric plausibility score for a person being a rider of a motorcycle.

    A rider sits atop a motorcycle, meaning:
    1. Horizontal spans overlap significantly.
    2. The person's bottom (legs/seat) overlaps with or is near the top portion of the motorcycle.
    3. The person's head is above the bottom of the motorcycle.

    Returns:
        A score >= 0.0 indicating overlap plausibility (0.0 means no association).
    """
    px1, py1, px2, py2 = person_bbox
    mx1, my1, mx2, my2 = moto_bbox

    p_w = max(1, px2 - px1)
    p_h = max(1, py2 - py1)
    m_w = max(1, mx2 - mx1)
    m_h = max(1, my2 - my1)

    # 1. Horizontal intersection
    ix1 = max(px1, mx1)
    ix2 = min(px2, mx2)
    horiz_overlap = max(0, ix2 - ix1)
    horiz_ratio = horiz_overlap / float(min(p_w, m_w))

    if horiz_ratio < 0.2:
        return 0.0

    # 2. Vertical plausibility:
    # Person should not be completely below motorcycle
    if py1 > my2:
        return 0.0
    # Person's bottom should reach into the motorcycle or be very close to its top
    # e.g., py2 >= my1 - 0.2 * m_h and py2 <= my2 + 0.3 * m_h
    if py2 < (my1 - 0.2 * m_h) or py1 > my2:
        return 0.0

    # Vertical intersection of person's lower half with motorcycle
    vert_overlap = max(0, min(py2, my2) - max(py1, my1))
    vert_ratio = vert_overlap / float(min(p_h, m_h))

    # Combined score
    score = (horiz_ratio * 0.6) + (vert_ratio * 0.4)
    return float(score)


def match_riders_to_motorcycles(
    persons: List[Dict[str, Any]],
    motorcycles: List[Dict[str, Any]],
    min_score_thresh: float = 0.15,
) -> Dict[int, int]:
    """Matches detected persons to motorcycles based on geometric overlap.

    Supports multiple riders (driver and pillion passenger) on the same motorcycle.

    Args:
        persons: List of dicts, each with 'id' and 'bbox' (x1, y1, x2, y2).
        motorcycles: List of dicts, each with 'id' and 'bbox' (x1, y1, x2, y2).
        min_score_thresh: Minimum overlap score to consider a valid rider association.

    Returns:
        Mapping of person_id -> motorcycle_id for all matched riders.
    """
    matches: Dict[int, int] = {}
    if not persons or not motorcycles:
        return matches

    for p in persons:
        p_id = p["id"]
        p_bbox = p["bbox"]

        best_m_id: Optional[int] = None
        best_score = 0.0

        for m in motorcycles:
            m_id = m["id"]
            m_bbox = m["bbox"]

            score = compute_rider_motorcycle_overlap(p_bbox, m_bbox)
            if score > best_score and score >= min_score_thresh:
                best_score = score
                best_m_id = m_id

        if best_m_id is not None:
            matches[p_id] = best_m_id

    return matches
