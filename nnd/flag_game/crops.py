from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class CropBox:
    crop_index: int
    top: int
    left: int
    height: int
    width: int

    def to_dict(self) -> dict[str, int]:
        return {
            "crop_index": self.crop_index,
            "top": self.top,
            "left": self.left,
            "height": self.height,
            "width": self.width,
        }


def sample_random_crops(
    canvas_width: int,
    canvas_height: int,
    tile_width: int,
    tile_height: int,
    n_agents: int,
    rng: random.Random,
    target_overlap: float | None = None,
    search_trials: int = 200,
    overlap_mode: str = "duplicated_redundancy",
) -> list[CropBox]:
    if tile_width > canvas_width:
        raise ValueError("tile_width must be <= canvas_width")
    if tile_height > canvas_height:
        raise ValueError("tile_height must be <= canvas_height")
    positions = [(box.top, box.left) for box in all_crop_boxes(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        tile_width=tile_width,
        tile_height=tile_height,
    )]
    if not positions:
        raise ValueError("No valid crop positions are available")

    if target_overlap is not None:
        if not (0.0 <= target_overlap <= 1.0):
            raise ValueError("target_overlap must be in [0, 1]")
        if search_trials < 1:
            raise ValueError("search_trials must be >= 1")
        # Exact zero-overlap is a discrete tiling case. When it is geometrically
        # possible, build it directly instead of relying on random search.
        exact_zero_positions = None
        if abs(target_overlap) <= 1e-12:
            exact_zero_positions = _sample_exact_zero_overlap_positions(
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                tile_width=tile_width,
                tile_height=tile_height,
                n_agents=n_agents,
                rng=rng,
            )
        if exact_zero_positions is not None:
            selected = exact_zero_positions
        elif overlap_mode == "distinct_geometric":
            selected = _sample_distinct_target_overlap_positions(
                positions=positions,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                tile_width=tile_width,
                tile_height=tile_height,
                n_agents=n_agents,
                rng=rng,
                target_overlap=target_overlap,
                search_trials=search_trials,
            )
        elif overlap_mode in {"duplicated_redundancy", "legacy_clustered"}:
            best_positions: list[tuple[int, int]] | None = None
            best_distance = float("inf")
            for candidate in _target_overlap_candidates(
                positions=positions,
                n_agents=n_agents,
                rng=rng,
                search_trials=search_trials,
            ):
                boxes = [
                    CropBox(
                        crop_index=crop_index,
                        top=top,
                        left=left,
                        height=tile_height,
                        width=tile_width,
                    )
                    for crop_index, (top, left) in enumerate(candidate)
                ]
                overlap = mean_pairwise_overlap(boxes)
                distance = abs(overlap - target_overlap)
                if distance < best_distance:
                    best_distance = distance
                    best_positions = candidate
                    if distance <= 1e-9:
                        break
            assert best_positions is not None
            selected = best_positions
        else:
            raise ValueError(
                "overlap_mode must be one of "
                f"{('duplicated_redundancy', 'legacy_clustered', 'distinct_geometric')}, got {overlap_mode!r}"
            )
    elif n_agents <= len(positions):
        selected = rng.sample(positions, n_agents)
    else:
        selected = [positions[rng.randrange(len(positions))] for _ in range(n_agents)]

    return [
        CropBox(
            crop_index=crop_index,
            top=top,
            left=left,
            height=tile_height,
            width=tile_width,
        )
        for crop_index, (top, left) in enumerate(selected)
    ]


def _sample_exact_zero_overlap_positions(
    *,
    canvas_width: int,
    canvas_height: int,
    tile_width: int,
    tile_height: int,
    n_agents: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    tilings: list[list[tuple[int, int]]] = []
    max_top_offset = min(tile_height - 1, canvas_height - tile_height)
    max_left_offset = min(tile_width - 1, canvas_width - tile_width)

    for top_offset in range(max_top_offset + 1):
        tops = list(range(top_offset, canvas_height - tile_height + 1, tile_height))
        if not tops:
            continue
        for left_offset in range(max_left_offset + 1):
            lefts = list(range(left_offset, canvas_width - tile_width + 1, tile_width))
            if not lefts:
                continue
            tiling = [(top, left) for top in tops for left in lefts]
            if len(tiling) >= n_agents:
                tilings.append(tiling)

    if not tilings:
        return None

    selected = list(rng.choice(tilings))
    if len(selected) > n_agents:
        selected = rng.sample(selected, n_agents)
    rng.shuffle(selected)
    return selected


def _pairwise_overlap_matrix(boxes: list[CropBox]) -> np.ndarray:
    size = len(boxes)
    matrix = np.zeros((size, size), dtype=float)
    for i, box_a in enumerate(boxes):
        for j in range(i + 1, size):
            overlap = crop_overlap_fraction(box_a, boxes[j])
            matrix[i, j] = overlap
            matrix[j, i] = overlap
    return matrix


def _selection_overlap_sum(selection: list[int], matrix: np.ndarray) -> float:
    total = 0.0
    for idx, left in enumerate(selection):
        for right in selection[idx + 1 :]:
            total += float(matrix[left, right])
    return total


def _selection_overlap_mean(selection: list[int], matrix: np.ndarray) -> float:
    if len(selection) < 2:
        return 0.0
    return _selection_overlap_sum(selection, matrix) / (len(selection) * (len(selection) - 1) / 2.0)


def _dense_distinct_seed_selection(
    *,
    boxes: list[CropBox],
    matrix: np.ndarray,
    n_agents: int,
    rng: random.Random,
) -> list[int]:
    anchor = rng.randrange(len(boxes))
    ranked = sorted(
        range(len(boxes)),
        key=lambda idx: (
            -float(matrix[anchor, idx]),
            abs(boxes[idx].top - boxes[anchor].top) + abs(boxes[idx].left - boxes[anchor].left),
            idx,
        ),
    )
    return ranked[:n_agents]


def _sample_distinct_target_overlap_positions(
    *,
    positions: list[tuple[int, int]],
    canvas_width: int,
    canvas_height: int,
    tile_width: int,
    tile_height: int,
    n_agents: int,
    rng: random.Random,
    target_overlap: float,
    search_trials: int,
) -> list[tuple[int, int]]:
    if n_agents > len(positions):
        raise ValueError(
            f"distinct_geometric overlap mode requires N <= available crop positions; "
            f"got N={n_agents}, positions={len(positions)}"
        )

    exact_zero_positions = None
    if abs(target_overlap) <= 1e-12:
        exact_zero_positions = _sample_exact_zero_overlap_positions(
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            tile_width=tile_width,
            tile_height=tile_height,
            n_agents=n_agents,
            rng=rng,
        )
    if exact_zero_positions is not None:
        return exact_zero_positions

    boxes = [
        CropBox(
            crop_index=idx,
            top=top,
            left=left,
            height=tile_height,
            width=tile_width,
        )
        for idx, (top, left) in enumerate(positions)
    ]
    matrix = _pairwise_overlap_matrix(boxes)
    all_indices = list(range(len(boxes)))
    pair_count = n_agents * (n_agents - 1) / 2.0

    best_selection: list[int] | None = None
    best_overlap = 0.0
    best_distance = float("inf")
    restarts = max(32, min(search_trials, 180))
    steps = max(600, min(search_trials * 12, 2400))

    for restart in range(restarts):
        if target_overlap >= 0.42 and restart % 3 == 0:
            selection = _dense_distinct_seed_selection(
                boxes=boxes,
                matrix=matrix,
                n_agents=n_agents,
                rng=rng,
            )
        else:
            selection = rng.sample(all_indices, n_agents)
        selected = set(selection)
        current_sum = _selection_overlap_sum(selection, matrix)
        current_overlap = current_sum / pair_count
        current_distance = abs(current_overlap - target_overlap)

        if current_distance < best_distance:
            best_selection = list(selection)
            best_overlap = current_overlap
            best_distance = current_distance
            if best_distance <= 1e-9:
                break

        for step in range(steps):
            out_pos = rng.randrange(n_agents)
            old_idx = selection[out_pos]
            new_idx = rng.choice(all_indices)
            if new_idx in selected:
                continue

            delta = 0.0
            for idx, other_idx in enumerate(selection):
                if idx == out_pos:
                    continue
                delta += float(matrix[new_idx, other_idx] - matrix[old_idx, other_idx])

            candidate_sum = current_sum + delta
            candidate_overlap = candidate_sum / pair_count
            candidate_distance = abs(candidate_overlap - target_overlap)
            temperature = max(0.0008, 0.035 * (1.0 - step / float(steps)))
            accept = candidate_distance < current_distance
            if not accept:
                accept = rng.random() < np.exp((current_distance - candidate_distance) / temperature)
            if accept:
                selection[out_pos] = new_idx
                selected.remove(old_idx)
                selected.add(new_idx)
                current_sum = candidate_sum
                current_overlap = candidate_overlap
                current_distance = candidate_distance

                if current_distance < best_distance:
                    best_selection = list(selection)
                    best_overlap = current_overlap
                    best_distance = current_distance
                    if best_distance <= 1e-9:
                        break
        if best_distance <= 1e-9:
            break

    if best_selection is None:
        raise ValueError("Failed to construct distinct-geometric crop assignment")
    if best_distance > 0.03:
        raise ValueError(
            f"Could not realize distinct_geometric target_overlap={target_overlap:.3f}; "
            f"closest realized value was {best_overlap:.3f}. Try a feasible target or increase overlap_search_trials."
        )
    return [positions[idx] for idx in best_selection]


def _target_overlap_candidates(
    *,
    positions: list[tuple[int, int]],
    n_agents: int,
    rng: random.Random,
    search_trials: int,
) -> list[list[tuple[int, int]]]:
    candidates: list[list[tuple[int, int]]] = []

    # Keep the old unconstrained random search as a baseline. This is good for
    # low-overlap regimes, but it almost never discovers high-overlap clustered
    # assignments at moderate or large N.
    for _ in range(search_trials):
        candidates.append([positions[rng.randrange(len(positions))] for _ in range(n_agents)])

    if n_agents <= 0:
        return candidates

    anchor_trials = max(1, min(search_trials, 32))
    anchors = [positions[rng.randrange(len(positions))] for _ in range(anchor_trials)]

    # Single-cluster candidates make high-redundancy conditions reachable.
    for anchor in anchors:
        candidates.append([anchor for _ in range(n_agents)])

    # Balanced K-cluster candidates approximate overlap ~= 1/K when cluster
    # anchors are well separated. This makes low/moderate target overlaps like
    # 0.3 reachable for large N without relying on rare random draws.
    max_clusters = min(n_agents, len(positions), 12)
    for cluster_count in range(2, max_clusters + 1):
        trials_for_k = max(1, search_trials // max_clusters)
        for _ in range(trials_for_k):
            cluster_anchors = rng.sample(positions, cluster_count)
            candidate: list[tuple[int, int]] = []
            base_count = n_agents // cluster_count
            remainder = n_agents % cluster_count
            for idx, anchor in enumerate(cluster_anchors):
                repeat = base_count + (1 if idx < remainder else 0)
                candidate.extend([anchor] * repeat)
            rng.shuffle(candidate)
            candidates.append(candidate)

    if len(positions) >= 2 and n_agents >= 2:
        # Unbalanced two-cluster candidates fill in high-but-not-perfect
        # redundancy, e.g. target_overlap around 0.9.
        for main_count in range(1, n_agents):
            secondary_count = n_agents - main_count
            for _ in range(max(1, search_trials // max(1, n_agents))):
                anchor_a, anchor_b = rng.sample(positions, 2)
                candidate = [anchor_a] * main_count + [anchor_b] * secondary_count
                rng.shuffle(candidate)
                candidates.append(candidate)

    return candidates


def all_crop_boxes(
    *,
    canvas_width: int,
    canvas_height: int,
    tile_width: int,
    tile_height: int,
) -> list[CropBox]:
    if tile_width > canvas_width:
        raise ValueError("tile_width must be <= canvas_width")
    if tile_height > canvas_height:
        raise ValueError("tile_height must be <= canvas_height")
    boxes: list[CropBox] = []
    crop_index = 0
    for top in range(canvas_height - tile_height + 1):
        for left in range(canvas_width - tile_width + 1):
            boxes.append(
                CropBox(
                    crop_index=crop_index,
                    top=top,
                    left=left,
                    height=tile_height,
                    width=tile_width,
                )
            )
            crop_index += 1
    return boxes


def crop_image(image: np.ndarray, box: CropBox) -> np.ndarray:
    return image[box.top : box.top + box.height, box.left : box.left + box.width, :].copy()


def scale_crop_box(box: CropBox, scale: int) -> CropBox:
    if scale < 1:
        raise ValueError("scale must be >= 1")
    return CropBox(
        crop_index=box.crop_index,
        top=box.top * scale,
        left=box.left * scale,
        height=box.height * scale,
        width=box.width * scale,
    )


def crop_overlap_fraction(box_a: CropBox, box_b: CropBox) -> float:
    top = max(box_a.top, box_b.top)
    left = max(box_a.left, box_b.left)
    bottom = min(box_a.top + box_a.height, box_b.top + box_b.height)
    right = min(box_a.left + box_a.width, box_b.left + box_b.width)
    if bottom <= top or right <= left:
        return 0.0
    intersection = float((bottom - top) * (right - left))
    min_area = float(min(box_a.height * box_a.width, box_b.height * box_b.width))
    if min_area <= 0.0:
        return 0.0
    return intersection / min_area


def mean_pairwise_overlap(boxes: list[CropBox]) -> float:
    if len(boxes) < 2:
        return 0.0
    total = 0.0
    count = 0
    for idx, box_a in enumerate(boxes):
        for box_b in boxes[idx + 1 :]:
            total += crop_overlap_fraction(box_a, box_b)
            count += 1
    return total / float(count) if count else 0.0
