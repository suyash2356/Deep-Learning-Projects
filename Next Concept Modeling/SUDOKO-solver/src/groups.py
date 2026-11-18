import os
import numpy as np
from src.config import N
from src.sudoku import _units, is_valid, show

# A fixed valid sudoku, used ONLY to decide which cells go in which group.
_PATTERN = {
    4: [[1, 2, 3, 4],
        [3, 4, 1, 2],
        [2, 1, 4, 3],
        [4, 3, 2, 1]],
    6: [[1, 2, 3, 4, 5, 6],
        [4, 5, 6, 1, 2, 3],
        [2, 3, 1, 5, 6, 4],
        [5, 6, 4, 2, 3, 1],
        [3, 1, 2, 6, 4, 5],
        [6, 4, 5, 3, 1, 2]],
}

NUM_LEVELS = N // 2                       # 6x6 -> 3 levels, 4x4 -> 2 levels
_pat = np.array(_PATTERN[N])

LEVEL_MAP = (_pat - 1) // 2               # (N, N) grid, values 0..NUM_LEVELS-1
LEVEL_FLAT = LEVEL_MAP.flatten()          # (N*N,) row-major: index = r*N + c


def cells_in_level(k):
    """Flat indices (0..35) of cells that belong to level k."""
    return np.where(LEVEL_FLAT == k)[0]


if __name__ == "__main__":
    # --- self-test ---
    assert is_valid(_pat), "pattern is not a valid sudoku"
    for k in range(NUM_LEVELS):
        idx = cells_in_level(k)
        for unit in _units():                      # rows, cols, boxes
            n_in_unit = sum(LEVEL_MAP[r, c] == k for r, c in unit)
            assert n_in_unit == 2, f"level {k} unbalanced"
        print(f"Level {k + 1}: {len(idx)} cells -> {idx.tolist()}")
    assert sorted(np.concatenate([cells_in_level(k) for k in range(NUM_LEVELS)])) \
        == list(range(N * N)), "levels must cover every cell exactly once"
    print("\nAll checks passed. Pass number of each cell:")
    show(LEVEL_MAP + 1)

    # --- picture ---
    import matplotlib.pyplot as plt
    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(LEVEL_MAP, cmap="Pastel1")
    for r in range(N):
        for c in range(N):
            ax.text(c, r, LEVEL_MAP[r, c] + 1, ha="center", va="center", fontsize=16)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Which pass fills each cell")
    plt.savefig("results/groups.png", dpi=150, bbox_inches="tight")
    print("Saved results/groups.png")