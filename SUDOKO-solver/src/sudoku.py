import random
import numpy as np
from src.config import N, BOX_R, BOX_C


# ---------- helpers ----------
def candidates(grid, r, c):
    """Digits that can legally go in cell (r, c)."""
    used = set(grid[r, :].tolist()) | set(grid[:, c].tolist())
    br, bc = (r // BOX_R) * BOX_R, (c // BOX_C) * BOX_C
    used |= set(grid[br:br + BOX_R, bc:bc + BOX_C].flatten().tolist())
    return [v for v in range(1, N + 1) if v not in used]


def _search(grid, found, limit, randomize):
    """Backtracking: always fill the empty cell with the fewest options first."""
    best = None
    for r in range(N):
        for c in range(N):
            if grid[r, c] == 0:
                cand = candidates(grid, r, c)
                if best is None or len(cand) < len(best[2]):
                    best = (r, c, cand)
                if len(cand) == 0:
                    return                      # dead end
    if best is None:                            # no empty cell left -> solved
        found.append(grid.copy())
        return
    r, c, cand = best
    if randomize:
        random.shuffle(cand)
    for v in cand:
        grid[r, c] = v
        _search(grid, found, limit, randomize)
        grid[r, c] = 0                          # undo (backtrack)
        if len(found) >= limit:
            return


# ---------- solving ----------
def solve(puzzle):
    """Return the solved grid, or None if unsolvable."""
    found = []
    _search(puzzle.copy(), found, limit=1, randomize=False)
    return found[0] if found else None


def count_solutions(puzzle, limit=2):
    """Count solutions, stopping early at `limit`."""
    found = []
    _search(puzzle.copy(), found, limit=limit, randomize=False)
    return len(found)


# ---------- generating ----------
def random_solution():
    """A random full valid grid."""
    found = []
    _search(np.zeros((N, N), dtype=int), found, limit=1, randomize=True)
    return found[0]


def make_puzzle(solution, n_clues):
    """Remove cells one by one, but only if the puzzle keeps exactly 1 solution."""
    puzzle = solution.copy()
    cells = [(r, c) for r in range(N) for c in range(N)]
    random.shuffle(cells)
    clues = N * N
    for r, c in cells:
        if clues <= n_clues:
            break
        old = puzzle[r, c]
        puzzle[r, c] = 0
        if count_solutions(puzzle) != 1:
            puzzle[r, c] = old                  # removal broke uniqueness, undo
        else:
            clues -= 1
    return puzzle


# ---------- checking ----------
def _units():
    """All rows, columns and boxes as lists of (r, c) cells."""
    units = []
    for r in range(N):
        units.append([(r, c) for c in range(N)])
    for c in range(N):
        units.append([(r, c) for r in range(N)])
    for br in range(0, N, BOX_R):
        for bc in range(0, N, BOX_C):
            units.append([(r, c) for r in range(br, br + BOX_R)
                                 for c in range(bc, bc + BOX_C)])
    return units


def count_violations(grid):
    """Number of repeated digits across all rows, columns, boxes."""
    total = 0
    for unit in _units():
        vals = [grid[r, c] for r, c in unit if grid[r, c] != 0]
        total += len(vals) - len(set(vals))
    return total


def is_valid(grid):
    """True only if the grid is full AND has zero rule breaks."""
    return bool((grid != 0).all()) and count_violations(grid) == 0


def matches_clues(solution, puzzle):
    """Model must not change the given clues."""
    mask = puzzle != 0
    return bool((solution[mask] == puzzle[mask]).all())


# ---------- pretty print ----------
def show(grid):
    for r in range(N):
        if r % BOX_R == 0 and r > 0:
            print("-" * (2 * N + 2 * (N // BOX_C - 1) - 1))
        row = ""
        for c in range(N):
            if c % BOX_C == 0 and c > 0:
                row += "| "
            row += (str(grid[r, c]) if grid[r, c] != 0 else ".") + " "
        print(row)


# ---------- self-test ----------
if __name__ == "__main__":
    import time
    t0 = time.time()
    n_tests = 200
    for i in range(n_tests):
        sol = random_solution()
        assert is_valid(sol), "random_solution gave an invalid grid"

        puz = make_puzzle(sol, random.randint(12, 24))
        assert count_solutions(puz) == 1, "puzzle not unique"
        assert (solve(puz) == sol).all(), "solver disagrees with solution"
        assert matches_clues(sol, puz), "clues changed"

    print(f"All {n_tests} tests passed in {time.time() - t0:.1f}s")

    print("\nSample puzzle:")
    show(puz)
    print("\nIts solution:")
    show(sol)

    # a deliberately broken grid -> violations must be > 0
    bad = sol.copy()
    bad[0, 0] = bad[0, 1]
    print("\nViolations in broken grid:", count_violations(bad))