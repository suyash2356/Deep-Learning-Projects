import os
import random
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from src.config import N
from src.sudoku import random_solution, make_puzzle, is_valid, count_solutions

DATA_DIR = "data"


def _make_one(seed):
    """Make one (puzzle, solution) pair. Seeded so every worker is different."""
    random.seed(seed)
    sol = random_solution()
    puz = make_puzzle(sol, random.randint(12, 24))
    return puz.astype(np.int8), sol.astype(np.int8)


def generate(n, start_seed):
    with Pool() as pool:
        out = list(tqdm(pool.imap(_make_one, range(start_seed, start_seed + n),
                                  chunksize=50), total=n))
    puzzles = np.stack([o[0] for o in out])
    solutions = np.stack([o[1] for o in out])
    return puzzles, solutions


def _remove_overlap(puzzles, solutions, banned):
    keep = [i for i, s in enumerate(solutions) if s.tobytes() not in banned]
    return puzzles[keep], solutions[keep]


def load(split):
    """split: 'train', 'val' or 'test'. Returns (puzzles, solutions), each (n, N, N)."""
    d = np.load(os.path.join(DATA_DIR, f"{split}.npz"))
    return d["puzzles"], d["solutions"]


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Generating test...")
    te_p, te_s = generate(1000, start_seed=10_000_000)
    print("Generating val...")
    va_p, va_s = generate(1000, start_seed=20_000_000)
    print("Generating train...")
    tr_p, tr_s = generate(20000, start_seed=0)

    # make sure no solution is shared between splits
    test_keys = {s.tobytes() for s in te_s}
    va_p, va_s = _remove_overlap(va_p, va_s, test_keys)
    val_keys = {s.tobytes() for s in va_s}
    tr_p, tr_s = _remove_overlap(tr_p, tr_s, test_keys | val_keys)

    for name, p, s in [("train", tr_p, tr_s), ("val", va_p, va_s), ("test", te_p, te_s)]:
        np.savez_compressed(os.path.join(DATA_DIR, f"{name}.npz"), puzzles=p, solutions=s)
        clues = (p != 0).sum(axis=(1, 2))
        print(f"{name:5s}: {len(p):6d} puzzles | clues min/mean/max = "
              f"{clues.min()}/{clues.mean():.1f}/{clues.max()}")

    # --- verification on a sample ---
    p, s = load("test")
    for i in range(100):
        assert is_valid(s[i].astype(int)), "bad solution"
        assert count_solutions(p[i].astype(int)) == 1, "puzzle not unique"
        assert ((p[i] == 0) | (p[i] == s[i])).all(), "clue != solution"
    print("Verification passed (100 test puzzles checked).")