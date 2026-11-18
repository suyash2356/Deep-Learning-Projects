# Sudoku grid settings
N = 6              # grid is N x N (use 4 for quick debugging)
BOX_R, BOX_C = 2, 3   # box shape: 2 rows x 3 cols (for 4x4 use 2, 2)
NUM_CELLS = N * N

# Token ids
BLANK = 0          # empty cell
SEP = N + 1        # separator token (transformer)
MASK = N + 1       # mask token (NCM-lite)
VOCAB_SIZE = N + 2