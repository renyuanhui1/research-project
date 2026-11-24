from pathlib import Path
import cv2, numpy as np, sys

p = Path(sys.argv[1])
rng = np.random.default_rng(51677)
field = np.zeros((768, 768), np.float32)
for cells, w in ((4, 0.54), (9, 0.31), (22, 0.11), (54, 0.04)):
    tile = rng.random((cells, cells), dtype=np.float32)
    padded = np.pad(tile, 1, mode="wrap")
    dim = int(round(768 / cells))
    enlarged = cv2.resize(
        padded, ((cells + 2) * dim, (cells + 2) * dim), interpolation=cv2.INTER_CUBIC
    )
    field += w * cv2.resize(enlarged[dim:-dim, dim:-dim], (768, 768))
cv2.imwrite(str(p / "MeadowMask.png"), (np.clip(field, 0, 1) * 255).astype("uint8"))
