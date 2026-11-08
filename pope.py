"""POPE subset: 300 examples drawn with a fixed seed from the lmms-lab/POPE 'random' split.

The paper uses "a 300-example POPE evaluation set" but does not say which 300, so this choice is ours.
"""
import io
from pathlib import Path

import pandas as pd
from PIL import Image

PARQUET = Path(__file__).parent / "data/pope/Full/random-00000-of-00001.parquet"
N, SEED = 300, 0


def load_subset() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET).sample(n=N, random_state=SEED).reset_index(drop=True)
    df["image"] = [Image.open(io.BytesIO(x["bytes"])).convert("RGB") for x in df["image"]]
    df["answer"] = df["answer"].str.strip().str.lower()
    return df
