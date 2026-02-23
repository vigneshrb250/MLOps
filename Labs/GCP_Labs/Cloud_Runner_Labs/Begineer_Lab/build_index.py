import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
    "popularity",
    "duration_ms",
]

def main() -> None:
    data_path = Path(os.getenv("SONGS_CSV", "data/songs.csv"))
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df = df.rename(columns={"artists": "artist"})
    df["artist"] = df["artist"].astype(str).str.split(";").str[0].str.strip()
    # Basic cleanup / required columns check
    required = {"track_name", "artist"} | set(FEATURE_COLS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {data_path}: {sorted(missing)}")

    # Drop rows with missing features
    df = df.dropna(subset=FEATURE_COLS + ["track_name", "artist"]).reset_index(drop=True)

    X = df[FEATURE_COLS].astype(float).to_numpy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # L2 normalize for cosine similarity via dot product
    norms = np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-12
    Xn = Xs / norms

    # Save artifacts
    np.save(out_dir / "embeddings.npy", Xn.astype(np.float32))

    meta = df[["track_name", "artist"]].to_dict(orient="records")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    (out_dir / "feature_cols.json").write_text(json.dumps(FEATURE_COLS, indent=2), encoding="utf-8")

    # Save scaler params (so inference is consistent)
    scaler_blob = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    (out_dir / "scaler.json").write_text(json.dumps(scaler_blob, indent=2), encoding="utf-8")

    print(f"✅ Built index: {len(df)} songs")
    print(f"Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()