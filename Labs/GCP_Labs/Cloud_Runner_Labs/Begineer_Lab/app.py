import os
import json
import socket
import logging
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from flask import Flask, request, jsonify

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("cloudrun-songrec")

app = Flask(__name__)

ART_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
EMB_PATH = ART_DIR / "embeddings.npy"
META_PATH = ART_DIR / "meta.json"

_embeddings = None
_meta = None


def _load():
    global _embeddings, _meta
    if _embeddings is None or _meta is None:
        if not EMB_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError("Missing artifacts. Did you run build_index.py during image build?")
        _embeddings = np.load(EMB_PATH)  # shape: (N, D), already normalized
        _meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        log.info("Loaded index. songs=%d dim=%d", _embeddings.shape[0], _embeddings.shape[1])
    return _embeddings, _meta


def _find_song_index(song_query: str, artist_query: str, meta):
    song_q = (song_query or "").strip().lower()
    artist_q = (artist_query or "").strip().lower()

    if not song_q:
        return None

    # If artist is provided, prioritize exact song+artist match
    if artist_q:
        for i, m in enumerate(meta):
            if m["track_name"].lower() == song_q and m["artist"].lower() == artist_q:
                return i

        # fallback: song exact + artist substring
        for i, m in enumerate(meta):
            if m["track_name"].lower() == song_q and artist_q in m["artist"].lower():
                return i

    # Exact match on song only
    for i, m in enumerate(meta):
        if m["track_name"].lower() == song_q:
            return i

    # Substring fallback on song name
    for i, m in enumerate(meta):
        if song_q in m["track_name"].lower():
            return i

    return None


@app.get("/")
def home():
    emb, meta = _load()
    return jsonify(
        {
            "service": "cloud-run-song-recommender",
            "message": "Song similarity recommender (cosine similarity on audio features) - Vignesh version",
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "songs_indexed": int(emb.shape[0]),
            "how_to": {
                "recommend": "GET /recommend?song=<song name>&k=10",
                "examples": [
                    "/recommend?song=Shape%20of%20You&k=5",
                    "/recommend?song=Blinding%20Lights&k=10",
                ],
            },
        }
    )

@app.get("/search")
def search():
    _, meta = _load()
    q = request.args.get("q", "").strip().lower()
    if not q:
        return jsonify({"error": "Provide q"}), 400

    results = []
    for m in meta:
        if q in m["track_name"].lower() or q in m["artist"].lower():
            results.append({"track_name": m["track_name"], "artist": m["artist"]})
        if len(results) >= 20:
            break

    return jsonify({"q": q, "results": results})

@app.get("/health")
def health():
    ok = EMB_PATH.exists() and META_PATH.exists()
    return jsonify({"status": "ok" if ok else "missing_artifacts", "artifacts_present": ok}), (200 if ok else 500)


@app.get("/recommend")
def recommend():
    emb, meta = _load()

    song = request.args.get("song", "").strip()
    k = request.args.get("k", "10").strip()

    try:
        k = int(k)
        k = max(1, min(k, 25))
    except Exception:
        return jsonify({"error": "k must be an integer between 1 and 25"}), 400
    
    artist = request.args.get("artist", "").strip()
    idx = _find_song_index(song, artist, meta)

    if idx is None:
        # Return a few suggestions
        q = song.lower()
        suggestions = []
        for m in meta:
            if q and (q in m["track_name"].lower() or q in m["artist"].lower()):
                suggestions.append(f"{m['track_name']} - {m['artist']}")
            if len(suggestions) >= 10:
                break
        return jsonify(
            {
                "error": "Song not found in local dataset",
                "hint": "Try an exact track name from data/songs.csv",
                "suggestions": suggestions,
            }
        ), 404

    # cosine similarity via dot product (since emb is normalized)
    query_vec = emb[idx]
    sims = emb @ query_vec  # shape: (N,)

    # exclude itself
    sims[idx] = -1.0

    order = np.argsort(-sims)
    recs = []
    seen = set()

    for j in order:
        key = (meta[j]["track_name"], meta[j]["artist"])
        if key in seen:
            continue
        seen.add(key)

        recs.append(
            {
                "track_name": meta[j]["track_name"],
                "artist": meta[j]["artist"],
                "similarity": float(sims[j]),
            }
        )
        if len(recs) >= k:
            break

    return jsonify(
        {
            "query": {"track_name": meta[idx]["track_name"], "artist": meta[idx]["artist"]},
            "k": k,
            "recommendations": recs,
        }
    )