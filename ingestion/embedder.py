# ingestion/embedder.py
# ─────────────────────────────────────────────────────────────────
# Generates embeddings for all L3 paragraphs using all-MiniLM-L6-v2
# Runs fully locally — no API key needed after first model download.
# Uses file cache so re-runs are instant.
# ─────────────────────────────────────────────────────────────────

import json
import hashlib
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

# ✅ KEEP — loads model once at import, reused for all batches
_model = SentenceTransformer(EMBEDDING_MODEL)   # "all-MiniLM-L6-v2"

CACHE_PATH = Path("embedding_cache.json")


# ✅ KEEP — cache helpers unchanged
def _load_cache() -> Dict[str, List[float]]:
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}

def _save_cache(cache: Dict[str, List[float]]):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)

def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


# 🔄 REPLACE — updated to accept List[L3Paragraph] from pipeline_task.py
def embed_l3_chunks(l3_paras: list) -> Dict[str, List[float]]:
    """
    Accepts List[L3Paragraph] from chunker/pipeline_task.
    Returns {text_hash: vector} — storage.py looks up by text hash.
    Cache ensures re-runs are instant.
    """
    cache      = _load_cache()
    embeddings : Dict[str, List[float]] = {}
    to_embed   : List[tuple] = []   # (text_hash, text)

    for para in l3_paras:
        h = _text_hash(para.text)
        if h in cache:
            embeddings[h] = cache[h]
        else:
            to_embed.append((h, para.text))

    print(f"[Embedder] {len(embeddings)} from cache, {len(to_embed)} to embed")

    if to_embed:
        for i in range(0, len(to_embed), 128):
            batch = to_embed[i : i + 128]
            texts = [item[1] for item in batch]
            print(f"  Embedding batch {i // 128 + 1} ({len(texts)} chunks)...")

            vecs = _model.encode(texts, batch_size=128, show_progress_bar=False)

            for j, (h, _) in enumerate(batch):
                vec = vecs[j].tolist()
                embeddings[h] = vec
                cache[h]      = vec

        _save_cache(cache)

    print(f"[Embedder] ✅ Done — {len(embeddings)} embeddings (dim=384)")
    return embeddings   # {text_hash: vector}