# ingestion/pipeline_task.py
# ─────────────────────────────────────────────────────────────────
# Celery task that orchestrates the full ingestion pipeline:
#
#   Step 1 → reader.py       : Parse DOCX → flat + hierarchical objects
#   Step 2 → chunker.py      : Split oversized chunks
#   Step 3 → tagger.py       : Attach domain labels
#   Step 4 → embedder.py     : Generate embeddings (all-MiniLM-L6-v2)
#   Step 5 → storage.py      : Upsert into NeonDB pgvector
#
# Can be triggered:
#   - As a Celery task  : run_ingestion.delay()
#   - Directly (dev)    : run_ingestion_pipeline() — for local testing
# ─────────────────────────────────────────────────────────────────

import os
import time
from dotenv import load_dotenv
from celery import Celery

from reader import read_knowledge_base
from chunker  import prepare_chunks, prepare_h_chunks
from tagger   import tag_all, tag_all_h
from embedder import embed_l3_chunks
from storage  import build_index
from config import KB_PATH, REDIS_URL

load_dotenv()

# ── Celery App ────────────────────────────────────────────────────
app = Celery(
    "ingestion",
    broker=REDIS_URL,       # e.g. "redis://localhost:6379/0"
)

app.conf.update(
    task_serializer   = "json",
    result_serializer = "json",
    accept_content    = ["json"],
    timezone          = "Asia/Kolkata",
)


# ── Core Pipeline Logic ───────────────────────────────────────────

def run_ingestion_pipeline(
    kb_path:     str  = None,
    reset_index: bool = False,
    verbose:     bool = True,
) -> dict:
    """
    Runs the full 5-step ingestion pipeline.
    Called directly for local dev/testing OR wrapped by Celery task.

    Returns a summary dict with chunk counts and timing.
    """
    kb_path = KB_PATH
    summary = {}
    t_start = time.time()

    # ── Step 1: Parse Knowledge Base ─────────────────────────────
    if verbose:
        print("\n=== STEP 1: Parsing Knowledge Base ===")

    question_variants, content_chunks, hierarchical_docs = read_knowledge_base(KB_PATH)

    if verbose:
        print(f"  Clusters      : {len(hierarchical_docs)}")
        print(f"  Questions     : {len(question_variants)}")
        print(f"  Flat chunks   : {len(content_chunks)}")

    summary["clusters"]       = len(hierarchical_docs)
    summary["questions"]      = len(question_variants)
    summary["flat_chunks"]    = len(content_chunks)

    # ── Step 2: Chunk ─────────────────────────────────────────────
    if verbose:
        print("\n=== STEP 2: Chunking ===")

    # 2a — Flat chunks (existing pipeline)
    questions, flat_chunks = prepare_chunks(question_variants, content_chunks)

    # 2b — Hierarchical chunks (L1 / L2 / L3)
    l1_docs, l2_sections, l3_paras = prepare_h_chunks(hierarchical_docs)

    if verbose:
        print(f"  Flat   chunks : {len(flat_chunks)}")
        print(f"  L1 clusters   : {len(l1_docs)}")
        print(f"  L2 sections   : {len(l2_sections)}")
        print(f"  L3 paragraphs : {len(l3_paras)}  ← these get embedded")

    summary["l1_docs"]     = len(l1_docs)
    summary["l2_sections"] = len(l2_sections)
    summary["l3_paras"]    = len(l3_paras)

    # ── Step 3: Tag Domains ───────────────────────────────────────
    if verbose:
        print("\n=== STEP 3: Tagging Domains ===")

    # 3a — Flat objects
    questions, flat_chunks = tag_all(questions, flat_chunks)

    # 3b — Hierarchical objects
    l1_docs, l2_sections, l3_paras = tag_all_h(l1_docs, l2_sections, l3_paras)

    if verbose:
        # Show domain distribution across L3 chunks
        from collections import Counter
        domain_counts = Counter(p.domain for p in l3_paras)
        for domain, count in sorted(domain_counts.items()):
            print(f"  {domain:25s} : {count:3d} L3 chunks")

    summary["domain_distribution"] = dict(
        Counter(p.domain for p in l3_paras)
    ) if verbose else {}

    # ── Step 4: Generate Embeddings ───────────────────────────────
    if verbose:
        print("\n=== STEP 4: Generating Embeddings ===")

    embeddings = embed_l3_chunks(l3_paras)   # returns {para_text_hash: vector}

    if verbose:
        print(f"  Embedded      : {len(embeddings)} L3 paragraphs")

    summary["embeddings"] = len(embeddings)

    # ── Step 5: Store in NeonDB ───────────────────────────────────
    if verbose:
        print("\n=== STEP 5: Storing in NeonDB (pgvector) ===")

    build_index(
        l1_docs      = l1_docs,
        l2_sections  = l2_sections,
        l3_paras     = l3_paras,
        embeddings   = embeddings,
        reset        = reset_index,
    )

    # ── Done ──────────────────────────────────────────────────────
    elapsed = round(time.time() - t_start, 2)
    summary["elapsed_seconds"] = elapsed

    if verbose:
        print(f"\n✅ Ingestion complete in {elapsed}s")
        print(f"   {len(l3_paras)} L3 vectors indexed in NeonDB\n")

    return summary


# ── Celery Task Wrapper ───────────────────────────────────────────

@app.task(
    name    = "ingestion.run_ingestion",
    bind    = True,
    max_retries = 3,
    default_retry_delay = 60,   # retry after 60s on failure
)
def run_ingestion(self, kb_path: str = None, reset_index: bool = False):
    """
    Celery-wrapped ingestion task.
    Triggered by: run_ingestion.delay()  or  run_ingestion.apply_async()
    """
    try:
        summary = run_ingestion_pipeline(
            kb_path     = kb_path,
            reset_index = reset_index,
            verbose     = True,
        )
        return summary

    except Exception as exc:
        # Retry on transient failures (DB connection, embedding timeout)
        raise self.retry(exc=exc)


# ── Direct Run (dev/testing — replaces demo.py) ──────────────────

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("Set GROQ_API_KEY in your .env file!")
    if not os.getenv("DATABASE_URL"):
        raise ValueError("Set DATABASE_URL (NeonDB connection string) in .env!")

    # Set reset_index=True to wipe and rebuild the NeonDB table
    run_ingestion_pipeline(reset_index=False, verbose=True)