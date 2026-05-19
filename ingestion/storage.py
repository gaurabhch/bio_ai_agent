# ingestion/storage.py
# ─────────────────────────────────────────────────────────────────
# Indexes L3 paragraphs + embeddings in NeonDB using pgvector.
# Parent (L1/L2) data stored as metadata alongside each L3 row.
# ─────────────────────────────────────────────────────────────────

import os
import json
import hashlib
import psycopg2
from psycopg2.extras import execute_values
from typing import List
from dotenv import load_dotenv

from reader import HierarchicalDoc, L2Section, L3Paragraph
from config import NEON_TABLE_NAME, EMBEDDING_DIM

load_dotenv()

# ── Helpers ───────────────────────────────────────────────────────

def _get_conn():
    """Opens a fresh psycopg2 connection to NeonDB."""
    return psycopg2.connect(os.environ["DATABASE_URL"])

def _vec_to_str(vec: List[float]) -> str:
    """Convert Python list → pgvector string: '[0.1,0.2,...]' """
    return "[" + ",".join(str(v) for v in vec) + "]"

def _text_hash(text: str) -> str:
    """MD5 hash of text — used as chunk_id and embedding cache key."""
    return hashlib.md5(text.encode()).hexdigest()

# ── Table Setup ───────────────────────────────────────────────────

def setup_table(reset: bool = False):
    """
    Creates pgvector extension + pcos_kb_chunks table in NeonDB.
    Safe to call multiple times — uses IF NOT EXISTS.
    Set reset=True to drop and recreate the table.
    """
    conn = _get_conn()
    cur  = conn.cursor()

    # Enable pgvector
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    if reset:
        cur.execute(f"DROP TABLE IF EXISTS {NEON_TABLE_NAME};")
        print(f"[Storage] Dropped table: {NEON_TABLE_NAME}")

    # Create table
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {NEON_TABLE_NAME} (
            -- Primary key (MD5 hash of chunk text)
            chunk_id    TEXT PRIMARY KEY,

            -- L3 Grandchild — what gets searched
            text        TEXT NOT NULL,
            embedding   vector({EMBEDDING_DIM}),

            -- L3 metadata
            cluster_id          TEXT,
            cluster_num         TEXT,
            category            TEXT,          -- stores para.domain (e.g. "diagnosis")
            section_name        TEXT,
            token_count         INTEGER,
            paragraph_index     INTEGER,

            -- L2 Child parent — returned as context on match
            l2_id       TEXT,
            l2_text     TEXT,
            l2_tokens   INTEGER,

            -- L1 Parent cluster — returned on auto-merge
            l1_id                   TEXT,
            l1_text                 TEXT,
            l1_tokens               INTEGER,
            l1_trigger_questions    TEXT,      -- JSON array as string
            reference_sources       TEXT,      -- JSON array of URLs

            -- Timestamp
            created_at  TIMESTAMP DEFAULT NOW()
        );
    """)

    # HNSW index — works immediately, no training step needed
    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS {NEON_TABLE_NAME}_hnsw_idx
        ON {NEON_TABLE_NAME}
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)

    conn.commit()
    cur.close()
    conn.close()
    print(f"[Storage] ✅ Table '{NEON_TABLE_NAME}' ready in NeonDB (dim={EMBEDDING_DIM})")

# ── Main Index Builder ────────────────────────────────────────────

def build_index(
    l1_docs:    list,   # List[HierarchicalDoc]
    l2_sections: list,  # List[L2Section]
    l3_paras:   list,   # List[L3Paragraph]
    embeddings: dict,   # {text_hash: vector}
    reset:      bool = False,
):
    """
    Upserts all L3 paragraphs + embeddings into NeonDB pgvector table.
    L2 and L1 parent text stored inline as metadata for fast retrieval.
    Calls setup_table() first to ensure table exists.
    """
    # Always setup first — safe with IF NOT EXISTS
    setup_table(reset=reset)

    conn = _get_conn()
    cur  = conn.cursor()

    # ── Build lookup maps ─────────────────────────────────────────
    l1_map = {doc.cluster_id: doc for doc in l1_docs}
    l2_map = {}
    for sec in l2_sections:
        key = f"{sec.cluster_id}_{sec.field_type}"
        l2_map[key] = sec

    # ── Build rows ────────────────────────────────────────────────
    rows    = []
    skipped = 0

    for para in l3_paras:
        h = _text_hash(para.text)
        if h not in embeddings:
            skipped += 1
            continue

        l1 = l1_map.get(para.cluster_id)
        l2 = l2_map.get(f"{para.cluster_id}_{para.field_type}")

        # ── reference_sources: pulled from L1 cluster ────────────
        ref_sources = []
        if l1 and hasattr(l1, "reference_sources"):
            ref_sources = l1.reference_sources or []

        rows.append((
            h,                                          # chunk_id
            para.text,                                  # text
            _vec_to_str(embeddings[h]),                 # embedding
            para.cluster_id,                            # cluster_id
            "",                                         # cluster_num
            para.domain,                                # category  ← domain label e.g. "diagnosis"
            para.section_name,                          # section_name
            len(para.text.split()),                     # token_count
            para.paragraph_index,                       # paragraph_index
            f"{para.cluster_id}_{para.field_type}",    # l2_id
            (l2.text if l2 else ""),                    # l2_text
            (len(l2.text.split()) if l2 else 0),        # l2_tokens
            para.cluster_id,                            # l1_id
            (l1.full_text[:4000] if l1 else ""),        # l1_text (capped at 4000 chars)
            (len(l1.full_text.split()) if l1 else 0),   # l1_tokens
            json.dumps(l1.trigger_questions if l1 else []),  # l1_trigger_questions
            json.dumps(ref_sources),                    # reference_sources ← FIXED (was commented out)
        ))

    if skipped:
        print(f"[Storage] ⚠️  Skipped {skipped} paragraphs with no embedding")

    if not rows:
        print("[Storage] ⚠️  No rows to insert — check reader/chunker output")
        return

    # ── Upsert in batches ─────────────────────────────────────────
    upsert_sql = f"""
        INSERT INTO {NEON_TABLE_NAME} (
            chunk_id, text, embedding,
            cluster_id, cluster_num, category, section_name,
            token_count, paragraph_index,
            l2_id, l2_text, l2_tokens,
            l1_id, l1_text, l1_tokens,
            l1_trigger_questions,
            reference_sources 
        )
        VALUES %s
        ON CONFLICT (chunk_id) DO UPDATE SET
            embedding            = EXCLUDED.embedding,
            text                 = EXCLUDED.text,
            category             = EXCLUDED.category,
            l2_text              = EXCLUDED.l2_text,
            l2_tokens            = EXCLUDED.l2_tokens,
            l1_text              = EXCLUDED.l1_text,
            l1_tokens            = EXCLUDED.l1_tokens,
            l1_trigger_questions = EXCLUDED.l1_trigger_questions,
            reference_sources    = EXCLUDED.reference_sources;
    """

    batch_size = 100
    total      = len(rows)

    for i in range(0, total, batch_size):
        batch = rows[i : i + batch_size]
        execute_values(
            cur,
            upsert_sql,
            batch,
            template  = "(%s, %s, %s::vector, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            page_size = batch_size,
        )
        conn.commit()
        done = min(i + batch_size, total)
        print(f"  [{done:4d}/{total}] chunks indexed...")

    # ── Final count ───────────────────────────────────────────────
    cur.execute(f"SELECT COUNT(*) FROM {NEON_TABLE_NAME};")
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    print(f"\n[Storage] ✅ {count} L3 vectors stored in NeonDB → '{NEON_TABLE_NAME}'")
