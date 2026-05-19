# ingestion/chunker.py
# ─────────────────────────────────────────────────────────────────
# Prepares final chunk lists from reader output.
# Handles two modes:
#   1. Flat chunks  → prepare_chunks()       (existing, unchanged)
#   2. Hierarchical → prepare_h_chunks()     (NEW — for hierarchical RAG)
# ─────────────────────────────────────────────────────────────────

import re
from typing import List, Tuple
from reader import (
    ContentChunkObject,
    HierarchicalDoc,
    L2Section,
    L3Paragraph,
)
from config import MAX_FIELD_WORDS


# ── Existing: Flat Chunking (unchanged) ──────────────────────────

def split_at_sentence(text: str) -> List[str]:
    """Splits a long text block into two halves at sentence boundary."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    mid   = len(sentences) // 2
    part1 = ' '.join(sentences[:mid]).strip()
    part2 = ' '.join(sentences[mid:]).strip()
    return [part1, part2]


def prepare_chunks(
    question_variants: list,
    content_chunks: list,
) -> Tuple[list, list]:
    """
    Existing flat chunking — splits oversized ContentChunkObjects
    by sentence midpoint. Returns (questions, content_chunks).
    Unchanged from original.
    """
    final_questions = question_variants
    final_content   = []

    for chunk in content_chunks:
        word_count = len(chunk.text.split())
        if word_count > MAX_FIELD_WORDS:
            parts = split_at_sentence(chunk.text)
            for i, part in enumerate(parts):
                new_chunk = ContentChunkObject(
                    text              = part,
                    cluster_id        = chunk.cluster_id,
                    category          = chunk.category,
                    topic             = chunk.topic,
                    field_type        = f"{chunk.field_type}_part{i + 1}",
                    reference_sources = chunk.reference_sources,
                )
                final_content.append(new_chunk)
        else:
            final_content.append(chunk)

    return final_questions, final_content


# ── New: Hierarchical Chunking ────────────────────────────────────

def _split_long_paragraph(text: str, max_words: int) -> List[str]:
    """
    Splits an oversized L3 paragraph into smaller pieces
    at sentence boundaries without cutting mid-sentence.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    parts     = []
    current   = []
    current_wc = 0

    for sent in sentences:
        wc = len(sent.split())
        if current_wc + wc > max_words and current:
            parts.append(' '.join(current).strip())
            current    = [sent]
            current_wc = wc
        else:
            current.append(sent)
            current_wc += wc

    if current:
        parts.append(' '.join(current).strip())

    return [p for p in parts if p.strip()]


def prepare_h_chunks(
    hierarchical_docs: List[HierarchicalDoc],
    max_l3_words: int = None,
) -> Tuple[List[HierarchicalDoc], List[L2Section], List[L3Paragraph]]:
    """
    Prepares final hierarchical chunk lists from HierarchicalDocs.

    - Splits oversized L3 paragraphs at sentence boundaries
    - Rebuilds parent L2 and L1 references cleanly
    - Returns flat lists of all three levels for easy downstream use

    Returns:
        l1_docs   : List[HierarchicalDoc]  — 25 cluster-level parents
        l2_sections: List[L2Section]       — all sub-section children
        l3_paras  : List[L3Paragraph]      — all paragraphs (embedded)
    """
    if max_l3_words is None:
        max_l3_words = MAX_FIELD_WORDS      # reuse same config constant

    l1_docs:    List[HierarchicalDoc] = []
    l2_sections:List[L2Section]       = []
    l3_paras:   List[L3Paragraph]     = []

    for doc in hierarchical_docs:
        processed_doc = HierarchicalDoc(
            cluster_id        = doc.cluster_id,
            category          = doc.category,
            topic             = doc.topic,
            trigger_questions = doc.trigger_questions,
            reference_sources = doc.reference_sources,
        )

        for section in doc.sections:
            processed_section = L2Section(
                section_name = section.section_name,
                text         = section.text,
                cluster_id   = section.cluster_id,
                category     = section.category,
                topic        = section.topic,
                field_type   = section.field_type,
            )

            para_counter = 1
            for para in section.paragraphs:
                word_count = len(para.text.split())

                if word_count > max_l3_words:
                    # Split oversized paragraph at sentence boundaries
                    sub_texts = _split_long_paragraph(para.text, max_l3_words)
                else:
                    sub_texts = [para.text]

                for sub_text in sub_texts:
                    if not sub_text.strip():
                        continue
                    new_para = L3Paragraph(
                        text             = sub_text.strip(),
                        cluster_id       = para.cluster_id,
                        category         = para.category,
                        topic            = para.topic,
                        field_type       = para.field_type,
                        section_name     = para.section_name,
                        paragraph_index  = para_counter,
                        reference_sources= para.reference_sources,
                    )
                    processed_section.paragraphs.append(new_para)
                    l3_paras.append(new_para)
                    para_counter += 1

            processed_doc.sections.append(processed_section)
            l2_sections.append(processed_section)

        l1_docs.append(processed_doc)

    print(f"[Chunker] Hierarchical: {len(l1_docs)} L1 | "
          f"{len(l2_sections)} L2 | {len(l3_paras)} L3 chunks")

    return l1_docs, l2_sections, l3_paras