# ingestion/tagger.py
# ─────────────────────────────────────────────────────────────────
# Maps category → domain label for all chunk types.
# Uses CATEGORY_TO_DOMAIN from config.
#
# Two nodes:
#   tag_all()    → existing flat pipeline (unchanged)
#   tag_all_h()  → new hierarchical pipeline (L1/L2/L3)
# ─────────────────────────────────────────────────────────────────

from config import CATEGORY_TO_DOMAIN
from reader import HierarchicalDoc, L2Section, L3Paragraph
from typing import List, Tuple


# ── Existing: Single Object Tagger (unchanged) ───────────────────

def tag_domain(obj):
    """
    Attaches a .domain attribute to any chunk object based on its category.
    Falls back to 'pcos_general' if category not found in config map.
    """
    obj.domain = CATEGORY_TO_DOMAIN.get(obj.category, "pcos_general")
    return obj


# ── Existing: Flat Tag All (unchanged) ───────────────────────────

def tag_all(question_variants: list, content_chunks: list):
    """
    Tags domain on flat QuestionObjects and ContentChunkObjects.
    Unchanged from original — existing pipeline keeps working.
    """
    return (
        [tag_domain(q) for q in question_variants],
        [tag_domain(c) for c in content_chunks],
    )


# ── New: Hierarchical Tag All ─────────────────────────────────────

def tag_all_h(
    l1_docs:     List[HierarchicalDoc],
    l2_sections: List[L2Section],
    l3_paras:    List[L3Paragraph],
) -> Tuple[List[HierarchicalDoc], List[L2Section], List[L3Paragraph]]:
    """
    Tags domain on all three hierarchy levels.
    Mutates objects in-place (same pattern as tag_domain).

    Returns the same three lists with .domain attached to every object.
    All three levels get the same domain since they share the same category.
    """
    for doc in l1_docs:
        tag_domain(doc)

    for section in l2_sections:
        tag_domain(section)

    for para in l3_paras:
        tag_domain(para)

    return l1_docs, l2_sections, l3_paras