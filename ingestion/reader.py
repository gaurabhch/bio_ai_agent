# ingestion/reader.py — paragraph-based (matches actual KB structure)

import re
import uuid
import docx
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from config import FIELD_TYPES

load_dotenv()


# ── Existing Flat Data Objects ────────────────────────────────────

@dataclass
class QuestionObject:
    text:       str
    cluster_id: str
    category:   str
    topic:      str
    obj_type:   str = "question_variant"


@dataclass
class ContentChunkObject:
    text:              str
    cluster_id:        str
    category:          str
    topic:             str
    field_type:        str
    reference_sources: List[str] = field(default_factory=list)
    obj_type:          str = "content_chunk"


# ── Hierarchical Data Objects ─────────────────────────────────────

@dataclass
class L3Paragraph:
    text:             str
    cluster_id:       str
    category:         str
    topic:            str
    field_type:       str
    section_name:     str
    paragraph_index:  int
    reference_sources: List[str] = field(default_factory=list)
    obj_type:         str = "l3_paragraph"


@dataclass
class L2Section:
    section_name:  str
    text:          str
    cluster_id:    str
    category:      str
    topic:         str
    field_type:    str
    paragraphs:    List[L3Paragraph] = field(default_factory=list)
    obj_type:      str = "l2_section"


@dataclass
class HierarchicalDoc:
    cluster_id:        str
    category:          str
    topic:             str
    trigger_questions: List[str]
    reference_sources: List[str]
    sections:          List[L2Section] = field(default_factory=list)
    obj_type:          str = "l1_cluster"

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.text for s in self.sections)


# ── Paragraph Helpers ─────────────────────────────────────────────

_SENTENCE_STARTERS = {
    'the', 'it', 'a', 'an', 'in', 'when', 'since', 'as', 'these', 'this',
    'those', 'there', 'however', 'furthermore', 'although', 'while', 'based',
    'according', 'among', 'during', 'studies', 'research', 'data', 'results',
    'patients', 'women', 'thus', 'both', 'because', 'if', 'several', 'many',
    'some', 'most'
}

_SKIP_EXACT = {
    'Content', 'References', 'Table of Contents', 'PCOS',
    'Trigger Questions (Sample User Queries)'
}

def _is_bold(para) -> bool:
    runs = [r for r in para.runs if r.text.strip()]
    if not runs:
        return False
    return sum(1 for r in runs if r.bold) >= len(runs) * 0.6

def _is_section_header(para) -> bool:
    text = para.text.strip()
    if len(text) < 4 or len(text) > 110:
        return False
    if not _is_bold(para):
        return False
    if text in _SKIP_EXACT:
        return False
    if text[0].islower():
        return False
    if text.endswith('?') or ' * ' in text:
        return False
    if text.startswith(('-', '•', '*', '·', 'http')):
        return False
    if re.match(r'^\d+[\.:\)]\s', text):
        return False
    words = text.split()
    if len(words) < 2 or len(words) > 12:
        return False
    if words[0].lower() in _SENTENCE_STARTERS:
        return False
    last = words[-1].lower().rstrip('.,;')
    if last in {'and', 'the', 'of', 'in', 'a', 'an', 'is', 'are', 'with', 'for', 'to', 'or'}:
        return False
    return True

def _split_into_paragraphs(
    text: str,
    cluster_id: str,
    category: str,
    topic: str,
    field_type: str,
    section_name: str,
    reference_sources: List[str],
) -> List[L3Paragraph]:
    raw = re.split(r'(?<=[.!?])\s{2,}|\n{2,}', text)
    paragraphs = [p.strip() for p in raw if len(p.strip()) > 60]
    if not paragraphs:
        paragraphs = [text.strip()]

    result = []
    for idx, para_text in enumerate(paragraphs, start=1):
        if not para_text:
            continue
        result.append(L3Paragraph(
            text              = para_text,
            cluster_id        = cluster_id,
            category          = category,
            topic             = topic,
            field_type        = field_type,
            section_name      = section_name,
            paragraph_index   = idx,
            reference_sources = reference_sources,
        ))
    return result


# ── Cluster Boundary Finder ───────────────────────────────────────

def _find_cluster_boundaries(paras) -> List[Tuple[int, int]]:
    entry_pattern = re.compile(r'Entry ID[:\s]*(PCOS-\S+)', re.IGNORECASE)
    starts = [i for i, p in enumerate(paras)
              if p.text.strip() and entry_pattern.search(p.text)]
    boundaries = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(paras)
        boundaries.append((s, e))
    return boundaries


# ── Single Cluster Parser ─────────────────────────────────────────

def _parse_cluster(paras, start: int, end: int):
    entry_id, cluster_num, category, topic = "", "", "", ""
    trigger_questions = []
    references        = []
    in_trigger        = False
    content_start     = start

    for i in range(start, min(start + 60, end)):
        text = paras[i].text.strip()
        if not text:
            continue
        m = re.match(r'Entry ID[:\s]*(PCOS-\S+)', text, re.IGNORECASE)
        if m:
            entry_id = m.group(1)
            continue
        m = re.match(r'Cluster[:\s]*(Cluster\s*\d+)', text, re.IGNORECASE)
        if m:
            cluster_num = m.group(1)
            continue
        m = re.match(r'Category[:\s]*(.+)', text, re.IGNORECASE)
        if m:
            category = m.group(1).strip()
            continue
        if re.match(r'\d+\.\s+.+PCOS', text):
            topic = re.sub(r'^\d+\.\s+', '', text).strip()
            continue
        if 'Trigger Questions' in text or 'Sample User Queries' in text:
            in_trigger = True
            continue
        if text == 'Content':
            in_trigger    = False
            content_start = i + 1
            break
        if in_trigger and len(text) > 3:
            trigger_questions.append(text.strip(' *-|'))

    if not entry_id:
        return None, [], []

    # Find references section
    ref_start = end
    for i in range(end - 1, content_start, -1):
        if paras[i].text.strip() == 'References' and _is_bold(paras[i]):
            ref_start = i
            break

    for i in range(ref_start + 1, end):
        t = paras[i].text.strip()
        if t.startswith('http'):
            references.append(t)

    # Split content into L2 sections
    sections: List[Tuple[str, str]] = []
    current_name  = "Introduction"
    current_lines = []

    for i in range(content_start, ref_start):
        para = paras[i]
        text = para.text.strip()
        if not text or text in _SKIP_EXACT:
            continue
        if text.startswith('http') or 'alternative text' in text.lower():
            continue
        if _is_section_header(para):
            section_text = " ".join(current_lines).strip()
            if section_text:
                sections.append((current_name, section_text))
            current_name  = text
            current_lines = []
        else:
            current_lines.append(text)

    section_text = " ".join(current_lines).strip()
    if section_text:
        sections.append((current_name, section_text))

    # Build objects
    question_variants = [
        QuestionObject(text=q, cluster_id=entry_id,
                       category=category, topic=topic)
        for q in trigger_questions if q
    ]

    content_chunks = []
    h_sections     = []

    for field_type, text in sections:
        if not text.strip():
            continue

        # Flat ContentChunkObject
        content_chunks.append(ContentChunkObject(
            text              = text,
            cluster_id        = entry_id,
            category          = category,
            topic             = topic,
            field_type        = field_type,
            reference_sources = references,
        ))

        # L2 section + L3 paragraphs
        l3_paragraphs = _split_into_paragraphs(
            text              = text,
            cluster_id        = entry_id,
            category          = category,
            topic             = topic,
            field_type        = field_type,
            section_name      = field_type,
            reference_sources = references,
        )
        h_sections.append(L2Section(
            section_name = field_type,
            text         = text,
            cluster_id   = entry_id,
            category     = category,
            topic        = topic,
            field_type   = field_type,
            paragraphs   = l3_paragraphs,
        ))

    h_doc = HierarchicalDoc(
        cluster_id        = entry_id,
        category          = category,
        topic             = topic,
        trigger_questions = trigger_questions,
        reference_sources = references,
        sections          = h_sections,
    )

    return h_doc, question_variants, content_chunks


# ── Main Entry Point ──────────────────────────────────────────────

def read_knowledge_base(docx_path: str):
    """
    Reads PCOS KB DOCX using paragraph structure (not tables).
    Returns:
        question_variants : List[QuestionObject]
        content_chunks    : List[ContentChunkObject]
        hierarchical_docs : List[HierarchicalDoc]
    """
    doc   = docx.Document(docx_path)
    paras = doc.paragraphs

    print(f"[Reader] Total paragraphs found: {len(paras)}")

    boundaries = _find_cluster_boundaries(paras)
    print(f"[Reader] Clusters found: {len(boundaries)}")

    question_variants = []
    content_chunks    = []
    hierarchical_docs = []

    for start, end in boundaries:
        h_doc, q_list, c_list = _parse_cluster(paras, start, end)
        if h_doc:
            hierarchical_docs.append(h_doc)
            question_variants.extend(q_list)
            content_chunks.extend(c_list)

    return question_variants, content_chunks, hierarchical_docs