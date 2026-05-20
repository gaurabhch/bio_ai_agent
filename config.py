# config.py
import os
from dotenv import load_dotenv
from urllib.parse import urlparse, urlunparse
load_dotenv(override=True)  

# ── Model ─────────────────────────────────────────────────────────────────────
# LOCKED: Pipeline 1 embeds KB with this model.
# Pipeline 2 MUST use the exact same model to embed user queries.
# If you change this, re-embed every row in medical_knowledge first.
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
EMBEDDING_MODEL_FAST = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
BATCH_SIZE  = 64

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL       = os.environ["DATABASE_URL"]

def _make_async_url(url: str) -> str:
    parsed = urlparse(url)
    # Force scheme to exactly "postgresql+asyncpg" regardless of what it was
    clean = parsed._replace(
        scheme = "postgresql+asyncpg",   
        query  = ""
    )
    return urlunparse(clean)



DATABASE_URL_ASYNC = _make_async_url(DATABASE_URL)

REDIS_URL        = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CONVERSATION_TTL = 86_400          # 24 hours in seconds
MAX_HISTORY_MSGS = 10              # messages kept in Redis per conversation
NEON_TABLE_NAME = 'pcos_kb_chunks'

# ── Knowledge base ────────────────────────────────────────────────────────────
KB_PATH = "PCOS_Knowledge_Base_v1.docx"
MAX_FIELD_WORDS       = 250              # chunker splits fields longer than this
TOP_K_RETRIEVAL       = 5                # chunks returned per PGVector search
KEYWORD_ROUTING_THRESHOLD = 2            # score above this → skip LLM routing
AUTO_MERGE_RATIO     = 2                 # merge to L2 if ≥2 L3 hits in same section
FINAL_CONTEXT_TOKENS = 2000              # max tokens passed to LLM

# ── Domain mapping ────────────────────────────────────────────────────────────
CATEGORY_TO_DOMAIN = {
    # Foundation / General
    'Definition':                               'pcos_general',
    'Foundation':                               'pcos_general',

    # Diagnosis
    'Diagnosis':                                'pcos_diagnosis',
    'Clinical Investigations':                  'pcos_diagnosis',

    # Metabolic / Nutrition
    'Metabolic':                                'pcos_metabolic',
    'Metabolic / Nutrition':                    'pcos_metabolic',
    'Diet Approaches':                          'pcos_nutrition',
    'Nutrition':                                'pcos_nutrition',

    # Mental Health
    'Mental Health':                            'pcos_mental_health',
    'Mental Support':                           'pcos_mental_health',

    # Fertility
    'Fertility':                                'pcos_fertility',
    'Fertility Basics':                         'pcos_fertility',
    'Ovulation with PCOS':                      'pcos_fertility',
    'Pregnancy & Fertility with PCOS':          'pcos_fertility',

    # Treatment / Pharma
    'Treatment':                                'pcos_treatment',
    'Contraceptive Treatment':                  'pcos_treatment',
    'Metformin and Its Working':                'pcos_treatment',
    'Supplements':                              'pcos_treatment',
    'Androgen Therapy':                         'pcos_treatment',

    # Lifestyle
    'Lifestyle':                                'pcos_lifestyle',
    'Lifestyle & Symptom Management':           'pcos_lifestyle',
    'Lifestyle Management: Physical Activity':  'pcos_lifestyle',
    'Lifestyle Management: Daily Routine and Sleep': 'pcos_lifestyle',
    'Stress and Cortisol Management':           'pcos_lifestyle',
    'Weight Management':                        'pcos_lifestyle',

    # Symptoms
    'Symptoms':                                 'pcos_symptoms',
    'Physical Symptoms in PCOS':                'pcos_symptoms',

    # Skin & Hair
    'Skin & Hair':                              'pcos_skin_hair',
    'Skin Management':                          'pcos_skin_hair',
    'Hair Loss Management':                     'pcos_skin_hair',

    # Complementary
    'Complementary':                            'pcos_complementary',

    # Menstrual
    'Menstrual Health':                         'pcos_menstrual',
    'Menstruation':                             'pcos_menstrual',
    'Menstruation & PCOS':                      'pcos_menstrual',

    # Hormonal
    'Hormones':                                 'pcos_hormonal',
    'Hormones & PCOS':                          'pcos_hormonal',
}

# ── Domain keyword scores for routing ─────────────────────────────────────────
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "pcos_metabolic":      ["insulin", "weight", "metabolism", "glucose", "sugar", "resistance"],
    "pcos_mental_health":  ["anxiety", "depression", "stress", "mood", "mental", "emotional", "scared", "worried"],
    "pcos_fertility":      ["pregnancy", "conceive", "fertility", "ovulation", "trying to get pregnant"],
    "pcos_nutrition":      ["diet", "food", "eat", "nutrition", "meal", "supplement", "vitamin"],
    "pcos_treatment":      ["medication", "metformin", "pill", "treatment", "medicine", "dose"],
    "pcos_lifestyle":      ["exercise", "sleep", "lifestyle", "workout", "stress management"],
    "pcos_symptoms":       ["bloating", "fatigue", "acne", "hair loss", "cramp", "pain", "symptom"],
    "pcos_diagnosis":      ["diagnosed", "test", "ultrasound", "blood test", "scan", "diagnosis"],
    "pcos_skin_hair":      ["skin", "hair", "acne", "hirsutism", "thinning", "oily"],
    "pcos_menstrual":      ["period", "cycle", "menstrual", "irregular", "spotting", "flow"],
    "pcos_hormonal":       ["hormone", "testosterone", "estrogen", "cortisol", "lh", "fsh", "prolactin"],
    "pcos_general":        ["pcos", "pcod", "polycystic", "ovary", "syndrome"],
}

# ── Response modes ────────────────────────────────────────────────────────────
RESPONSE_MODES = {
    "INFORMATION":    "information",
    "EMOTIONAL":      "emotional",
    "CLARIFICATION":  "clarification",
    "CRISIS":         "crisis",
}

EMOTIONAL_KEYWORDS = [
    "scared", "anxious", "worried", "depressed", "crying", "hopeless",
    "overwhelmed", "lost", "don't know what to do", "help me",
    "frustrated", "sad", "upset",
]

VAGUE_PATTERNS = [
    "i don't feel like myself",
    "something is wrong",
    "i feel weird",
    "i don't know",
    "not sure",
]

# ── Crisis detection ──────────────────────────────────────────────────────────
CRISIS_KEYWORDS: list[str] = [
    "suicide", "suicidal", "kill myself", "end my life", "want to die",
    "self harm", "self-harm", "cut myself", "hurt myself", "not worth living",
    "abuse", "being abused", "domestic violence", "overdose",
]

FALSE_POSITIVE_GUARD: list[str] = [
    "killing it", "dying of laughter", "headache",
]

HELPLINE_RESPONSE = (
    "I can hear that you are going through something very difficult. "
    "Please reach out to someone who can help:\n\n"
    "• iCall: 9152987821\n"
    "• Vandrevala Foundation: 1860-2662-345\n\n"
    "You deserve support and you are not alone."
)

# ── Output field types (from KB Row 4) ────────────────────────────────────────
CLUSTER_TABLE_START = 0
FIELD_TYPES: list[str] = [
    "summary",
    "explanation",
    "actionable",
    "symptoms_it_explains",
    "who_it_affects",
    "red_flags",
]

# ── Streaming ─────────────────────────────────────────────────────────────────
STREAM_WORD_DELAY = 0.03           # seconds between streamed words

# ── Timeouts ─────────────────────────────────────────────────────────────────
GEMINI_TIMEOUT  = 10.0             # seconds before returning fallback message
PUBMED_TIMEOUT  = 5.0              # seconds before skipping PubMed verification
GROQ_TIMEOUT = 10.0


GROQ_TIMEOUT_MESSAGE = (
    "I am having trouble connecting right now. Please try again in a moment."
)


