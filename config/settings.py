# config/settings.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
METRICS_FILE = DATA_DIR / "metrics_data.xlsx"
INSIGHTS_FILE = DATA_DIR / "insights_data.xlsx"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TREND_SLOPE_TOLERANCE = 0.01

# Vector store
PERSIST_DIRECTORY = PROJECT_ROOT / "chroma_store"

# Prompt locations
PROMPTS_DIR = PROJECT_ROOT / "prompts"
CLARIFY_PROMPT_PATH = PROMPTS_DIR / "clarify_prompt.txt"
INTENT_PROMPT_PATH = PROMPTS_DIR / "intent_prompt.txt"
REASONING_PROMPT_PATH = PROMPTS_DIR / "reasoning_prompt.txt"
METADATA_FILTER_PROMPT_PATH = PROMPTS_DIR / "query_construct_prompt.txt"
FINAL_FORMATTING_PROMPT_PATH = PROMPTS_DIR / "final_formatting_prompt.txt"
