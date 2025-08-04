from langchain_ollama import OllamaLLM
from config.settings import REASONING_PROMPT_PATH

class AnswerGenerator:
    def __init__(self, model_name="llama3"):
        self.llm = OllamaLLM(model=model_name)
        if not REASONING_PROMPT_PATH.exists():
            raise FileNotFoundError(f"Prompt file not found: {REASONING_PROMPT_PATH}")
        self.prompt_template = REASONING_PROMPT_PATH.read_text()

    def run(self, question: str, context: str) -> str:
        prompt = self.prompt_template.replace("{{ retrieved_docs }}", context).replace("{{ question }}", question)
        return self.llm.invoke(prompt)
