from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from config.settings import INTENT_PROMPT_PATH
from pathlib import Path

class IntentClassifier:
    def __init__(self, model_name="llama3"):
        if not isinstance(INTENT_PROMPT_PATH, Path) or not INTENT_PROMPT_PATH.exists():
            raise FileNotFoundError(f"Prompt file not found: {INTENT_PROMPT_PATH}")
        self.llm = ChatOllama(model=model_name, temperature=0, system="Only return valid JSON. Do not explain or comment. No text outside the JSON block.")
        self.prompt = PromptTemplate.from_template(INTENT_PROMPT_PATH.read_text())
        self.parser = JsonOutputParser()

    def run(self, input_text: str):
        chain = self.prompt | self.llm | self.parser
        return chain.invoke(input_text)
