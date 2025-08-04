from typing import Literal
from pydantic import create_model
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from config.settings import METADATA_FILTER_PROMPT_PATH
from chain.schema import TimeSpan
from pathlib import Path

# --- Define Pydantic Model ---

PartialParsedQuestion = create_model(
    "PartialParsedQuestion",
    question_type=(Literal["overall_summary", "metric_summary"], "overall_summary"),
    products=(list[str], []),
    metrics=(list[str], []),
    time_spans=(list[TimeSpan], [])
)

# --- Load Prompt ---
metadata_prompt = PromptTemplate.from_template(
    METADATA_FILTER_PROMPT_PATH.read_text()
)


# --- LLM Configuration ---
llm = ChatOllama(
    model="llama3",
    temperature=0,
    system="Only return valid JSON. Do not explain or comment. No text outside the JSON block."
)

# --- Output Parser ---
base_parser = PydanticOutputParser(pydantic_object=PartialParsedQuestion)
parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
format_instructions = parser.get_format_instructions()

class QueryConstructChain:
    def __init__(self, model_name="llama3"):
        if not isinstance(METADATA_FILTER_PROMPT_PATH, Path) or not METADATA_FILTER_PROMPT_PATH.exists():
            raise FileNotFoundError(f"Prompt file not found: {METADATA_FILTER_PROMPT_PATH}")
        self.llm = ChatOllama(model=model_name, temperature=0, system="Only return valid JSON. Do not explain or comment. No text outside the JSON block.")
        self.prompt = PromptTemplate.from_template(METADATA_FILTER_PROMPT_PATH.read_text())
        base_parser = PydanticOutputParser(pydantic_object=PartialParsedQuestion)
        self.parser = OutputFixingParser.from_llm(parser=base_parser, llm=self.llm)
        self._format_instructions = self.parser.get_format_instructions()

    @property
    def format_instructions(self):
        return self._format_instructions

    def run(self, input_dict: dict):
        chain = self.prompt | self.llm | self.parser
        return chain.invoke(input_dict)
