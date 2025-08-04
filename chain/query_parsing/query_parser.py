from typing import List
from pathlib import Path
import json

from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from langchain.schema import OutputParserException
from pydantic import RootModel

from config.settings import METADATA_FILTER_PROMPT_PATH
from schemas.schema import QueryObject
from json_utils import extract_json_from_text


class QueryObjectList(RootModel[List[QueryObject]]):
    """
    Pydantic root model for a list of QueryObject.
    Used for robust LLM output parsing and validation.
    """
    pass


class QueryParser:
    """
    Parses LLM output into a validated list of QueryObject using a Pydantic root model.
    - Uses QueryObjectList for output parsing (required by LangChain/Pydantic v2+).
    - The run() method returns a plain list of QueryObject for downstream use.
    """
    def __init__(self, model_name="llama3", debug: bool = False):
        if not isinstance(METADATA_FILTER_PROMPT_PATH, Path) or not METADATA_FILTER_PROMPT_PATH.exists():
            raise FileNotFoundError(f"Prompt file not found: {METADATA_FILTER_PROMPT_PATH}")

        self.debug = debug
        self.llm = ChatOllama(
            model=model_name,
            temperature=0,
            format="json",
            system=(
                "You are a strict JSON generator. "
                "Your ONLY output must be a valid JSON ARRAY. "
                "EVEN IF ONLY ONE OBJECT IS PRESENT. NEVER RETURN A SINGLE DICTIONARY."
                "Do not include any extra text, comments, or apologies. "
                "Ensure the output can be parsed by a standard JSON parser (e.g., Python's json.loads). "
                "Return [] if nothing valid can be extracted."
            )
        )

        self.prompt = PromptTemplate.from_template(METADATA_FILTER_PROMPT_PATH.read_text())
        base_parser = PydanticOutputParser(pydantic_object=QueryObjectList)
        self.parser = OutputFixingParser.from_llm(parser=base_parser, llm=self.llm)
        self._format_instructions = self.parser.get_format_instructions()

    def _debug_log(self, message):
        if self.debug:
            print("[DEBUG]", message)

    @property
    def format_instructions(self):
        return self._format_instructions

    def run(self, input_dict: dict) -> list:
        """
        Run the query parsing chain and return a validated list of QueryObject.
        """
        # Get prompt string
        prompt_str = self.prompt.format(**input_dict)
        # Get LLM output as string
        llm_output = self.llm.invoke(prompt_str)
        # print("[DEBUG] Raw LLM output:", llm_output)
        if hasattr(llm_output, "content"):
            llm_output_str = llm_output.content
        else:
            llm_output_str = llm_output
        parsed = json.loads(llm_output_str)
        # Handle case where LLM returns a dict with a single key whose value is the list
        if isinstance(parsed, dict):
            if len(parsed) == 1 and isinstance(next(iter(parsed.values())), list):
                parsed = next(iter(parsed.values()))
                print("[DEBUG] Extracted list from single-key dict:", parsed)
            else:
                parsed = [parsed]
                print("[DEBUG] Wrapped single object in list for consistency:", parsed)
        
        # Normalize field names to singular form (fix common LLM mistakes)
        normalized_parsed = []
        for obj in parsed:
            if isinstance(obj, dict):
                normalized_obj = {}
                # Handle plural -> singular field name conversion
                if 'products' in obj and 'product' not in obj:
                    normalized_obj['product'] = obj['products'][0] if obj['products'] else ""
                else:
                    normalized_obj['product'] = obj.get('product', "").strip()
                
                if 'metrics' in obj and 'metric' not in obj:
                    normalized_obj['metric'] = obj['metrics'][0] if obj['metrics'] else ""
                else:
                    normalized_obj['metric'] = obj.get('metric', "").strip()
                
                # Copy other fields as-is but trim strings
                normalized_obj['time_spans'] = obj.get('time_spans', [])
                question_type = obj.get('question_type', 'overall_summary')
                if isinstance(question_type, str):
                    question_type = question_type.strip()
                normalized_obj['question_type'] = question_type
                
                normalized_parsed.append(normalized_obj)
            else:
                normalized_parsed.append(obj)
        
        return QueryObjectList.model_validate(normalized_parsed).root
