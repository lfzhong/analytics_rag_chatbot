"""
Final Formatting Chain

Takes the raw output from the reasoning chain and formats it according to 
specific presentation standards and business requirements.
"""

from langchain_ollama import OllamaLLM
from pathlib import Path
from config.settings import FINAL_FORMATTING_PROMPT_PATH


class FinalFormatter:
    """
    Final formatting chain that ensures consistent output format, 
    proper percentage conversion, and business-friendly presentation.
    """
    
    def __init__(self, model_name="llama3"):
        self.llm = OllamaLLM(model=model_name, temperature=0)
        
        # Load the final formatting prompt from file
        if not FINAL_FORMATTING_PROMPT_PATH.exists():
            raise FileNotFoundError(f"Final formatting prompt file not found: {FINAL_FORMATTING_PROMPT_PATH}")
        
        self.prompt_template = FINAL_FORMATTING_PROMPT_PATH.read_text()
    
    def run(self, raw_analysis: str, question: str) -> str:
        """
        Format the raw analysis output into clean, professional bullet points.
        
        Args:
            raw_analysis: Raw output from the reasoning chain
            question: Original user question for context
            
        Returns:
            Professionally formatted bullet points
        """
        prompt = self.prompt_template.replace("{{ raw_analysis }}", raw_analysis)
        prompt = prompt.replace("{{ question }}", question)
        
        formatted_output = self.llm.invoke(prompt)
        
        # Post-process to ensure clean formatting
        return self._post_process(formatted_output)
    
    def _post_process(self, output: str) -> str:
        """
        Apply final cleanup to ensure perfect formatting.
        """
        lines = output.strip().split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('Based on', 'I will', 'Please note', 'This response')):
                # Ensure bullet point format
                if line and not line.startswith('•'):
                    line = '• ' + line
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
