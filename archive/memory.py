class MemoryManager:
    def __init__(self, max_turns=5):
        self.chat_history = []
        self.max_turns = max_turns

    def add_turn(self, user_input: str, assistant_response: str):
        self.chat_history.append({"user": user_input, "assistant": assistant_response})
        if len(self.chat_history) > self.max_turns:
            self.chat_history.pop(0)

    def format_memory(self) -> str:
        formatted = ""
        for turn in self.chat_history:
            formatted += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        return formatted
