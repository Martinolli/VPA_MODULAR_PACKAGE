import os
import json

class MemoryManager:
    def __init__(self, memory_file="vpa_chat_memory.json"):
        self.memory_file = memory_file
        self.memory = self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def save_message(self, role, content):
        self.memory.append({"role": role, "content": content})
        self._save_memory()

    def _save_memory(self):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, indent=2)

    def get_history(self):
        return self.memory[-10:]  # Return last 10 messages (you can adjust this)

    def clear(self):
        self.memory = []
        self._save_memory()
