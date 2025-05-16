import os
import json
import logging

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, memory_file="vpa_chat_memory.json"):
        self.memory_file = os.path.abspath(memory_file)
        self.system_messages = []
        self.memory = self._load_memory()
        logger.info(f"MemoryManager initialized with file: {self.memory_file}")

    def _load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r", encoding="utf-8") as f:
                logger.info(f"Loading memory from {self.memory_file}")
                return json.load(f)
        logger.info(f"No existing memory file found at {self.memory_file}. Starting with empty memory.")
        return []

    def save_message(self, role, content):
        self.memory.append({"role": role, "content": content})
        self._save_memory()
        if content:
            logger.debug(f"Saved message: role={role}, content={content[:50]}...")
        else:
            logger.debug(f"Saved empty message: role={role}")

    def _save_memory(self):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, indent=2)
        logger.info(f"Memory saved to {self.memory_file}")

    def get_history(self):
        return self.system_messages + self.memory[-10:]

    def add_system_message(self, content):
        self.system_messages.append({"role": "system", "content": content})
        logger.info(f"Added system message: {content[:50]}...")

    def clear(self):
        self.memory = []
        self._save_memory()
        logger.info("Memory cleared (except for system messages)")

    def __str__(self):
        return f"MemoryManager(file={self.memory_file}, messages={len(self.memory)})"