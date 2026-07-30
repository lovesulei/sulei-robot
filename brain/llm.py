from ollama import chat

from brain.personality import SYSTEM_PROMPT


class LLM:
    def __init__(self, model="gemma3:1b"):
        self.model = model

    def chat(self, message: str):
        response = chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": message,
                },
            ],
        )

        return response["message"]["content"]