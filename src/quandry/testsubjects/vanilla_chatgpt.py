from quandry.classes import *
from quandry.classes import ExpectationCase
import keys as keys

import openai

class VanillaChatGPT(ISubject):
    def initialize():
        static=VanillaChatGPT
        if not hasattr(static, 'client'):
            static.client = openai.Client(api_key=keys.KEY_OpenAI)
    def respond(self, prompt: str) -> str:
        type(self).initialize()
        completion = type(self).client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt},
            ]
        )
        return completion.choices[0].message.content