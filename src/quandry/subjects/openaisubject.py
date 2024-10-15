from quandry.classes import *
from quandry.classes import ExpectationCase
import os
import openai
from quandry.utils import static_init

ENV_OPENAI_API_KEY = "KEY_OPENAI"

@static_init
class OpenAiSubject(ISubject):
    @classmethod
    def static_init(static):
        static.client = openai.Client(api_key=os.environ[ENV_OPENAI_API_KEY])
            
    def __init__(self, model:str="gpt-4o-mini"):
        """
        Initializes the OpenAiSubject instance with the specified model.
        Args:
            model (str): The model identifier to be used. Default is "gpt-4o-mini".
        """
        
        self.model = model
            
    def respond(self, prompt: str) -> str:
        completion = type(self).client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ]
        )
        return completion.choices[0].message.content