from quandry.classes import *
from quandry.classes import ExpectationCase
import os
import openai
from quandry.utils import static_init
import asyncio
from concurrent.futures import ThreadPoolExecutor

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
    
    
    async def fetch_responses(self, prompts: Collection[str]) -> Collection[str]:
        loop = asyncio.get_event_loop()
        semaphore = asyncio.Semaphore(5)

        async def async_respond(prompt):
            async with semaphore:
                return await loop.run_in_executor(None, self.respond, prompt)

        with ThreadPoolExecutor() as executor:
            tasks = [async_respond(prompt) for prompt in prompts]
            responses = await asyncio.gather(*tasks)
            return responses
    
    def respond_batch(self, prompts):
        responses = asyncio.run(self.fetch_responses(prompts))
        return responses