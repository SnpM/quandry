from quandry.classes import *
from quandry.classes import ExpectationCase
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai
from quandry.utils import static_init

# Replace with the appropriate environment variable for the Google Gemini API key
ENV_GEMINI_API_KEY = "KEY_GEMINIAPI"

@static_init
class GeminiSubject(ISubject):
    @classmethod
    def static_init(static):
        # Initialize the Google Gemini client here, assuming a similar interface to OpenAI
        gemini_key = os.environ[ENV_GEMINI_API_KEY]
        genai.configure(api_key=gemini_key)
        static.config = genai.GenerationConfig(
            temperature=1, # Medium temperature to allow for dynamic inputs
            top_p=0.5, # Lower top_p for more determinism
        )
        
    def __init__(self, model: str = "gemini-pro-3"):
        """
        Initializes the GoogleGeminiSubject instance with the specified model.

        Args:
            model (str): The model identifier to be used. Default is "gemini-pro-3".
        """
        self.model = model

    def respond(self, prompt: str) -> str:
        # Replace this with the actual API call to Google Gemini
        model = genai.GenerativeModel(model_name=self.model)
        response = model.generate_content(prompt)
        if response._done:
            return response.candidates[0].content.parts[0].text
        else:
            return f"Gemini API request with error `{response.error}`"

    async def fetch_responses(self, prompts: Collection[str]) -> Collection[str]:
        loop = asyncio.get_event_loop()
        semaphore = asyncio.Semaphore(2)

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
