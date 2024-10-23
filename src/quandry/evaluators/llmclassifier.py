
""" 
ChatGPT Configuration
Temperature: .1
Top P: .1
"""
_assistant_name = "LLM Expectation Classifier"

from quandry import *
import openai
import json
import google.generativeai as genai
from typing import *
from enum import Enum
import os
import asyncio
from quandry.utils import static_init
import uuid
from quandry.evaluators.llmclassifier_prompts import get_instruction


ENV_GEMINI_API_KEY = "KEY_GEMINIAPI"
ENV_OPENAI_API_KEY = "KEY_OPENAI"

def parse_response(response: str) -> Evaluation:
    response = response.strip()
    parsed = response.split(':::')
    if len(parsed) != 2:
        return Evaluation(EvalCode.ERROR, f"API response invalid: `{response}`")

    raw_code, explanation = parsed
    code = EvalCode.ERROR
    match raw_code:
        case 'ERROR':
            code = EvalCode.ERROR
        case 'PASS':
            code = EvalCode.PASS
        case 'FAIL':
            code = EvalCode.FAIL
        case _:
            explanation = f"Decision invalid: `{response}`"
    return Evaluation(code, explanation)

def get_case_content(prompt: str, expectation: str, response: str) -> str:
    return json.dumps({"Prompt": prompt, "Response": response, "Expectation": expectation})

@static_init
class LlmClassifier_Gemini(IEvaluator):
    @classmethod
    def static_init(static:Type):
        gemini_key = os.environ[ENV_GEMINI_API_KEY]
        genai.configure(api_key=gemini_key)
        static.config = genai.GenerationConfig(
            temperature=.5, # Medium temperature to allow for dynamic inputs
            top_p=0.8, # Lower top_p for more determinism
            top_k=48 # Medium top_k to allow for explanations
        )
        # Define safety settings so outputs aren't blocked
        static.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
    @classmethod
    def get_model(cls, instruction:str)-> genai.GenerativeModel:
        if cls.config is None:
            raise ValueError("Gemini API not initialized")
                
        return genai.GenerativeModel(
        "gemini-1.5-flash",
        generation_config=cls.config,
        safety_settings=cls.safety_settings,
        system_instruction=instruction)
    
        


    def evaluate(self, prompt: str, expectation: str, response: str) -> Evaluation:
        eval = self._send_gemini(prompt, expectation, response)
        return eval

    def _send_gemini(self, prompt, expectation, response) -> Evaluation:

        content = json.dumps({"Prompt": prompt, "Response": response, "Expectation": expectation})
        
        from google.api_core.exceptions import ResourceExhausted
        from quandry.exceptions import ResourceExhaustedException
        
        model = LlmClassifier_Gemini.get_model(get_instruction().text)

        # Google API could return a ResourceExhausted error; catch and return our own exception
        try:
            response = model.generate_content(content)
        except ResourceExhausted as e:
            raise ResourceExhaustedException("Gemini API") from e
        if response._done:
            return parse_response(response.candidates[0].content.parts[0].text)
        else:
            return Evaluation(EvalCode.ERROR, f"API query failed with error `{response.error}`")
        
    def evaluate_batch(self, case_responses: Collection[CaseResponse]) -> Collection[Evaluation]:
        evals = self._send_gemini_batch(case_responses)
        return evals
    
    def _send_gemini_batch(self, case_responses: Collection[CaseResponse]) -> Collection[Evaluation]:
        content = [get_case_content(cr.prompt, cr.expectation, cr.response) for cr in case_responses]
    
        
        instruction = get_instruction(True)
        
        # Parse content by splitting with batch separator
        content = f"\n{instruction.batch_separator}\n".join(content) 
        
        # Get model with batch instruction
        model:genai.GenerativeModel = LlmClassifier_Gemini.get_model(instruction.text)
        response = model.generate_content(content)

        # Split the responses by lines and parse them for each case in order
        responses = response.candidates[0].content.parts[0].text.split(instruction.batch_separator)
        # Strip all splits
        responses = [r.strip() for r in responses]
        # Delete splits that are empty
        responses = [r for r in responses if r]
        if len(responses) != len(case_responses):
            return [Evaluation(EvalCode.ERROR, "API response count does not match input count. Response:\n" +
                               response.candidates[0].content.parts[0].text)]
        
        evals = [parse_response(response) for response in responses]
        return evals

@static_init
class LlmClassifier_ChatGPT(IEvaluator):
    @classmethod
    def static_init(static:Type):
        openai_key = os.environ[ENV_OPENAI_API_KEY]
        static.client:openai.Client = openai.Client(api_key=openai_key)        
        
        # Find assistant with name _assistant_name
        assistants = static.client.beta.assistants.list()
        assistant = None
        for a in assistants.data:
            if a.name == _assistant_name:
                assistant = a
                break
        if assistant is None:
            assistant = static.client.beta.assistants.create(
                name=_assistant_name,
                description="Evaluates responses to prompts based on given expectations.",
                instructions=get_instruction().text,
                model="gpt-4o",
                temperature=0.5, # Medium temperature to allow for dynamic inputs
                top_p=0.6, # Lower top_p for more determinism
            )
        static.assistant = assistant
                
        static.initialized = True

    def evaluate(self, prompt: str, expectation: str, response: str) -> Evaluation:
        eval = self._send_chatgpt(prompt, expectation, response)
        return eval

    def _send_chatgpt(self, prompt, expectation, response) -> Evaluation:
        static = LlmClassifier_ChatGPT         
        content = json.dumps({"Prompt": prompt, "Response": response, "Expectation": expectation})
        client = static.client
        assistant = static.assistant
        thread = static.client.beta.threads.create()

        client.beta.threads.messages.create(
            thread_id=thread.id,
            content=content,
            role="user"
        )

        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant.id
        )

        if run.status == 'completed':
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            response: str = messages.data[0].content[0].text.value
            return parse_response(response)
        else:
            print("Failure: " + run.status + " | " + run.error)
            return EvalCode.ERROR, f"API query failed with error `{run.error}`"

    def evaluate_batch(self, case_responses: Collection[CaseResponse]) -> Collection[Evaluation]:
        evals = self._send_chatgpt_batch(case_responses)
        return evals

    def _send_chatgpt_batch(self, case_responses: Collection[CaseResponse]) -> Collection[Evaluation]:
        static = LlmClassifier_ChatGPT
        content = [get_case_content(cr.prompt, cr.expectation, cr.response) for cr in case_responses]

        instruction = get_instruction()

        # Split content by batch separator
        content = f"\n{instruction.batch_separator}\n".join(content)
        client:openai.Client = static.client
        assistant = static.assistant
        thread = static.client.beta.threads.create()
        # Add content to thread
        client.beta.threads.messages.create(
            thread_id=thread.id,
            content=content,
            role="user"
        )
        
        print(instruction.text)
        print(content)
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant.id,
            instructions=instruction.text,
            model = "gpt-4o"
        )

        if run.status == 'completed':
            print("ranned")
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            response: str = messages.data[0].content[0].text.value
            responses = response.split(instruction.batch_separator)
            responses = [r.strip() for r in responses if r]
            if len(responses) != len(case_responses):
                return [Evaluation(EvalCode.ERROR, "API response count does not match input count. Response:\n" + response)]
            
            evals = [parse_response(response) for response in responses]
            return evals
        else:
            print("Failure: " + run.status + " | " + run.error)
            return [Evaluation(EvalCode.ERROR, f"API query failed with error `{run.error}`")]