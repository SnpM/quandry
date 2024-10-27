
""" 
ChatGPT Configuration
Temperature: .1
Top P: .1
"""
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

def get_case_content(prompt: str, expectation: str, response: str, prompt_encapsulator:str) -> str:
    # Encapsulate prompt with nonce for security
    prompt = f"{prompt_encapsulator} {prompt} {prompt_encapsulator}"
    return json.dumps({"__Response__": response, "__Expectation__": expectation})

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

        instruction = get_instruction()
        content = get_case_content(prompt, expectation, response, instruction.prompt_encapsulator)
        
        from google.api_core.exceptions import ResourceExhausted
        from quandry.exceptions import ResourceExhaustedException
        
        model = LlmClassifier_Gemini.get_model(instruction.text)

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
        max_batch = 8
        batches = [case_responses[i:i + max_batch] for i in range(0, len(case_responses), max_batch)]
        evals = []
        for batch in batches:
            batch_evals = self._send_gemini_batch(batch)
            evals.extend(batch_evals)
        
        return evals
    
    def _send_gemini_batch(self, case_responses: Collection[CaseResponse]) -> Collection[Evaluation]:
        instruction = get_instruction(batch=True)
        content = [get_case_content(cr.prompt, cr.expectation, cr.response, instruction.prompt_encapsulator)
                   for cr in case_responses]
    
        # Parse content by splitting with batch separator
        content = f"\n{instruction.batch_separator}\n".join(content)
        #content += f"\n{instruction.batch_separator}"
        
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
        for i in range(len(responses), len(case_responses)):
                evals.append(Evaluation(EvalCode.ERROR, f"API response missing for case {i}"))
                
        # If the number of responses is more than the number of cases, add an error evaluation
        if len(responses) > len(case_responses):
            evals.append(Evaluation(EvalCode.ERROR, f"API response count exceeded input count. Response:\n{response}"))
        
        return evals

@static_init
class LlmClassifier_ChatGPT(IEvaluator):
    @classmethod
    def static_init(static:Type):
        openai_key = os.environ[ENV_OPENAI_API_KEY]
        static.client:openai.Client = openai.Client(api_key=openai_key)        
        
        static.model_config = {
            "temperature": .5, # Medium temperature to allow for dynamic inputs
            "top_p": .3, # Lower top_p for more determinism
            "model": "gpt-4o",
        }
                
        static.initialized = True

    def evaluate(self, prompt: str, expectation: str, response: str) -> Evaluation:
        eval = self._send_chatgpt(prompt, expectation, response)
        return eval
    
    def package_message(self, instruction:str, message:str) -> List[Dict[str,Any]]:
        return [
            {"role": "system",
            "content": instruction},
            {"role": "user",
            "content": message},
        ]

    def _send_chatgpt(self, prompt, expectation, response) -> Evaluation:
        static = LlmClassifier_ChatGPT
        instruction = get_instruction()
        content = get_case_content(prompt, expectation, response, instruction.prompt_encapsulator)
        client:openai.Client = static.client   
        
        print(f"Instruction: {instruction.text}")
        print(f"Content: {content}")
        try:
            completion = client.chat.completions.create(
                messages=self.package_message(instruction.text, content),
                response_format={"type":"text"},
                **static.model_config,
                
            )
            response: str = completion.choices[0].message.content
            return parse_response(response)
        except openai.OpenAIError as e:
            # Debug error
            print(f"Failure: {e}")
            return EvalCode.ERROR, f"API query failed with error `{e}`."

    def evaluate_batch(self, case_responses: Collection[CaseResponse]) -> Collection[Evaluation]:
        max_batch = 16
        batches = [case_responses[i:i + max_batch] for i in range(0, len(case_responses), max_batch)]
        evals = []
        for batch in batches:
            batch_evals = self._send_chatgpt_batch(batch)
            evals.extend(batch_evals)
        return evals

    def _send_chatgpt_batch(self, case_responses: Collection[CaseResponse]) -> Collection[Evaluation]:
        static = LlmClassifier_ChatGPT
        
        instruction = get_instruction(batch=True)
        content = [get_case_content(cr.prompt, cr.expectation, cr.response,instruction.prompt_encapsulator)
                   for cr in case_responses]


        # Split content by batch separator
        content = f"\n{instruction.batch_separator}\n".join(content)
        content += f"\n{instruction.batch_separator}"
        client:openai.Client = static.client
        
        try:
            completion = client.chat.completions.create(
                messages=self.package_message(instruction.text, content),
                response_format={"type":"text"},
                **static.model_config
            )
            response: str = completion.choices[0].message.content
            
            responses = response.split(instruction.batch_separator)
            responses = [r.strip() for r in responses if r]
            
            evals = [parse_response(response) for response in responses]
            for i in range(len(responses), len(case_responses)):
                 evals.append(Evaluation(EvalCode.ERROR, f"API response missing for case {i}"))
                 
            # If the number of responses is more than the number of cases, add an error evaluation
            if len(responses) > len(case_responses):
                evals.append(Evaluation(EvalCode.ERROR, f"API response count exceeded input count. Response:\n{response}"))
            return evals
        except openai.OpenAIError as e:
            print(f"Failure: {e}")
            return [Evaluation(EvalCode.ERROR, f"API query failed with error `{e}`")]