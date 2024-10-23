from abc import ABC, abstractmethod
from typing import final
from collections.abc import Collection
import json
import quandry.exceptions as exceptions
import random
class ExpectationCase():
    name:str
    desc:str
    prompt:str
    expectation:str

    def __init__(self, prompt:str, expect:str, name:str="Unnamed", desc:str=""):
        self.name = name
        self.desc = desc
        self.prompt = prompt
        self.expectation = expect

    def __str__(self):
        return json.dumps(self.__dict__)

class CaseResponse():
    prompt:str
    response:str
    expectation:str
    def __init__(self, prompt:str, expectation:str, response:str):
        self.prompt = prompt
        self.expectation = expectation
        self.response = response
    def __str__(self):
        return json.dumps(self.__dict__)

from enum import IntEnum
import asyncio

@final
class EvalCode(IntEnum):
    ERROR=0
    PASS=1
    FAIL=2
    def __str__(self):
        return self.name

@final
class Evaluation():
    code:EvalCode
    explanation:str
    def __init__(self, code:EvalCode, explanation:str):
        self.code = code
        self.explanation = explanation
    def __str__(self):
        return json.dumps(self.__dict__)

@final
class CaseResult():
    name:str
    """Name of the Test"""
    desc:str
    """Description of the Test"""
    prompt:str
    """Prompt of the Test"""
    expectation:str
    """Expectation of the response to the prompt for the Test"""
    response:str
    """Response to the prompt of the Test"""
    evaluation:EvalCode
    """Whether the response to the test meets the expectation"""
    explanation:str
    """Explanation for evaluation decision."""

    def __init__(self, case:ExpectationCase, response:str, evalcode:EvalCode, explanation:str=""):
        self.name = case.name
        self.desc = case.desc
        self.prompt = case.prompt
        self.expectation = case.expectation
        self.response = response
        self.evalcode = evalcode
        self.explanation = explanation

    def __str__(self):
        return json.dumps(self.__dict__, default=str)

from typing import Tuple
import warnings
"""Evaluates the result of a TestResult"""
class IEvaluator(ABC):
    @abstractmethod
    def evaluate(self, prompt:str, expectation:str, response:str) -> Evaluation:
        pass
    def evaluate_batch(self, case_responses: Collection[CaseResponse]) -> Collection[Evaluation]:
        # Warning: This is the default serial implementation
        warnings.warn("Using default serial implementation for evaluate_batch. This may not be efficient for large batches.", UserWarning)
        return [self.evaluate(cr.prompt, cr.expectation, cr.response) for cr in case_responses]
       
class ISubject(ABC):
    @abstractmethod
    def respond(self, prompt:str) -> str:
        pass

    def respond_batch(
            self,
            prompts: Collection[str]) -> Collection[str]:
        return [self.respond(prompt) for prompt in prompts]
        
    
@final
class ExpectationTester():
    def __init__(self, subject:ISubject, evaluator:IEvaluator):
        self.subject = subject
        self.evaluator = evaluator

    def test_one(self, case:ExpectationCase) -> CaseResult:
        response = self.subject.respond(case.prompt)
        eval = self.evaluator.evaluate(case.prompt, case.expectation, response)
        result = CaseResult(case, response, eval.code, eval.explanation)
        return result
    
    async def test_one_async(self, case: ExpectationCase, max_wait=10, max_retries=5) -> CaseResult:
        def calculate_backoff(retries, max_wait, max_retries):
            """
            Calculate exponential backoff scaled to max_wait.
            """
            # Scale the backoff to be between 0 and max_wait
            scaling_factor = max_wait / (2 ** max_retries - 1)  # Scale backoff relative to retries and max_wait
            backoff_time = min(max_wait, (2 ** retries - 1) * scaling_factor + random.uniform(0, 1))
            return backoff_time 
        
        # Initialize variables
        response = None
        eval = None

        # Get the event loop
        loop = asyncio.get_event_loop()

        # First loop for response
        retries = 0
        while retries <= max_retries:
            try:
                # Try to get the response asynchronously
                response = await loop.run_in_executor(None, self.subject.respond, case.prompt)
                break  # Exit loop on success
            except exceptions.ResourceExhaustedException as e:
                retries += 1
                if retries > max_retries:
                    print(f"Failed to get response after {max_retries} retries. Returning None.")
                    eval = Evaluation(EvalCode.ERROR, f"Failed to get response after {max_retries} retries.")

                # Scaled exponential backoff with jitter
                backoff_time = calculate_backoff(retries, max_wait, max_retries)
                print(f"Error occurred in response: {e}. Retrying in {backoff_time:.2f} seconds (attempt {retries}/{max_retries})...")
                await asyncio.sleep(backoff_time)

        # Separate loop for evaluation
        retries = 0
        if response is not None:
            while retries <= max_retries:
                try:
                    # Try to evaluate asynchronously
                    eval = await loop.run_in_executor(None, self.evaluator.evaluate, case.prompt, case.expectation, response)
                    break  # Exit loop on success
                except exceptions.ResourceExhaustedException as e:
                    retries += 1
                    if retries > max_retries:
                        print(f"Failed to evaluate after {max_retries} retries. Returning ERROR evaluation.")
                        eval = Evaluation(EvalCode.ERROR,
                                          f"Failed to evaluate after {max_retries} retries.")

                    # Scaled exponential backoff with jitter
                    backoff_time = calculate_backoff(retries, max_wait, max_retries)
                    print(f"Error occurred in evaluation: {e}. Retrying in {backoff_time:.2f} seconds (attempt {retries}/{max_retries})...")
                    await asyncio.sleep(backoff_time)
        
        # Return the final result after both response and evaluation are successful
        code = eval.code if eval else EvalCode.ERROR
        explanation = eval.explanation if eval else "Evaluation missing"
        result = CaseResult(case, response, code, explanation)
        return result
    
    def test_batch(self, cases: Collection[ExpectationCase]) -> Collection[CaseResult]:
        responses = self.subject.respond_batch([case.prompt for case in cases])
        case_responses = [CaseResponse(c.prompt, c.expectation, r) for c, r in zip(cases, responses)]
        evaluations = self.evaluator.evaluate_batch(case_responses)
        results = [CaseResult(case, cr.response, eval.code, eval.explanation)
                   for case, cr, eval in zip(cases, case_responses, evaluations)]
        return results