from abc import ABC, abstractmethod
from typing import final
from collections.abc import Collection
import json

class PromptTestCase():
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
class PromptTestResult():
    test_case:PromptTestCase
    response:str
    def __init__(self, test_case:PromptTestCase, response:str):
        self.test_case = test_case
        self.response = response

    def __str__(self):
        return json.dumps(self.__dict__)

from enum import IntEnum
class Evaluation(IntEnum):
    ERROR=0
    PASS=1
    FAIL=2
    def __str__(self):
        return self.name

class EvaluationResult():

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

    evaluation:Evaluation
    """Whether the response to the test meets the expectation"""

    explanation:str
    """Explanation for evaluation decision."""

    def __init__(self, test_result:PromptTestResult, evaluation:bool, explanation:str=""):
        self.name = test_result.test_case.name
        self.desc = test_result.test_case.desc
        self.prompt = test_result.test_case.prompt
        self.expectation = test_result.test_case.expectation
        self.response = test_result.response
        self.evaluation = evaluation
        self.explanation = explanation

    def __str__(self):
        return json.dumps(self.__dict__, default=str)

"""Evaluates the result of a TestResult"""
class IEvaluator(ABC):
    @abstractmethod
    def validate (self, TestResult) -> EvaluationResult:
        pass

class ITestSubject(ABC):
    @abstractmethod
    def generate_output(self, prompt:str) -> str:
        pass

    @final
    def run_test(self, test_case:PromptTestCase) -> PromptTestResult:
        response = self.generate_output(test_case.prompt)
        return PromptTestResult(test_case, response)
    

@final
class PromptTest():
    def __init__(self, test_subject:ITestSubject, evaluator:IEvaluator):
        self.test_subject = test_subject
        self.evaluator = evaluator

    def test_one(self, test_case:PromptTestCase):
        result = self.test_subject.run_test(test_case)
        evaluation = self.evaluator.validate(result)
        return evaluation
    
    def test(self, test_cases:Collection[PromptTestCase]):
        return [self.test_one(test_case) for test_case in test_cases]