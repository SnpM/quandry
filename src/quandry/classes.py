from abc import ABC, abstractmethod
from typing import final
from collections.abc import Collection
import json

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

from enum import IntEnum
class EvalCode(IntEnum):
    ERROR=0
    PASS=1
    FAIL=2
    def __str__(self):
        return self.name

class Evaluation():
    code:EvalCode
    explanation:str
    def __init__(self, code:EvalCode, explanation:str):
        self.code = code
        self.explanation = explanation
    def __str__(self):
        return json.dumps(self.__dict__)

class TestResult():
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

    def __init__(self, case:ExpectationCase, response, evalcode:EvalCode, explanation:str=""):
        self.name = case.name
        self.desc = case.desc
        self.prompt = case.prompt
        self.expectation = case.expectation
        self.response = response
        self.evalcode = evalcode
        self.explanation = explanation

    def __str__(self):
        return json.dumps(self.__dict__, default=str)

"""Evaluates the result of a TestResult"""
class IEvaluator(ABC):
    @abstractmethod
    def validate(self, prompt:str, expectation:str, response:str) -> Evaluation:
        pass

class ISubject(ABC):
    @abstractmethod
    def respond(self, prompt:str) -> str:
        pass
    
@final
class ExpectationTester():
    def __init__(self, subject:ISubject, evaluator:IEvaluator):
        self.test_subject = subject
        self.evaluator = evaluator

    def test_one(self, case:ExpectationCase) -> TestResult:
        response = self.test_subject.respond(case.prompt)
        eval = self.evaluator.validate(case.prompt, case.expectation, response)
        result = TestResult(case, response, eval.code, eval.explanation)
        return result