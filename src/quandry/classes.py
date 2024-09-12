from abc import ABC, abstractmethod

class TestPrompt():
    desc:str
    prompt:str
    expectation:str

    def __init__(self, desc:str, prompt:str, expect:str):
        self.desc = desc
        self.prompt = prompt
        self.expectation = expect

class TestResult():
    test_prompt:TestPrompt
    response:str
    def __init__(self, test_prompt:TestPrompt, response:str):
        self.test_prompt = test_prompt
        self.response = response

class Evaluation():
    test_result:TestResult
    valid:bool

"""Evaluates the result of a TestResult"""
class IEvaluator(ABC):
    @abstractmethod
    def validate (self, TestResult) -> Evaluation:
        pass

