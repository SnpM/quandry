from abc import ABC, abstractmethod

class TestPrompt():
    name:str
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

    passed:bool
    """Whether the response to the test meets the expectation"""

    def __init__(self, test_result:TestResult, valid:bool):
        self.name = test_result.test_prompt.name
        self.desc = test_result.test_prompt.desc
        self.prompt = test_result.test_prompt.prompt
        self.expectation = test_result.test_prompt.expectation
        self.response = test_result.response
        self.valid = valid

"""Evaluates the result of a TestResult"""
class IEvaluator(ABC):
    @abstractmethod
    def validate (self, TestResult) -> Evaluation:
        pass

class ITestSubject(ABC):
    @abstractmethod
    def generate_output(self, prompt:str) -> str:
        pass

    def run_test(self, test_prompt:TestPrompt) -> TestResult:
        response = self.generate_output(test_prompt.prompt)
        return TestResult(test_prompt, response)