# import sys, os
# sys.path.insert(0,
#     os.path.dirname("../" + os.path.dirname(os.path.abspath(__file__))))

import pytest

from quandry.classes import *
from quandry.subjects import OpenAiSubject
from quandry.evaluators import LlmClassifier_Gemini, LlmClassifier_ChatGPT


def test_test_one():
    test = ExpectationTester(OpenAiSubject(), LlmClassifier_Gemini())
    result = test.test_one(ExpectationCase("What is the capital of France?",
               expect="The response should correctly convey Paris"))

    assert result.evalcode == EvalCode.PASS, result.explanation

def test_gemini_classifier():
    test = ExpectationTester(OpenAiSubject(), LlmClassifier_Gemini())
    result = test.test_one(ExpectationCase("What is the capital of France?",
        expect="The response should correctly convey Paris"))

    assert result.evalcode == EvalCode.PASS, result.explanation

def test_chatgptclassifier():
    class mock_subject(ISubject): respond = lambda self, prompt: "The Capital of France is Paris"
    for i in range(10):
        test = ExpectationTester(mock_subject(), LlmClassifier_ChatGPT())
        result = test.test_one(ExpectationCase("What is the capital of France?",
            expect="The response should convey 'Paris'"))
        if result.evalcode != EvalCode.PASS:
            print(result.explanation)
        assert result.evalcode == EvalCode.PASS, result.explanation

def test_async_test_one():
    tester = ExpectationTester(OpenAiSubject(), LlmClassifier_Gemini())
    result = asyncio.run(tester.test_one_async(ExpectationCase("What is the capital of France?",
               expect="The response should correctly convey Paris")))

    assert result.evalcode == EvalCode.PASS, result.explanation

def test_batch_gemini():
    cases = [
        ExpectationCase("What is the capital of France?",
            expect="The response should correctly convey Paris"),
        ExpectationCase("What is the capital of Germany?",
            expect="The response should correctly convey Berlin"),
        ExpectationCase("What is the capital of Italy?",
            expect="The response should correctly convey Rome"),
        ExpectationCase("What is the capital of Spain?",
            expect="The response should correctly convey Madrid"),
        ExpectationCase("What is the capital of Portugal?",
            expect="The response should correctly convey Lisbon"),
    ]
    # Create an array of 20 cases that are randomly selected from cases
    import random
    cases = random.choices(cases, k=20)

    # Create a mock subject that responds correctly to cases for capitals of countries
    class MockSubject(ISubject):
        def respond(self, prompt: str) -> str:
            responses = {
            "What is the capital of France?": "Paris",
            "What is the capital of Germany?": "Berlin",
            "What is the capital of Italy?": "Rome",
            "What is the capital of Spain?": "Madrid",
            "What is the capital of Portugal?": "Lisbon",
            }
            return responses.get(prompt, "I don't know")
        
    tester = ExpectationTester(MockSubject(), LlmClassifier_Gemini())
    results = tester.test_batch(cases)

    for result in results:
        assert result.evalcode == EvalCode.PASS, result.explanation
    
def test_batch_chatgpt():
    cases = [
        ExpectationCase("What is the capital of France?",
            expect="The response should correctly convey Paris"),
        ExpectationCase("What is the capital of Germany?",
            expect="The response should correctly convey Berlin"),
        ExpectationCase("What is the capital of Italy?",
            expect="The response should correctly convey Rome"),
        ExpectationCase("What is the capital of Spain?",
            expect="The response should correctly convey Madrid"),
        ExpectationCase("What is the capital of Portugal?",
            expect="The response should correctly convey Lisbon"),
    ] * 5

    # Create a mock subject that responds correctly to cases for capitals of countries
    class MockSubject(ISubject):
        def respond(self, prompt: str) -> str:
            responses = {
            "What is the capital of France?": "Paris",
            "What is the capital of Germany?": "Berlin",
            "What is the capital of Italy?": "Rome",
            "What is the capital of Spain?": "Madrid",
            "What is the capital of Portugal?": "Lisbon",
            }
            return responses.get(prompt, "I don't know")
        
    tester = ExpectationTester(MockSubject(), LlmClassifier_ChatGPT())
    results = tester.test_batch(cases)
    
    for result in results:
        assert result.evalcode == EvalCode.PASS, result.explanation

"""
@pytest.mark.benchmark(
    group="group-name",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    disable_gc=True,
    warmup=False,
    warmup_iterations=1,
)
def test_bench_chatgpt(benchmark):  
    @benchmark
    def chatgpt():
        test = PromptTest(VanillaChatGPT(), LlmClassifier(LlmClassifierBackend.ChatGPT))
        result = test.test_one(PromptTestCase("What is the capital of France?",
            expect="The response should correctly convey Paris"))
@pytest.mark.benchmark(
    group="group-name",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    disable_gc=True,
    warmup=False,
    warmup_iterations=1,
)
def test_bench_gemini(benchmark):
    @benchmark  
    def gemini():
        test = PromptTest(VanillaChatGPT(), LlmClassifier(LlmClassifierBackend.Gemini))
        result = test.test_one(PromptTestCase("What is the capital of France?",
            expect="The response should correctly convey Paris"))
"""