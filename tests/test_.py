# import sys, os
# sys.path.insert(0,
#     os.path.dirname("../" + os.path.dirname(os.path.abspath(__file__))))

import pytest

from quandry.classes import *
from quandry.subjects import VanillaChatGPT
from quandry.evaluators import LlmClassifier_Gemini, LlmClassifier_ChatGPT


def test_test_one():
    test = ExpectationTester(VanillaChatGPT(), LlmClassifier_Gemini())
    result = test.test_one(ExpectationCase("What is the capital of France?",
               expect="The response should correctly convey Paris"))

    assert result.evalcode == EvalCode.PASS

def test_gemini_classifier():
    test = ExpectationTester(VanillaChatGPT(), LlmClassifier_Gemini())
    result = test.test_one(ExpectationCase("What is the capital of France?",
        expect="The response should correctly convey Paris"))

    assert result.evalcode == EvalCode.PASS

def test_chatgptclassifier():
    test = ExpectationTester(VanillaChatGPT(), LlmClassifier_ChatGPT())
    result = test.test_one(ExpectationCase("What is the capital of France?",
        expect="The response should correctly convey Paris"))

    assert result.evalcode == EvalCode.PASS

def test_async_test_one():
    tester = ExpectationTester(VanillaChatGPT(), LlmClassifier_Gemini())
    result = asyncio.run(tester.test_one_async(ExpectationCase("What is the capital of France?",
               expect="The response should correctly convey Paris")))

    assert result.evalcode == EvalCode.PASS

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
        
    tester = ExpectationTester(MockSubject(), LlmClassifier_Gemini())
    results = tester.test_batch(cases)

    assert all([result.evalcode == EvalCode.PASS for result in results])

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