# import sys, os
# sys.path.insert(0,
#     os.path.dirname("../" + os.path.dirname(os.path.abspath(__file__))))

import pytest

from quandry.classes import *
from quandry.testsubjects import VanillaChatGPT
from quandry.evaluators import LlmClassifier, LlmClassifierBackend


def test_test_one():
    test = PromptTest(VanillaChatGPT(), LlmClassifier())
    result = test.test_one(PromptTestCase("What is the capital of France?",
               expect="The response should correctly convey Paris"))

    assert result.evaluation == Evaluation.PASS

def test_gemini_classifier():
    test = PromptTest(VanillaChatGPT(), LlmClassifier(LlmClassifierBackend.Gemini))
    result = test.test_one(PromptTestCase("What is the capital of France?",
        expect="The response should correctly convey Paris"))

    assert result.evaluation == Evaluation.PASS

def test_chatgptclassifier():
    test = PromptTest(VanillaChatGPT(), LlmClassifier(LlmClassifierBackend.ChatGPT))
    result = test.test_one(PromptTestCase("What is the capital of France?",
        expect="The response should correctly convey Paris"))

    assert result.evaluation == Evaluation.PASS


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