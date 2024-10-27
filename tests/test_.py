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
    
    badPrompts = [
        "What is the capital of Spain?",
        "What is the capital of Switzerland?",
    ]
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
        ExpectationCase("What is the capital of Netherlands?",
            expect="The response should correctly convey Amsterdam"),
        ExpectationCase("What is the capital of Belgium?",
            expect="The response should correctly convey Brussels"),
        ExpectationCase("What is the capital of Switzerland?",
            expect="The response should correctly convey Bern"),
        ExpectationCase("What is the capital of Austria?",
            expect="The response should correctly convey Vienna"),
        ExpectationCase("What is the capital of Greece?",
            expect="The response should correctly convey Athens"),
    ]
    
    # Create a mock subject that responds correctly to most cases but incorrectly to some
    class MockSubjectWithFailures(ISubject):
        def respond(self, prompt: str) -> str:
            responses = {
                "What is the capital of France?": "Paris",
                "What is the capital of Germany?": "Berlin",
                "What is the capital of Italy?": "Rome",
                "What is the capital of Spain?": "Madrid",
                "What is the capital of Portugal?": "Lisbon",
                "What is the capital of Netherlands?": "Amsterdam",
                "What is the capital of Belgium?": "Brussels",
                "What is the capital of Switzerland?": "Bern",
                "What is the capital of Austria?": "Vienna",
                "What is the capital of Greece?": "Athens",
            }
            if prompt in badPrompts:
                return "New Zealand"
            return responses.get(prompt, "I don't know")
        
    tester = ExpectationTester(MockSubjectWithFailures(), LlmClassifier_Gemini())
    results = tester.test_batch(cases)

    for result in results:
        if result.prompt in ["What is the capital of Spain?", "What is the capital of Switzerland?"]:
            assert result.evalcode == EvalCode.FAIL, result.explanation
        else:
            assert result.evalcode == EvalCode.PASS, result.explanation

def test_chatgpt_classifier_with_failures():
    class MockSubjectWithFailures(ISubject):
        def respond(self, prompt: str) -> str:
            if prompt == "What is the capital of France?":
                return "Lyon"
            return "The Capital of France is Paris"
    
    for i in range(10):
        test = ExpectationTester(MockSubjectWithFailures(), LlmClassifier_ChatGPT())
        result = test.test_one(ExpectationCase("What is the capital of France?",
            expect="The response should convey 'Paris'"))
        if result.evalcode != EvalCode.PASS:
            print(result.explanation)

def test_batch_chatgpt():
    # Define the cases exactly as per the table data
    cases = [
        ExpectationCase("What is the capital of France?", 
            expect="When asked about the capital of France, correctly convey Paris"),
        ExpectationCase("What is the capital of Germany?", 
            expect="When asked about the capital of Germany, correctly convey Berlin"),
        ExpectationCase("What is the capital of Italy?", 
            expect="When asked about the capital of Italy, correctly convey Rome"),
        ExpectationCase("What is the capital of Spain?", 
            expect="When asked about the capital of Spain, correctly convey Madrid"),
        ExpectationCase("What is the capital of Portugal?", 
            expect="When asked about the capital of Portugal, correctly convey Lisbon"),
        ExpectationCase("What is the capital of Netherlands?", 
            expect="When asked about the capital of Netherlands, correctly convey Amsterdam"),
        ExpectationCase("What is the capital of Belgium?", 
            expect="When asked about the capital of Belgium, correctly convey Brussels"),
        ExpectationCase("What is the capital of Switzerland?", 
            expect="When asked about the capital of Switzerland, correctly convey Bern"),
        ExpectationCase("What is the capital of Austria?", 
            expect="When asked about the capital of Austria, correctly convey Vienna"),
        ExpectationCase("What is the capital of Greece?", 
            expect="When asked about the capital of Greece, correctly convey Athens"),
    ]
    
    import random
    random.seed(0)
    cases = random.choices(cases, k=30)
    #cases = cases[:8]

    badPrompts = [
        "What is the capital of Spain?",
        "What is the capital of Switzerland?",
        "What is the capital of France?",
    ]
    # Create a mock subject that responds correctly to each capital question
    class MockSubject(ISubject):
        def respond(self, prompt: str) -> str:
            responses = {
                "What is the capital of France?": "Paris",
                "What is the capital of Germany?": "Berlin",
                "What is the capital of Italy?": "Rome",
                "What is the capital of Spain?": "Madrid",
                "What is the capital of Portugal?": "Lisbon",
                "What is the capital of Netherlands?": "Amsterdam",
                "What is the capital of Belgium?": "Brussels",
                "What is the capital of Switzerland?": "Bern",
                "What is the capital of Austria?": "Vienna",
                "What is the capital of Greece?": "Athens",
            }
            for badPrompt in badPrompts:
                if prompt == badPrompt:
                    return "New Zealand"
            return responses.get(prompt, "I don't know")
        
    # Instantiate the tester and run the test batch
    tester = ExpectationTester(MockSubject(), LlmClassifier_ChatGPT())
    results = tester.test_batch(cases)

    # Assert that all results meet expectations
    for i in range(len(results)):
        result = results[i]
        expected = EvalCode.FAIL if result.prompt in badPrompts else EvalCode.PASS
        assert result.evalcode == expected, (i, str(result))
        
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