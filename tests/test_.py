# import sys, os
# sys.path.insert(0,
#     os.path.dirname("../" + os.path.dirname(os.path.abspath(__file__))))

import pytest

from quandry.classes import *
from quandry.subjects import VanillaChatGPT
from quandry.evaluators import LlmClassifier
def test_test_one():
    test = PromptTest(VanillaChatGPT(), LlmClassifier())
    result = test.test_one(PromptTestCase("What is the capital of France?",
               expect="The capital of France, Paris."))
    assert result.evaluation == True



# @pytest.mark.benchmark(
#     group="group-name",
#     min_time=0.1,
#     max_time=0.5,
#     min_rounds=5,
#     disable_gc=True,
#     warmup=True,
# )
# def test_expensive_operation_iteration(benchmark):  
#     @benchmark
#     def asdf():
#         test = PromptTest(VanillaChatGPT(), LlmClassifier())

#         result = test.test_one(
#             PromptTestCase("What is the capital of France?",
#                expect="The capital of France, Paris.") )
#         return result