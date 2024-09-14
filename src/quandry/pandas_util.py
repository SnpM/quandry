import pandas as pd
from quandry.classes import *
from typing import List
def df2cases(df:pd.DataFrame,
        name_col:str = "name", 
        desc_col:str = "desc",
        prompt_col:str = "prompt",
        expect_col:str = "expectation") -> List[PromptTestCase]:
    """Creates a list of TestCases from a DataFrame"""
    return [PromptTestCase(row[prompt_col], row[expect_col], row[name_col], row[desc_col]) for index, row in df.iterrows()]


def eval2df(evaluations:List[EvaluationResult]) -> pd.DataFrame:
    """Converts a list of Evaluations to a DataFrame"""
    df = pd.DataFrame([evaluation.__dict__ for evaluation in evaluations])
    # Ensure evaluation enum is properly encoded
    df["evaluation"] = df["evaluation"].apply(lambda x: Evaluation(x).name)
    return df