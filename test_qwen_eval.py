# test_qwen_eval.py
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from bedrock_qwen import call_qwen       # app under test (Qwen)
from qwen_judge import QwenJudge        # judge model (same Qwen)

qwen_judge = QwenJudge()

answer_relevancy = AnswerRelevancyMetric(
    model=qwen_judge,
    threshold=0.7,
)

TEST_CASES = [
    LLMTestCase(
        input="What is the capital of France?",
        expected_output="Paris",
    ),
]


def run_app(input_text: str) -> str:
    return call_qwen(input_text)


def test_qwen_answer_relevancy():
    case = TEST_CASES[0]
    case.actual_output = run_app(case.input)
    assert_test(metrics=[answer_relevancy], test_case=case)
