"""
Qwen Judge Model for DeepEval
Uses Qwen3-235B as the judge/evaluator model for LLM evaluation metrics.
This larger model evaluates outputs from the smaller Qwen3-32B application model.
"""
import json
import boto3
from deepeval.models.base_model import DeepEvalBaseLLM

BEDROCK_REGION = "ap-south-1"
JUDGE_MODEL_ID = "qwen.qwen3-235b-a22b-2507-v1:0"  # Judge model (larger, more capable)

bedrock_runtime = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)


def _raw_qwen_call(prompt: str, max_tokens: int = 1024) -> str:
    """
    Make a raw API call to Qwen3-235B judge model.
    
    Args:
        prompt: The evaluation prompt
        max_tokens: Maximum tokens for judge response
    
    Returns:
        Judge model response
    """
    body = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,  # Low temperature for consistent evaluation
    }
    resp = bedrock_runtime.invoke_model(
        modelId=JUDGE_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    payload = json.loads(resp["body"].read())
    return payload["choices"][0]["message"]["content"]


class QwenJudge(DeepEvalBaseLLM):
    """
    DeepEval-compatible wrapper for Qwen3-235B as a judge model.
    
    This class implements the DeepEvalBaseLLM interface to allow
    using Qwen3-235B on AWS Bedrock as an evaluation judge for
    various DeepEval metrics.
    """
    
    def load_model(self):
        """No model loading required - uses Bedrock API."""
        return None

    def generate(self, prompt: str) -> str:
        """
        Generate evaluation response from judge model.
        
        Args:
            prompt: The evaluation prompt from DeepEval metric
        
        Returns:
            Judge model's evaluation response
        """
        return _raw_qwen_call(prompt)

    async def a_generate(self, prompt: str) -> str:
        """Async wrapper for generate method."""
        return self.generate(prompt)

    def get_model_name(self) -> str:
        """Return model identifier."""
        return "Qwen3-235B-A22B (Bedrock Judge)"
