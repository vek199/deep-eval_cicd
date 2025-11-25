import json
import boto3
from deepeval.models.base_model import DeepEvalBaseLLM

BEDROCK_REGION = "ap-south-1"
MODEL_ID = "qwen.qwen3-235b-a22b-2507-v1:0"

bedrock_runtime = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)


def _raw_qwen_call(prompt: str) -> str:
    body = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 512,
        "temperature": 0.2,
    }
    resp = bedrock_runtime.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    payload = json.loads(resp["body"].read())
    # Qwen3 on Bedrock returns response in choices[0].message.content format
    return payload["choices"][0]["message"]["content"]


class QwenJudge(DeepEvalBaseLLM):
    def load_model(self):
        # Nothing heavy to load; we just use the global client.
        return None

    def generate(self, prompt: str) -> str:
        return _raw_qwen_call(prompt)

    async def a_generate(self, prompt: str) -> str:
        # Optional: simple async wrapper
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return "Qwen3-235B-A22B-2507 (Bedrock Judge)"
