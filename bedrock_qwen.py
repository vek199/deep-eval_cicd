import json
import boto3

BEDROCK_REGION = "ap-south-1"  # change if needed
MODEL_ID = "qwen.qwen3-235b-a22b-2507-v1:0"

bedrock_runtime = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

def call_qwen(prompt: str) -> str:
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
