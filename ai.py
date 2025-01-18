import requests
from typing import Optional
import argparse
import json
from argparse import RawTextHelpFormatter
import os

BASE_API_URL = "https://api.langflow.astra.datastax.com"
LANGFLOW_ID = "a251ca29-c516-4b2d-b0a8-dc39c2749687"
FLOW_ID = "3c997afa-eb72-4552-8dd0-f8e8fb321f27"
APPLICATION_TOKEN = os.getenv("LANGFLOW_TOKEN")
ENDPOINT = "" # You can set a specific endpoint name in the flow settings

def get_results(ip):
        TWEAKS = {
                "ChatInput-TWdFJ": {
                "input_value": "ip"
                }
        }
        return run_flow("", tweaks=TWEAKS, application_token=APPLICATION_TOKEN)

def run_flow(message: str,
  output_type: str = "chat",
  input_type: str = "chat",
  tweaks: Optional[dict] = None,
  application_token: Optional[str] = None) -> dict:
    api_url = f"{BASE_API_URL}/lf/{LANGFLOW_ID}/api/v1/run/mihir"

    payload = {
        "input_value": message,
        "output_type": output_type,
        "input_type": input_type,
    }
    headers = None
    if tweaks:
        payload["tweaks"] = tweaks
    if application_token:
        headers = {"Authorization": "Bearer " + application_token, "Content-Type": "application/json"}
    response = requests.post(api_url, json=payload, headers=headers)
    return response.json()#["outputs"][0]["results"]["text"]["data"]["text"]

result = get_results("ip: milton")
print(result)