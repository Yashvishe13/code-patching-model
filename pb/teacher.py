"""GPT-5.6-terra via the Azure OpenAI Responses API, as a policy (act(messages) -> (content, calls)).

Stateless calls (store=False) with the model's encrypted reasoning cached per turn and replayed, so it
keeps its plan across turns while staying opaque to us. One tool call per turn. Creds from .env:
AZURE_API_KEY, AZURE_ENDPOINT (an .../openai/v1 base URL).
"""
import json
import os
import random
import time
from pathlib import Path

from dotenv import dotenv_values
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

from pb.env import TOOLS

ENV = Path(__file__).resolve().parent.parent / ".env"
RESPONSE_TOOLS = [{"type": "function", **t} for t in TOOLS]


class Teacher:
    def __init__(self, model="gpt-5.6-terra", reasoning_effort="medium", max_output_tokens=16000):
        cfg = {**dotenv_values(ENV), **os.environ}
        self.client = OpenAI(base_url=cfg["AZURE_ENDPOINT"].strip().rstrip("/"),
                             api_key=cfg["AZURE_API_KEY"].strip(), max_retries=6, timeout=600)
        self.model = model
        self.kwargs = dict(reasoning={"effort": reasoning_effort}, max_output_tokens=max_output_tokens,
                           store=False, include=["reasoning.encrypted_content"])
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        self._turns = {}  # assistant message index -> the API output items that produced it

    def _input(self, messages):
        items, pending = [], []
        for i, m in enumerate(messages):
            if m["role"] == "assistant":
                names = [tc["function"]["name"] for tc in m.get("tool_calls", [])]
                cached = self._turns.get(i)
                if cached is None or [c["name"] for c in cached if c["type"] == "function_call"] != names:
                    cached = ([{"role": "assistant", "content": m["content"]}] if m["content"] else []) + [
                        {"type": "function_call", "call_id": f"c{i}_{j}", "name": tc["function"]["name"],
                         "arguments": json.dumps(tc["function"]["arguments"])}
                        for j, tc in enumerate(m.get("tool_calls", []))]
                items += cached
                pending = [c["call_id"] for c in cached if c["type"] == "function_call"]
            elif m["role"] == "tool":
                items.append({"type": "function_call_output", "call_id": pending.pop(0), "output": m["content"]}
                             if pending else {"role": "user", "content": m["content"]})
            else:
                items.append({"role": m["role"], "content": m["content"]})
        return items

    def _create(self, messages):
        delay = 10
        for attempt in range(10):
            try:
                return self.client.responses.create(model=self.model, input=self._input(messages),
                                                     tools=RESPONSE_TOOLS, tool_choice="auto",
                                                     parallel_tool_calls=False, **self.kwargs)
            except (RateLimitError, APIConnectionError, InternalServerError, APITimeoutError):
                if attempt == 9:
                    raise
                time.sleep(delay + random.uniform(0, delay))
                delay = min(delay * 2, 300)

    def act(self, messages):
        r = self._create(messages)
        self.usage["calls"] += 1
        self.usage["input_tokens"] += r.usage.input_tokens
        self.usage["output_tokens"] += r.usage.output_tokens
        text, kept, call = "", [], None
        for item in r.output:
            if item.type == "reasoning":
                kept.append(item.model_dump(exclude_none=True))
            elif item.type == "message":
                text += "".join(c.text for c in item.content if c.type == "output_text")
                kept.append({"role": "assistant", "content": text})
            elif item.type == "function_call" and call is None:
                call = item
                kept.append({"type": "function_call", "call_id": item.call_id, "name": item.name, "arguments": item.arguments})
        self._turns[len(messages)] = kept
        calls = []
        if call is not None:
            try:
                calls = [{"name": call.name, "arguments": json.loads(call.arguments or "{}")}]
            except json.JSONDecodeError:
                calls = []
        return text.strip(), calls
