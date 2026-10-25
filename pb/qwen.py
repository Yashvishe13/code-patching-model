"""Qwen3-1.7B as a policy (act(messages) -> (content, calls)), using its native <tool_call> format."""
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pb.env import TOOLS

CHAT_TOOLS = [{"type": "function", "function": t} for t in TOOLS]
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# Fallback for a malformed call: a JSON object ending at a lone </tool_call> with the opening tag
# dropped (the 1.7B model does this often enough to lose whole episodes to a bogus "no tool call").
_ORPHAN_RE = re.compile(r"(\{.*?\})\s*</tool_call>", re.DOTALL)


def _extract_objs(text):
    """Tool-call JSON objects in the model's output, tolerant of a missing opening <tool_call> tag."""
    blocks = TOOL_CALL_RE.findall(text)
    if not blocks:  # no well-formed pair: try orphaned closing tags
        blocks = _ORPHAN_RE.findall(text)
    return blocks


def pick_dtype():
    if torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    # Turing (2080 Ti) has no FlashAttention; transformers' SDPA then uses enable_gqa, for which there
    # is no Turing kernel, so it falls back to the math kernel that materialises the full N*N attention
    # matrix (OOM at long context). Forcing repeat_kv keeps the memory-efficient kernel.
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa = lambda *a, **k: False
    return torch.float16


def parse_tool_calls(text):
    calls = []
    for block in _extract_objs(text):
        try:
            obj = json.loads(block)
            calls.append({"name": obj["name"], "arguments": obj.get("arguments") or {}})
        except (json.JSONDecodeError, KeyError):
            pass
    return text.split("<tool_call>", 1)[0].strip(), calls


class Qwen:
    def __init__(self, model_name="Qwen/Qwen3-1.7B", adapter=None, max_context=24000, max_new_tokens=1024):
        dtype = pick_dtype()  # also installs the Turing SDPA fix
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map="cuda",
                                                          attn_implementation="sdpa")
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter).merge_and_unload()
        self.model.eval()
        self.max_context, self.max_new_tokens = max_context, max_new_tokens

    @torch.no_grad()
    def act(self, messages):
        enc = self.tok.apply_chat_template(messages, tools=CHAT_TOOLS, add_generation_prompt=True,
                                           enable_thinking=False, return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to(self.model.device)
        if ids.shape[1] + self.max_new_tokens > self.max_context:
            return "", []  # context full -> no tool call -> episode ends
        out = self.model.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=True,
                                  temperature=0.7, top_p=0.8, top_k=20, pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        del out, ids, enc
        torch.cuda.empty_cache()
        return parse_tool_calls(text)
