import os, requests
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_MODEL_CACHE = {}

class OpenAICompatLM:
    def __init__(self, model_id: str, base: str, api_key: str | None = None):
        self.model_id = model_id
        self.base = base.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key or 'dummy'}"}

    def generate(self, prompts: List[str], max_tokens=128, temperature=0.7, top_p=0.95) -> Tuple[List[str], List[int]]:
        outs, toks = [], []
        for p in prompts:
            payload = {
                "model": self.model_id,
                "prompt": p,
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
            }
            r = requests.post(f"{self.base}/v1/completions", json=payload, headers=self.headers, timeout=600)
            j = r.json()
            outs.append((j["choices"][0].get("text") or "").strip())
            toks.append(int(j.get("usage", {}).get("completion_tokens", 0)))
        return outs, toks

def make_llm(model_name: str, **local_kwargs):
    """
    Choose backend by env:
      LTOT_BACKEND=local (default)  -> LocalLM(hf_model_id(...))
      LTOT_BACKEND=openai           -> OpenAICompatLM at OPENAI_API_BASE
    """
    backend = os.getenv("LTOT_BACKEND", "local").lower()
    if backend == "openai":
        base = os.environ["OPENAI_API_BASE"]       # e.g., http://<ip>:8000
        key  = os.getenv("OPENAI_API_KEY")         # vLLM usually ignores it; pass dummy if unset
        return OpenAICompatLM(hf_model_id(model_name), base, api_key=key)
    else:
        # Pass through dtype/kwargs so you can run 70B across 4 GPUs with device_map="auto"
        return LocalLM(hf_model_id(model_name), **local_kwargs)

def hf_model_id(name: str) -> str:
    # Direct mapping; adjust to your local IDs if needed
    return {
        "llama-3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "mixtral-8x7b-instruct": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "llama-3.1-70b-instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    }.get(name, name)

class LocalLM:
    def __init__(self, model_id: str, dtype: str = "bfloat16", device: str = None):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=getattr(torch, dtype) if hasattr(torch, dtype) else torch.float16,
            device_map="auto"
        )

    def _toklen(self, s: str) -> int:
        return len(self.tokenizer.encode(s, add_special_tokens=False))

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> Tuple[List[str], List[int]]:
        outputs, toks = [], []
        for p in prompts:
            ipt = self.tokenizer(p, return_tensors="pt").to(self.model.device)
            out = self.model.generate(
                **ipt,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            tail = text[len(self.tokenizer.decode(ipt["input_ids"][0], skip_special_tokens=True)):]
            outputs.append(tail.strip())
            toks.append(self._toklen(tail))
            prompt_tok = int(ipt["input_ids"].shape[1])
            toks.append(prompt_tok + self._toklen(tail))
        return outputs, toks
