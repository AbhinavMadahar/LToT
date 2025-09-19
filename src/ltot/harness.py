
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os, json, time, math, base64, io, random
import numpy as np
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

from ltot.controller import LToTController, BranchEnvelope, PlateauDetector, LateralRacing
from ltot.evaluators import MathVerifier, QAGate

def load_local_model(path: str, dtype: str="bfloat16", max_model_len: int=8192):
    torch_dtype = torch.bfloat16 if dtype=="bfloat16" else torch.float16
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch_dtype, device_map="auto",
        local_files_only=True, trust_remote_code=True
    )
    pipe = TextGenerationPipeline(model=model, tokenizer=tok, device=0 if torch.cuda.is_available() else -1)
    return pipe

@dataclass
class HFModelHarness:
    name: str
    path: str
    dtype: str = "bfloat16"
    max_len: int = 8192
    pipe: Optional[TextGenerationPipeline] = None

    def ensure(self):
        if self.pipe is None:
            self.pipe = load_local_model(self.path, self.dtype, self.max_len)

    def generate(self, prompt: str, max_new_tokens: int=256, temperature: float=0.7, top_p: float=0.95) -> str:
        self.ensure()
        out = self.pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=top_p, eos_token_id=self.pipe.tokenizer.eos_token_id)
        return out[0]["generated_text"]

def dataset_iter(task_cfg: Dict[str, Any]):
    src = task_cfg["source"]
    if src.startswith("file:"):
        path = Path(src[5:])
        if path.exists():
            with path.open() as f:
                for line in f:
                    obj = json.loads(line)
                    yield obj
        else:
            for _ in range(3):
                yield {"question": "2+2=?", "answer": "4"}
    else:
        for _ in range(3):
            yield {"question": "2+2=?", "answer": "4"}

def make_figure_png_base64():
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.linspace(0, 1, 50)
    y = np.sin(2*np.pi*x)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title("Placeholder curve")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def run_all(config: Dict[str, Any], artifact_path: str, run_id: Optional[str]=None):
    run_id = run_id or str(int(time.time()))
    models = config["models"]
    budgets = config["budgets"]
    tasks = config["benchmarks"]
    methods = config["methods"]

    os.makedirs("results", exist_ok=True)
    tabdir = Path("results/tables"); tabdir.mkdir(parents=True, exist_ok=True)

    with open(artifact_path, "w") as af:
        for model_id, m in models.items():
            h = HFModelHarness(name=m["name"], path=os.path.expandvars(m["path"]), dtype=m.get("dtype","bfloat16"), max_len=m.get("max_model_len",8192))
            for task_id, tc in tasks.items():
                for budget in ("low","med","high"):
                    for method in methods:
                        metrics = {"Success@1": 0.0, "Pass@1": 0.0, "time_to_first_hit": None, "false_promotion_rate": None}
                        fig = {"name":"placeholder", "png_base64": make_figure_png_base64(), "caption": f"{task_id}/{method} example plot"}
                        rec = {"run_id": run_id, "task_id": task_id, "model_id": model_id, "budget": budget, "method": method, "metrics": metrics, "tables": {}, "figures": [fig], "notes": "Populated structure; run the real pipeline on ARC to fill actual numbers."}
                        af.write(json.dumps(rec) + "\n")
