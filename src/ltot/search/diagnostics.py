from typing import Callable, Tuple, List
from .baselines import cot_prompt

class _Branch:
    __slots=("text","score")
    def __init__(self, text:str, score:float=0.0):
        self.text=text; self.score=float(score)

def sh_only_lateralization(model, task: str, q: str, budget_tokens: int,
                           scorer: Callable, initial_width: int=128,
                           eta:int=4, b0:int=1, micro_beam:int=3) -> Tuple[str,int,int]:
    tokens_spent, expansions = 0, 0
    laterals: List[_Branch] = []
    for _ in range(int(initial_width)):
        outs, toks = model.generate([f"{cot_prompt(q)}\nConsider a logically different path:"], max_tokens=max(24, budget_tokens//96), temperature=0.8, top_p=0.95)
        tokens_spent += int(toks[0]); expansions += 1
        s, vcost = scorer(model, task, outs[0]); tokens_spent += int(vcost)
        laterals.append(_Branch(outs[0], s))

    r = 0
    while laterals and tokens_spent < budget_tokens:
        Qr = max(1, len(laterals)//max(2, eta))
        laterals.sort(key=lambda b: b.score, reverse=True)
        keep = laterals[:Qr]
        next_laterals: List[_Branch] = []
        step_tokens = max(24, int(b0 * (eta ** r) * max(16, budget_tokens//(initial_width*8))))
        for br in keep:
            outs, toks = model.generate([br.text + "\nContinue one step:" for _ in range(micro_beam)], max_tokens=step_tokens, temperature=0.8, top_p=0.95)
            tokens_spent += int(sum(toks)); expansions += len(outs)
            scores = []
            for o in outs:
                s, vcost = scorer(model, task, o); tokens_spent += int(vcost)
                scores.append(float(s))
            br.text = (br.text + "\n" + outs[0]).strip()
            br.score = sum(sorted(scores)[-micro_beam:])/float(micro_beam)
            next_laterals.append(br)
        laterals = next_laterals
        r += 1
        if len(laterals) <= 1: break

    best = max(laterals, key=lambda b: b.score) if laterals else _Branch("")
    return best.text, tokens_spent, expansions

def sh_on_mainlines(model, task: str, q: str, budget_tokens: int,
                    scorer: Callable, beam:int=8, eta:int=4, b0:int=1) -> Tuple[str,int,int]:
    tokens_spent, expansions = 0, 0
    main: List[_Branch] = []
    outs, toks = model.generate([cot_prompt(q) for _ in range(beam)], max_tokens=max(32, budget_tokens//32), temperature=0.7, top_p=0.95)
    tokens_spent += int(sum(toks)); expansions += len(outs)
    for o in outs:
        s, vcost = scorer(model, task, o); tokens_spent += int(vcost)
        main.append(_Branch(o, s))
    r = 0
    while main and tokens_spent < budget_tokens:
        Qr = max(1, len(main)//max(2, eta))
        main.sort(key=lambda b: b.score, reverse=True)
        keep = main[:Qr]; next_main=[]
        step_tokens = max(24, int(b0 * (eta ** r) * max(16, budget_tokens//(beam*16))))
        for br in keep:
            out, toks = model.generate([br.text + "\nContinue:"], max_tokens=step_tokens, temperature=0.7, top_p=0.95)
            tokens_spent += int(toks[0]); expansions += 1
            s, vcost = scorer(model, task, out[0]); tokens_spent += int(vcost)
            br.text = (br.text + "\n" + out[0]).strip(); br.score = float(s)
            next_main.append(br)
        main = next_main
        r += 1
        if len(main) <= 1: break
    best = max(main, key=lambda b: b.score) if main else _Branch("")
    return best.text, tokens_spent, expansions
