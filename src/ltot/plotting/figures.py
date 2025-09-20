import pandas as pd

def main_equal_compute_figure(df: pd.DataFrame) -> str:
    lines = []
    for (t, mth), g in df.groupby(["task","method"]):
        score = float(g["score"].mean())
        lines.append(f"{t}:{mth}:{score:.3f}")
    text = "\\n".join(lines)
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="400"><text x="10" y="20" font-size="14">{text}</text></svg>'
    return svg
