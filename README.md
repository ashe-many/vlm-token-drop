# Image-token dropping on POPE, on an 8 GB Apple M1

A reproduction of the motivation experiment in
[Seeing without Looking: Do Vision-Language Benchmarks Really Test Vision?](https://arxiv.org/abs/2605.22903)
(arXiv 2605.22903, App. A.4), with Qwen3-VL-4B running in 4-bit on MLX.

**Explainer page with the decisions, diagrams and results:** https://ashe-many.github.io/vlm-token-drop/

## Result

Removing image tokens after the vision encoder and merger hardly changes POPE accuracy, up to 75%.
With no image tokens, the model answers "No" to almost every question.

| Image tokens removed | Kept (mean) | Accuracy, all 300 | Accuracy, answer is Yes (147) | Model says Yes |
|---|---|---|---|---|
| 0%   | 269.3 | 91.3% | 85.7% | 43.7% |
| 25%  | 202.0 | 90.0% | 83.0% | 42.3% |
| 50%  | 134.7 | 90.7% | 84.4% | 43.0% |
| 75%  | 67.3  | 87.3% | 76.2% | 38.3% |
| 100% | 0     | 50.7% | 0.7%  | 1.0%  |

## New: which path carries the image?

Qwen3-VL sends image data to the LLM on two paths: the image tokens in the input (path 1), and the
deepstack copies from vision layers 6, 12 and 18, added after LLM layers 1–3 (path 2).
The paper does not examine the two paths. `run_deepstack.py` tests them on the same 300 questions.

| Condition | Accuracy |
|---|---|
| Both paths (normal model), 0% removed | 91.3% |
| Path 1 only (deepstack off), 0% removed | 90.3% |
| Path 1 only, 75% removed | 86.7% |
| Path 2 only (input image embeddings set to zero) | 75.3% |
| No image data | 50.7% |

- Turning deepstack off changes at most 7 of 300 answers (McNemar p ≥ 0.125 at every drop rate):
  deepstack does not cause the flat curve.
- Deepstack alone reaches 75.3% vs. 50.7% with no image data (p < 0.001): it carries real image
  information, but adds almost nothing when the input tokens are present.
- The 4-point drop at 75% removal (91.3% → 87.3%) is small but real (p = 0.012).

## How it works

`token_drop.py` runs the vision tower once per image, then removes a random fraction of the image
tokens from four places at once:

1. the input embeddings (text + image),
2. the three deepstack copies that Qwen3-VL adds after LLM layers 1–3,
3. the M-RoPE position ids (kept tokens keep their original positions),
4. the image-position mask.

It then decodes greedily (max 6 new tokens) with its own short loop.

**Check:** at 0% drop, the first-token logits equal those of the unpatched mlx-vlm model
(max |difference| = 0 on 5 examples, `check_logits.py`), and the answers match `mlx_vlm.generate`
(`check_zero_drop.py`).

## Differences from the paper

- 4-bit MLX weights on an Apple M1, not bf16 with Hugging Face transformers on an RTX 4090.
- The paper does not say which 300 POPE questions it used. Here: 300 drawn with seed 0 from the
  `lmms-lab/POPE` random split (147 have the answer Yes; the paper's set has 149).
- One random drop pattern per question (seed = question index).
- Positions of the kept tokens: not stated in the paper; here they keep their original M-RoPE positions.
- 100% drop is an extra point, not in the paper.

## Run

```bash
uv venv --python 3.12 && uv pip install -r requirements.txt
hf download mlx-community/Qwen3-VL-4B-Instruct-4bit
hf download lmms-lab/POPE --repo-type dataset --include "Full/random-00000-of-00001.parquet" --local-dir data/pope
python check_logits.py      # 0% drop == unpatched model
python run_pope.py          # ~2 h on an 8 GB M1; resumable
python run_deepstack.py     # deepstack test, ~2-3 h
python make_chart.py
```

Outputs: `results/answers.jsonl` (every answer), `results/summary.json`, `results/chart.html`,
`results/deepstack_answers.jsonl`, `results/deepstack_summary.json`.

Figure 2 in the page is from the paper, CC BY 4.0.
