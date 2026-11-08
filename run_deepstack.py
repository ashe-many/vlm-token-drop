"""Deepstack test: which path carries the image data into the LLM?

Qwen3-VL sends image data along two paths: the image tokens in the input, and the deepstack
copies added after LLM layers 1-3. Conditions (same 300 questions and same dropped tokens as run_pope.py):
  no_deepstack   : input image tokens only (drop 0/25/50/75%), deepstack copies not added
  deepstack_only : input image embeddings set to zero, deepstack copies added (no drop)
Writes results/deepstack_answers.jsonl and results/deepstack_summary.json.
"""
import json
import time
from pathlib import Path

from pope import load_subset
from token_drop import MODEL, TokenDropper, to_yes_no

CONDS = [("no_deepstack", d, False, False) for d in (0.0, 0.25, 0.5, 0.75)] + [
    ("deepstack_only", 0.0, True, True)]
OUT = Path(__file__).parent / "results"


def main():
    df = load_subset()
    td = TokenDropper()
    path = OUT / "deepstack_answers.jsonl"
    done = {(r["idx"], r["condition"], r["drop"]) for r in map(json.loads, path.open())} if path.exists() else set()
    t0 = time.time()
    with path.open("a") as f:
        for i, row in df.iterrows():
            todo = [c for c in CONDS if (i, c[0], c[1]) not in done]
            if not todo:
                continue
            ids, feats = td.encode(row["image"], row["question"])
            for name, d, ds, zero in todo:
                text, kept = td.answer(ids, feats, drop=d, seed=i, deepstack=ds, zero_input_image=zero)
                f.write(json.dumps({"idx": i, "question_id": str(row["question_id"]), "label": row["answer"],
                                    "condition": name, "drop": d, "kept_tokens": kept, "output": text,
                                    "pred": to_yes_no(text)}) + "\n")
            f.flush()
            print(f"{i + 1}/{len(df)}  {time.time() - t0:.0f}s", flush=True)
    summarize()


def stats(rs):
    pos = [r for r in rs if r["label"] == "yes"]
    n = max(len(rs), 1)
    return {"n": len(rs),
            "accuracy": round(sum(r["pred"] == r["label"] for r in rs) / n, 4),
            "accuracy_yes_subset": round(sum(r["pred"] == "yes" for r in pos) / max(len(pos), 1), 4),
            "yes_rate": round(sum(r["pred"] == "yes" for r in rs) / n, 4),
            "unparsed": sum(r["pred"] == "other" for r in rs)}


def summarize():
    base = [json.loads(l) for l in (OUT / "answers.jsonl").open()]
    new = [json.loads(l) for l in (OUT / "deepstack_answers.jsonl").open()]
    rows = [{"condition": "both_paths", "drop": d, **stats([r for r in base if r["drop"] == d])}
            for d in (0.0, 0.25, 0.5, 0.75, 1.0)]
    rows += [{"condition": c, "drop": d, **stats([r for r in new if r["condition"] == c and r["drop"] == d])}
             for c, d, _, _ in CONDS]
    out = {"model": MODEL, "rows": rows}
    (OUT / "deepstack_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
