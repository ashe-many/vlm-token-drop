"""Run POPE at several image-token drop rates. Writes results/answers.jsonl and results/summary.json."""
import json
import time
from pathlib import Path

from pope import load_subset
from token_drop import MODEL, TokenDropper, to_yes_no

DROPS = [0.0, 0.25, 0.5, 0.75, 1.0]  # paper: 0-0.75; 1.0 (no image tokens) is our extra point
OUT = Path(__file__).parent / "results"


def main():
    OUT.mkdir(exist_ok=True)
    df = load_subset()
    td = TokenDropper()
    done = set()
    answers_path = OUT / "answers.jsonl"
    if answers_path.exists():  # resume
        done = {(r["idx"], r["drop"]) for r in map(json.loads, answers_path.open())}
    t0 = time.time()
    with answers_path.open("a") as f:
        for i, row in df.iterrows():
            todo = [d for d in DROPS if (i, d) not in done]
            if not todo:
                continue
            ids, feats = td.encode(row["image"], row["question"])
            for d in todo:
                text, kept = td.answer(ids, feats, drop=d, seed=i)
                f.write(json.dumps({"idx": i, "question_id": str(row["question_id"]),
                                    "question": row["question"], "label": row["answer"],
                                    "drop": d, "kept_tokens": kept, "output": text,
                                    "pred": to_yes_no(text)}) + "\n")
            f.flush()
            print(f"{i + 1}/{len(df)}  {time.time() - t0:.0f}s", flush=True)
    summarize(answers_path)


def summarize(answers_path: Path):
    rows = [json.loads(l) for l in answers_path.open()]
    summary = {"model": MODEL, "n_examples": len({r["idx"] for r in rows}), "drops": []}
    for d in DROPS:
        rs = [r for r in rows if r["drop"] == d]
        pos = [r for r in rs if r["label"] == "yes"]
        summary["drops"].append({
            "drop": d,
            "n": len(rs),
            "mean_kept_tokens": round(sum(r["kept_tokens"] for r in rs) / max(len(rs), 1), 1),
            "accuracy": round(sum(r["pred"] == r["label"] for r in rs) / max(len(rs), 1), 4),
            "accuracy_yes_subset": round(sum(r["pred"] == "yes" for r in pos) / max(len(pos), 1), 4),
            "n_yes_subset": len(pos),
            "yes_rate": round(sum(r["pred"] == "yes" for r in rs) / max(len(rs), 1), 4),
            "unparsed": sum(r["pred"] == "other" for r in rs),
        })
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
