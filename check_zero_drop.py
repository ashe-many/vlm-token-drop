"""Check: with 0% drop, our loop must give the same answer as unpatched mlx-vlm generate."""
import sys

from mlx_vlm import generate
from mlx_vlm.prompt_utils import apply_chat_template

from pope import load_subset
from token_drop import MAX_NEW_TOKENS, POPE_SUFFIX, TokenDropper

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10

td = TokenDropper()
same = 0
for i, row in load_subset().head(N).iterrows():
    image = row["image"]
    ids, feats = td.encode(image, row["question"])
    ours, _ = td.answer(ids, feats, drop=0.0)
    prompt = apply_chat_template(td.processor, td.config, row["question"] + POPE_SUFFIX, num_images=1)
    ref = generate(td.model, td.processor, prompt, image=[image], max_tokens=MAX_NEW_TOKENS,
                   temperature=0.0, verbose=False).text.strip()
    ok = ours == ref
    same += ok
    print(f"{i:3d} {'OK ' if ok else 'DIFF'} ours={ours!r} ref={ref!r}", flush=True)
print(f"{same}/{N} identical")
