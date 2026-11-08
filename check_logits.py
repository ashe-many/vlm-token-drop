"""Check: with 0% drop, first-token logits of our path equal the unpatched model's forward pass."""
import mlx.core as mx
import numpy as np

from mlx_vlm.models.cache import make_prompt_cache
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

from pope import load_subset
from token_drop import POPE_SUFFIX, TokenDropper

td = TokenDropper()
lm = td.model.language_model
for i, row in load_subset().head(5).iterrows():
    ids, f = td.encode(row["image"], row["question"])
    ours = lm(ids, inputs_embeds=f.inputs_embeds, cache=make_prompt_cache(lm), position_ids=f.position_ids,
              visual_pos_masks=f.visual_pos_masks,
              deepstack_visual_embeds=[d for d in f.deepstack_visual_embeds]).logits[0, -1]
    prompt = apply_chat_template(td.processor, td.config, row["question"] + POPE_SUFFIX, num_images=1)
    inp = prepare_inputs(td.processor, images=[row["image"]], prompts=prompt,
                         image_token_index=td.config.image_token_index)
    ref = td.model(inp["input_ids"], inp["pixel_values"], cache=make_prompt_cache(lm),
                   image_grid_thw=inp["image_grid_thw"], mask=inp.get("attention_mask")).logits[0, -1]
    d = float(mx.abs(ours.astype(mx.float32) - ref.astype(mx.float32)).max())
    print(f"{i}: image tokens={int(np.array(f.visual_pos_masks).sum())}  max |logit diff| = {d:.4g}  "
          f"top1 same={int(mx.argmax(ours)) == int(mx.argmax(ref))}")
