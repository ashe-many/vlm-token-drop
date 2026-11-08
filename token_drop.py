"""Image-token dropping for Qwen3-VL on Apple silicon (MLX).

Reproduces the motivation experiment of "Seeing without Looking" (arXiv 2605.22903, App. A.4):
randomly remove a fraction r of the image tokens after the vision encoder + merger, then let the
LLM answer POPE yes/no questions with greedy decoding (max 6 new tokens).

The same tokens are removed from:
  * the input embeddings (text + image),
  * the deepstack copies added after LLM layers 1-3,
  * the M-RoPE position ids (kept tokens keep their original positions; our assumption).
"""
import numpy as np
import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.models.cache import make_prompt_cache
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
MAX_NEW_TOKENS = 6
POPE_SUFFIX = " Answer the question using a single word or phrase."


class TokenDropper:
    def __init__(self, repo: str = MODEL):
        self.model, self.processor = load(repo)
        self.config = self.model.config
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.eos = set(self.config.eos_token_id if isinstance(self.config.eos_token_id, list)
                       else [self.config.eos_token_id])
        self.eos.add(self.tokenizer.eos_token_id)

    def encode(self, image, question: str):
        """Run the vision tower once. Returns everything needed for any drop rate."""
        prompt = apply_chat_template(self.processor, self.config, question + POPE_SUFFIX, num_images=1)
        inputs = prepare_inputs(self.processor, images=[image], prompts=prompt,
                                image_token_index=self.config.image_token_index)
        input_ids = inputs["input_ids"]
        feats = self.model.get_input_embeddings(
            input_ids, inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"], mask=inputs.get("attention_mask"))
        mx.eval(feats.inputs_embeds, feats.position_ids, feats.deepstack_visual_embeds)
        return input_ids, feats

    def answer(self, input_ids, feats, drop: float, seed: int = 0,
               deepstack: bool = True, zero_input_image: bool = False) -> tuple[str, int]:
        """Greedy answer after removing a random fraction `drop` of the image tokens.

        deepstack=False: do not add the deepstack copies after LLM layers 1-3.
        zero_input_image=True: set the image-token input embeddings to zero, so image data
        can enter only through the deepstack copies.
        """
        img_pos = np.flatnonzero(np.array(feats.visual_pos_masks[0]))
        n_img = len(img_pos)
        n_drop = int(round(drop * n_img))
        rng = np.random.default_rng(seed)
        dropped = rng.choice(n_img, size=n_drop, replace=False) if n_drop else np.array([], int)
        keep_img = np.setdiff1d(np.arange(n_img), dropped)  # indices into the image tokens

        L = input_ids.shape[1]
        keep_seq = np.setdiff1d(np.arange(L), img_pos[dropped])  # indices into the full sequence
        ks = mx.array(keep_seq, dtype=mx.uint32)

        ids = input_ids[:, ks]
        embeds = feats.inputs_embeds[:, ks]
        pos = feats.position_ids[:, :, ks]  # (3, 1, L'): original M-RoPE positions
        vis_mask = feats.visual_pos_masks[:, ks]
        ki = mx.array(keep_img, dtype=mx.uint32)
        ds = [d[ki] for d in feats.deepstack_visual_embeds] if deepstack else None  # 3 x (n_kept, 2560)
        if zero_input_image:
            embeds = mx.where(vis_mask[..., None], mx.zeros_like(embeds), embeds)

        lm = self.model.language_model
        cache = make_prompt_cache(lm)
        out = lm(ids, inputs_embeds=embeds, cache=cache, position_ids=pos,
                 visual_pos_masks=vis_mask, deepstack_visual_embeds=ds)
        next_pos = int(feats.position_ids.max().item()) + 1

        tokens = []
        for step in range(MAX_NEW_TOKENS):
            tok = int(mx.argmax(out.logits[0, -1]).item())
            if tok in self.eos:
                break
            tokens.append(tok)
            p = mx.full((3, 1, 1), next_pos + step, dtype=pos.dtype)
            out = lm(mx.array([[tok]]), cache=cache, position_ids=p)
        return self.tokenizer.decode(tokens).strip(), len(keep_img)


def to_yes_no(text: str) -> str:
    t = text.strip().lower()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    return "other"
