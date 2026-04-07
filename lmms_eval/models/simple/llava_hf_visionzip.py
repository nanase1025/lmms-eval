"""
LLaVA-1.5 (HuggingFace) + VisionZip visual token pruning.

VisionZip reference: https://github.com/dvlab-research/VisionZip
Adapted for HuggingFace LlavaForConditionalGeneration (llava-hf/llava-1.5-*-hf)
with transformers >= 5.x.

Key design decisions
--------------------
* No patching of CLIPAttention / CLIPEncoderLayer (new transformers API changed
  those signatures completely).  Instead, forward hooks capture the key-state
  metric and attention weights from the penultimate CLIP encoder layer.

* The new transformers LlavaModel.forward uses masked_scatter, which requires
  the number of image-feature tokens to exactly match the number of <image>
  placeholder IDs in input_ids.  Because VisionZip reduces the count (e.g.
  576 → 63), we override generate_until to:
    1. Tokenise with the VICUNA chat template (fixes a chat-template stripping
       bug in transformers 5.x when <image> is embedded in a string message).
    2. Run the vision tower manually to obtain pruned features.
    3. Project them with multi_modal_projector.
    4. Embed input_ids → inputs_embeds, overwrite the first N_vz image slots,
       and drop the remaining (576 - N_vz) slots.
    5. Call model.generate(inputs_embeds=...) — no pixel_values needed.

Example usage:

    python -m lmms_eval \\
        --model llava_hf_visionzip \\
        --model_args pretrained=llava-hf/llava-1.5-7b-hf,device_map=auto,dominant=54,contextual=10 \\
        --tasks pope \\
        --batch_size 1
"""

import time
from typing import List, Optional

import numpy as np
import PIL
import torch
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.simple.llava_hf import DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, VICUNA_CHAT_TEMPLATE, LlavaHf


# ── Hook-based capture of CLIP intermediate states ────────────────────────────

class _VisionZipCapture:
    """
    Registers forward hooks on the penultimate CLIP encoder layer.
    Captures:
      hidden_states – output of that layer          (for token selection)
      attn_weights  – attention weight matrix        (for dominant token selection)
      metric        – mean key-vector per position   (for contextual merging)
    """

    def __init__(self):
        self.hidden_states: Optional[torch.Tensor] = None
        self.attn_weights: Optional[torch.Tensor] = None
        self.metric: Optional[torch.Tensor] = None
        self._hooks: list = []

    def register(self, encoder_layer):
        attn = encoder_layer.self_attn

        def _attn_pre_hook(module, args, kwargs):
            hs = args[0] if args else kwargs.get("hidden_states")
            if hs is None:
                return
            B, seq, _ = hs.shape
            with torch.no_grad():
                keys = module.k_proj(hs)
                queries = module.q_proj(hs) * module.scale
                num_heads, head_dim = module.num_heads, module.head_dim
                keys_r = keys.view(B, seq, num_heads, head_dim).transpose(1, 2)
                queries_r = queries.view(B, seq, num_heads, head_dim).transpose(1, 2)
                self.metric = keys_r.mean(1).clone()
                attn_w = torch.matmul(queries_r, keys_r.transpose(-2, -1))
                self.attn_weights = torch.softmax(attn_w, dim=-1).clone()

        def _layer_post_hook(module, input, output):
            hs = output if isinstance(output, torch.Tensor) else output[0]
            self.hidden_states = hs.clone()

        self._hooks.append(attn.register_forward_pre_hook(_attn_pre_hook, with_kwargs=True))
        self._hooks.append(encoder_layer.register_forward_hook(_layer_post_hook))

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ── VisionZip token selection (adapted from visionzip/clip_encoder.py) ────────

def _visionzip_token_select(capture: _VisionZipCapture, dominant: int, contextual: int) -> torch.Tensor:
    """
    Returns pruned hidden states of shape [B, dominant + contextual, dim].
    Index 0 is the CLS token (preserved as-is).
    """
    dominant_num = dominant - 1
    hs = capture.hidden_states    # [B, seq, dim]
    aw = capture.attn_weights     # [B, heads, seq, seq]
    metric = capture.metric       # [B, seq, head_dim]
    B, seq, dim = hs.shape

    # Dominant: top-k patches by CLS attention sum
    cls_attn_sum = aw[:, :, 0, 1:].sum(dim=1)              # [B, seq-1]
    topk = cls_attn_sum.topk(dominant_num, dim=1).indices + 1
    all_idx = torch.cat([
        torch.zeros((B, 1), dtype=topk.dtype, device=topk.device), topk
    ], dim=1)
    mask = torch.ones_like(hs[:, :, 0], dtype=torch.bool).scatter_(1, all_idx, False)
    dominant_tokens = hs.masked_select(~mask.unsqueeze(-1)).view(B, dominant_num + 1, dim)

    # Contextual: uniform sample + weighted merge of remaining
    n_rem = seq - (dominant_num + 1)
    m_filt = metric[mask].view(B, n_rem, metric.shape[2])
    h_filt = hs.masked_select(mask.unsqueeze(-1)).view(B, n_rem, dim)
    m_norm = m_filt / m_filt.norm(dim=-1, keepdim=True)

    step = max(1, n_rem // contextual)
    tgt_idx = torch.arange(0, n_rem, step, device=m_norm.device)[:contextual]
    tgt_m = m_norm[:, tgt_idx, :]

    non_tgt = ~torch.isin(torch.arange(n_rem, device=m_norm.device), tgt_idx)
    mrg_m = m_norm[:, non_tgt, :]
    mrg_h = h_filt[:, non_tgt, :]

    sim = torch.bmm(mrg_m, tgt_m.transpose(1, 2))
    assign = torch.zeros(B, mrg_m.shape[1], contextual, dtype=h_filt.dtype, device=m_norm.device)
    assign.scatter_(2, sim.argmax(dim=2).unsqueeze(-1), 1)
    counts = assign.sum(dim=1).clamp(min=1).unsqueeze(-1)
    agg = torch.bmm(assign.transpose(1, 2), mrg_h) / counts
    ctx_tokens = h_filt[:, tgt_idx, :] + agg

    return torch.cat([dominant_tokens, ctx_tokens], dim=1)  # [B, dominant+contextual, dim]


# ── Model ─────────────────────────────────────────────────────────────────────

@register_model("llava_hf_visionzip")
class LlavaHfVisionZip(LlavaHf):
    """LLaVA-1.5 HF + VisionZip visual token pruning."""

    def __init__(self, dominant: int = 54, contextual: int = 10, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dominant = dominant
        self.contextual = contextual
        self._capture = _VisionZipCapture()
        self._setup_hooks()
        eval_logger.info(
            f"VisionZip: dominant={dominant}, contextual={contextual} "
            f"→ {dominant + contextual} visual tokens"
        )

    def _setup_hooks(self):
        encoder_layers = self._model.model.vision_tower.vision_model.encoder.layers
        self._capture.register(encoder_layers[-2])

    # ── core: run VisionZip and return projected image features ───────────────

    @torch.no_grad()
    def _get_visionzip_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Runs the CLIP vision tower (hooks capture intermediate states),
        applies VisionZip token selection, and returns projected features.

        Returns:
            Tensor [B, dominant+contextual-1, lm_dim]
            (CLS is removed, consistent with llava-hf's "default" strategy)
        """
        vision_tower = self._model.model.vision_tower
        # LLaVA-NeXT processor returns pixel_values as [1, n_patches, C, H, W].
        # CLIP expects [batch, C, H, W], so flatten the first two dims.
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.flatten(0, 1)
        vision_tower(pixel_values)  # hooks fire here
        pruned = _visionzip_token_select(self._capture, self.dominant, self.contextual)
        # pruned: [B, dominant+contextual, clip_dim]; index 0 = CLS
        pruned_no_cls = pruned[:, 1:, :]   # [B, dominant+contextual-1, clip_dim]
        image_features = self._model.model.multi_modal_projector(pruned_no_cls)
        # [B, dominant+contextual-1, lm_dim]
        # For LLaVA-NeXT anyres, B = n_patches (e.g. 5 for a 2x2 grid + base).
        # Flatten so callers always get [1, total_vz_tokens, lm_dim].
        image_features = image_features.flatten(0, 1).unsqueeze(0)
        # [1, B*(dominant+contextual-1), lm_dim]
        return image_features

    # ── override generate_until ───────────────────────────────────────────────

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        total_elapsed_time = 0
        total_tokens = 0

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            if len(visuals) == 0:
                task_type = "text"
            elif isinstance(visuals[0], PIL.Image.Image):
                task_type = "image"
            elif isinstance(visuals[0], str):
                task_type = "video"

            gen_kwargs = all_gen_kwargs[0]
            until = [self.tok_decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]

            assert self.batch_size_per_gpu == 1
            context = contexts[0]

            # ── build prompt text ─────────────────────────────────────────────
            if DEFAULT_IMAGE_TOKEN not in context:
                if task_type == "image":
                    img_toks = " ".join([DEFAULT_IMAGE_TOKEN] * len(visuals))
                    context = f"{img_toks}\n{context}"
                elif task_type == "video":
                    img_toks = " ".join([DEFAULT_VIDEO_TOKEN] * len(visuals))
                    context = f"{img_toks}\n{context}"

            messages = [{"role": "user", "content": context}]
            # Force VICUNA template: the built-in llava-hf chat template in
            # transformers 5.x strips <image> when content is a plain string.
            self.tokenizer.chat_template = VICUNA_CHAT_TEMPLATE
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Prompt for doc ID {doc_id[0]}:\n\n{text}\n")

            gen_kwargs.setdefault("max_new_tokens", 1024)
            gen_kwargs.setdefault("temperature", 0)
            gen_kwargs.setdefault("top_p", None)
            gen_kwargs.setdefault("num_beams", 1)
            do_sample = gen_kwargs["temperature"] > 0

            try:
                if task_type == "video":
                    try:
                        visuals = [self.load_video(visuals, self.max_frames_num)]
                    except Exception as e:
                        res.append("")
                        eval_logger.info(f"Error {e} loading video: {visuals}")
                        pbar.update(1)
                        continue

                # ── tokenise (processor expands <image> → 576 image token IDs) ─
                if task_type == "image":
                    inputs = self._image_processor(images=visuals, text=text, return_tensors="pt")
                else:
                    inputs = self._image_processor(videos=visuals, text=text, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self._device, self.model.dtype)
                input_ids = inputs["input_ids"].to(self._device)
                attention_mask = inputs["attention_mask"].to(self._device)

                # ── run VisionZip → projected features ────────────────────────
                image_features = self._get_visionzip_features(pixel_values)
                # [1, n_vz, lm_dim]   n_vz = dominant + contextual - 1
                n_vz = image_features.shape[1]

                # ── build inputs_embeds, replacing image placeholders ──────────
                embed = self._model.model.language_model.get_input_embeddings()
                inputs_embeds = embed(input_ids)              # [1, seq, lm_dim]

                img_tok_id = self._model.config.image_token_id  # 32000
                img_pos = (input_ids[0] == img_tok_id).nonzero(as_tuple=True)[0]
                # img_pos: 576 positions of image tokens in input_ids

                # overwrite first n_vz slots with VisionZip features
                inputs_embeds[0, img_pos[:n_vz]] = image_features[0]

                # drop the remaining (576 - n_vz) image token slots
                keep = torch.ones(input_ids.shape[1], dtype=torch.bool, device=self._device)
                if len(img_pos) > n_vz:
                    keep[img_pos[n_vz:]] = False
                inputs_embeds = inputs_embeds[:, keep, :]
                attention_mask = attention_mask[:, keep]
                prompt_len = inputs_embeds.shape[1]

                # ── generate ──────────────────────────────────────────────────
                start_time = time.time()
                cont = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    do_sample=do_sample,
                    temperature=gen_kwargs["temperature"] if do_sample else None,
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                    pad_token_id=self.eot_token_id,
                    eos_token_id=self.eot_token_id,
                )
                end_time = time.time()
                # when using inputs_embeds, generate() returns only new tokens
                # (no prompt prefix to strip)
                total_elapsed_time += end_time - start_time
                total_tokens += cont.shape[1]

            except Exception as e:
                import traceback
                eval_logger.error(f"Error {e} in generating\n{traceback.format_exc()}")
                cont = torch.tensor([[]], dtype=torch.long, device=self._device)

            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Generated text for doc ID {doc_id[0]}:\n\n{text_outputs}\n")

            res.append(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        log_metrics(
            total_gen_tokens=total_tokens,
            total_elapsed_time=total_elapsed_time,
            avg_speed=total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0,
            additional_metrics={"rank": self.rank},
        )
        return res
