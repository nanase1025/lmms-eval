import types
from typing import Optional

import torch
import torch.nn as nn


def _clip_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    causal_attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
    **kwargs,
):
    del kwargs
    batch_size, seq_length, embed_dim = hidden_states.size()

    queries = self.q_proj(hidden_states)
    keys = self.k_proj(hidden_states)
    values = self.v_proj(hidden_states)

    queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
    keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
    values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

    raw_key_states = keys.clone()
    attn_weights = torch.matmul(queries * self.scale, keys.transpose(-2, -1))

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    if causal_attention_mask is not None:
        attn_weights = attn_weights + causal_attention_mask

    attn_weights_reshaped = nn.functional.softmax(attn_weights, dim=-1)
    self._visionzip_last_attn_weights = attn_weights_reshaped
    attn_probs = nn.functional.dropout(attn_weights_reshaped, p=self.dropout, training=self.training)
    attn_output = torch.matmul(attn_probs, values)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_length, embed_dim)
    attn_output = self.out_proj(attn_output)

    returned_attn = attn_weights_reshaped if output_attentions else None
    return attn_output, returned_attn, raw_key_states.mean(dim=1)


def _clip_encoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    causal_attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
    **kwargs,
):
    del kwargs
    residual = hidden_states
    hidden_states = self.layer_norm1(hidden_states)

    hidden_states, attn_weights, metric = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
    )

    hidden_states = residual + hidden_states

    self.metric = metric
    self._visionzip_last_attn_weights = getattr(self.self_attn, "_visionzip_last_attn_weights", None)

    residual = hidden_states
    hidden_states = self.layer_norm2(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (attn_weights,)
    return outputs


def _visionzip_select(hidden_states, attn_weights, metric, dominant_num, contextual_num):
    cls_attention = attn_weights[:, :, 0, 1:]
    cls_attention_sum = cls_attention.sum(dim=1)
    patch_hidden_states = hidden_states[:, 1:, :]
    patch_metric = metric[:, 1:, :]
    patch_count = patch_hidden_states.shape[1]

    dominant_num = min(dominant_num, patch_count)
    topk_indices = cls_attention_sum.topk(dominant_num, dim=1).indices

    mask = torch.ones(
        (patch_hidden_states.shape[0], patch_count),
        dtype=torch.bool,
        device=patch_hidden_states.device,
    ).scatter_(1, topk_indices, False)
    dominant_tokens = patch_hidden_states.masked_select(~mask.unsqueeze(-1)).view(
        patch_hidden_states.shape[0], dominant_num, patch_hidden_states.shape[2]
    )

    remaining_count = patch_count - dominant_num
    metric_filtered = patch_metric[mask].view(patch_hidden_states.shape[0], remaining_count, patch_metric.shape[2])
    hidden_states_filtered = patch_hidden_states.masked_select(mask.unsqueeze(-1)).view(
        patch_hidden_states.shape[0], remaining_count, patch_hidden_states.shape[2]
    )

    contextual_num = min(contextual_num, metric_filtered.shape[1])
    if contextual_num <= 0:
        return dominant_tokens, topk_indices

    metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    step = max(1, metric_normalized.shape[1] // contextual_num)
    token_index_range = torch.arange(metric_normalized.shape[1], device=metric_normalized.device)
    target_indices = token_index_range[::step][:contextual_num]
    target_tokens = metric_normalized[:, target_indices, :]

    non_target_mask = ~torch.isin(token_index_range, target_indices)
    tokens_to_merge = metric_normalized[:, non_target_mask, :]
    hidden_to_merge = hidden_states_filtered[:, non_target_mask, :]
    target_hidden = hidden_states_filtered[:, target_indices, :]

    if tokens_to_merge.shape[1] == 0:
        contextual_tokens = target_hidden
    else:
        similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
        assign_one_hot = torch.zeros(
            tokens_to_merge.shape[0],
            tokens_to_merge.shape[1],
            contextual_num,
            dtype=hidden_states_filtered.dtype,
            device=metric_normalized.device,
        )
        assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
        counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
        aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts
        contextual_tokens = target_hidden + aggregated_hidden

    hidden_states_save = torch.cat([dominant_tokens, contextual_tokens], dim=1)
    return hidden_states_save, topk_indices


def _get_image_features_visionzip(self, pixel_values, vision_feature_layer, vision_feature_select_strategy, **kwargs):
    kwargs.pop("image_sizes", None)
    image_outputs = self.vision_tower(pixel_values, output_hidden_states=True, **kwargs)
    hidden_states = image_outputs.hidden_states[-2]
    target_layer = self.vision_tower.vision_model.encoder.layers[-2]
    attn_weights = getattr(target_layer, "_visionzip_last_attn_weights", None)
    metric = getattr(target_layer, "metric", None)
    if attn_weights is None or metric is None:
        raise RuntimeError("VisionZip statistics were not captured from the CLIP penultimate layer.")
    info = self.vision_tower._visionzip_info

    compressed_hidden, _ = _visionzip_select(
        hidden_states=hidden_states,
        attn_weights=attn_weights,
        metric=metric,
        dominant_num=info["dominant"],
        contextual_num=info["contextual"],
    )
    image_features = self.multi_modal_projector(compressed_hidden)
    return image_features


def _merge_inputs_with_image_features(
    self,
    input_ids,
    inputs_embeds,
    image_features,
    attention_mask=None,
    labels=None,
):
    def _as_2d(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 1:
            return tensor.unsqueeze(0)
        if tensor.ndim == 2:
            return tensor
        return tensor.reshape(-1, tensor.shape[-1])

    if isinstance(image_features, torch.Tensor):
        if image_features.ndim == 2:
            image_features = image_features.unsqueeze(0)
        elif image_features.ndim > 3:
            image_features = image_features.reshape(-1, image_features.shape[-2], image_features.shape[-1])

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()

    if labels is None:
        labels = torch.full_like(input_ids, -100)

    image_token_index = self.config.image_token_index

    batch_input_ids = [cur_input_ids[cur_mask] for cur_input_ids, cur_mask in zip(input_ids, attention_mask)]
    batch_labels = [cur_labels[cur_mask] for cur_labels, cur_mask in zip(labels, attention_mask)]

    new_input_embeds = []
    new_labels = []
    cur_image_idx = 0

    for cur_input_ids, cur_labels in zip(batch_input_ids, batch_labels):
        image_mask = cur_input_ids == image_token_index
        image_spans = []
        span_start = None
        for token_idx, is_image_token in enumerate(image_mask.tolist()):
            if is_image_token and span_start is None:
                span_start = token_idx
            elif not is_image_token and span_start is not None:
                image_spans.append((span_start, token_idx))
                span_start = None
        if span_start is not None:
            image_spans.append((span_start, cur_input_ids.shape[0]))

        cur_chunks = []
        cur_label_chunks = []
        prev_end = 0
        for span_start, span_end in image_spans:
            cur_chunks.append(cur_input_ids[prev_end:span_start])
            cur_label_chunks.append(cur_labels[prev_end:span_start])
            prev_end = span_end
        cur_chunks.append(cur_input_ids[prev_end:])
        cur_label_chunks.append(cur_labels[prev_end:])

        split_sizes = [chunk.shape[0] for chunk in cur_chunks]
        if sum(split_sizes) > 0:
            text_embeds = self.get_input_embeddings()(torch.cat(cur_chunks))
            text_embeds = list(torch.split(text_embeds, split_sizes, dim=0))
        else:
            text_embeds = [inputs_embeds.new_zeros((0, inputs_embeds.shape[-1])) for _ in cur_chunks]

        merged_embeds = []
        merged_labels = []
        for idx in range(len(cur_chunks)):
            merged_embeds.append(_as_2d(text_embeds[idx]))
            merged_labels.append(cur_label_chunks[idx])
            if idx < len(image_spans):
                if cur_image_idx >= image_features.shape[0]:
                    raise ValueError(
                        f"Image feature count mismatch after compression: spans={len(image_spans)}, available_features={image_features.shape[0]}"
                    )
                cur_image_features = image_features[cur_image_idx]
                cur_image_idx += 1
                cur_image_features = _as_2d(cur_image_features)
                merged_embeds.append(cur_image_features)
                merged_labels.append(
                    torch.full(
                        (cur_image_features.shape[0],),
                        -100,
                        device=cur_labels.device,
                        dtype=cur_labels.dtype,
                    )
                )

        merged_embeds = [_as_2d(embed) for embed in merged_embeds]
        new_input_embeds.append(torch.cat(merged_embeds, dim=0))
        new_labels.append(torch.cat(merged_labels, dim=0))

    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)
    embed_dim = new_input_embeds[0].shape[1]

    padded_embeds = []
    padded_labels = torch.full((batch_size, max_len), -100, dtype=new_labels[0].dtype, device=new_labels[0].device)
    padded_attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=attention_mask.device)

    for idx, (cur_embed, cur_label) in enumerate(zip(new_input_embeds, new_labels)):
        cur_len = cur_embed.shape[0]
        pad_len = max_len - cur_len
        padded_embeds.append(
            torch.cat(
                [cur_embed, cur_embed.new_zeros((pad_len, embed_dim))],
                dim=0,
            )
        )
        padded_labels[idx, :cur_len] = cur_label
        padded_attention_mask[idx, :cur_len] = True
        position_ids[idx, :cur_len] = torch.arange(cur_len, device=position_ids.device)

    return torch.stack(padded_embeds, dim=0), padded_attention_mask, padded_labels, position_ids


def patch_hf_llava_with_visionzip(model, dominant=54, contextual=10):
    base_model = getattr(model, "model", model)
    if not hasattr(base_model, "vision_tower"):
        raise ValueError("Could not find vision_tower on HF Llava model. Expected model.model.vision_tower or model.vision_tower.")

    vision_tower = base_model.vision_tower
    encoder_layers = vision_tower.vision_model.encoder.layers
    original_vision_forward = vision_tower.forward
    target_layer = encoder_layers[-2]

    vision_tower._visionzip_info = {
        "dominant": max(int(dominant), 1),
        "contextual": max(int(contextual), 0),
    }

    target_layer._visionzip_info = vision_tower._visionzip_info
    target_layer.self_attn.forward = types.MethodType(_clip_attention_forward, target_layer.self_attn)
    target_layer.forward = types.MethodType(_clip_encoder_layer_forward, target_layer)

    def vision_forward_visionzip(self, *args, **kwargs):
        target_layer.metric = None
        target_layer._visionzip_last_attn_weights = None
        target_layer.self_attn._visionzip_last_attn_weights = None
        return original_vision_forward(*args, **kwargs)

    vision_tower.forward = types.MethodType(vision_forward_visionzip, vision_tower)

    original_forward = base_model.forward

    def forward_visionzip(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        logits_to_keep=0,
        image_sizes=None,
        **lm_kwargs,
    ):
        from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast

        has_image_tokens = bool(
            input_ids is not None
            and self.config.image_token_index is not None
            and (input_ids == self.config.image_token_index).any().item()
        )
        cache_starts_at_zero = bool(
            cache_position is None
            or cache_position.numel() == 0
            or cache_position.reshape(-1)[0].item() == 0
        )
        is_prefill_like = bool(
            pixel_values is not None
            and inputs_embeds is None
            and cache_starts_at_zero
        )
        should_merge_images = bool(is_prefill_like and has_image_tokens)

        if not should_merge_images:
            # In generation, some code paths can still forward pixel_values when the current input_ids no longer
            # contain any <image> placeholder. The original HF Llava forward would then try to merge image features
            # again and fail on the token/feature count check. Drop image inputs on those decode-like steps.
            if pixel_values is not None and not has_image_tokens:
                pixel_values = None
                image_sizes = None
            return original_forward(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                image_sizes=image_sizes,
                logits_to_keep=logits_to_keep,
                **lm_kwargs,
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if vision_feature_layer is None:
            vision_feature_layer = getattr(self.config, "vision_feature_layer", -2)
        if vision_feature_select_strategy is None:
            vision_feature_select_strategy = getattr(self.config, "vision_feature_select_strategy", "default")

        image_features = self.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            image_sizes=image_sizes,
        )
        if isinstance(image_features, (list, tuple)):
            image_features = torch.cat(image_features, dim=0)
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

        inputs_embeds, attention_mask, merged_labels, position_ids = _merge_inputs_with_image_features(
            self=self,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_features,
            attention_mask=attention_mask,
            labels=labels,
        )
        labels = merged_labels if labels is not None else None

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            if attention_mask is not None:
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

    base_model.get_image_features = types.MethodType(_get_image_features_visionzip, base_model)
    base_model.forward = types.MethodType(forward_visionzip, base_model)
    model._visionzip_enabled = True
    model._visionzip_dominant = dominant
    model._visionzip_contextual = contextual
    return model
