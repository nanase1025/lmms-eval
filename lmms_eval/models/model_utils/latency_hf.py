import types

import torch


def _patch_decoder_forward_latency_measurement(model, decoder_model, applied_attr):
    if getattr(decoder_model, applied_attr, False):
        return model

    original_forward = decoder_model.forward

    model.measure_latency = False
    model.prefill_latency = 0.0
    model.decode_latency = 0.0
    model._latency_seen_prefill = False

    def reset_time():
        model.prefill_latency = 0.0
        model.decode_latency = 0.0
        model._latency_seen_prefill = False

    model.reset_time = reset_time

    def _extract_seq_len(input_ids, inputs_embeds):
        if input_ids is not None:
            return input_ids.shape[1]
        if inputs_embeds is not None:
            return inputs_embeds.shape[1]
        return None

    def decoder_forward_with_latency(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids")
        inputs_embeds = kwargs.get("inputs_embeds")
        seq_len = _extract_seq_len(input_ids, inputs_embeds)

        should_measure = bool(getattr(model, "measure_latency", False) and torch.cuda.is_available())
        is_prefill = bool(
            should_measure
            and not model._latency_seen_prefill
            and inputs_embeds is not None
            and (seq_len is not None and seq_len > 1)
        )
        is_decode = bool(should_measure and model._latency_seen_prefill and seq_len == 1)

        if not (is_prefill or is_decode):
            return original_forward(*args, **kwargs)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        outputs = original_forward(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)

        if is_prefill:
            model.prefill_latency += elapsed_ms
            model._latency_seen_prefill = True
        else:
            model.decode_latency += elapsed_ms

        return outputs

    decoder_model.forward = types.MethodType(decoder_forward_with_latency, decoder_model)
    setattr(decoder_model, applied_attr, True)
    return model


def patch_hf_llava_latency_measurement(model):
    base_model = getattr(model, "model", model)
    if not hasattr(base_model, "language_model"):
        raise ValueError("Could not find language_model on HF Llava model for latency measurement.")
    return _patch_decoder_forward_latency_measurement(
        model=model,
        decoder_model=base_model.language_model,
        applied_attr="_latency_patch_applied",
    )


def patch_hf_qwen2_5_vl_latency_measurement(model):
    base_model = getattr(model, "model", model)
    if not hasattr(base_model, "layers"):
        raise ValueError("Could not find decoder layers on HF Qwen2.5-VL model for latency measurement.")
    return _patch_decoder_forward_latency_measurement(
        model=model,
        decoder_model=base_model,
        applied_attr="_latency_patch_applied",
    )
