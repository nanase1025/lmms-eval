import time
import warnings
from typing import List, Optional, Union

import PIL
import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor

from lmms_eval import utils
from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.latency_hf import patch_hf_llava_latency_measurement
from lmms_eval.models.model_utils.visionzip_hf import patch_hf_llava_with_visionzip
from lmms_eval.models.simple.llava_hf import LlavaHf

warnings.filterwarnings("ignore")


@register_model("llava_hf_visionzip")
class LlavaHfVisionZip(LlavaHf):
    def __init__(
        self,
        pretrained: str = "llava-hf/llava-1.5-7b-hf",
        revision: str = "main",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = False,
        attn_implementation: Optional[str] = None,
        device_map: str = "",
        chat_template: Optional[str] = None,
        use_cache: bool = True,
        max_frames_num: Optional[int] = 32,
        visionzip_dominant: int = 54,
        visionzip_contextual: int = 10,
        visionzip_dominant_ratio: Optional[float] = None,
        visionzip_contextual_ratio: Optional[float] = None,
        dominant_ratio: Optional[float] = None,
        contextual_ratio: Optional[float] = None,
        visionzip_enabled: bool = True,
        **kwargs,
    ) -> None:
        lmms.__init__(self)
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1 and device_map == "":
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else "auto"
        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)

        config = AutoConfig.from_pretrained(pretrained)
        self.max_frames_num = max_frames_num
        if getattr(config, "model_type", "llava") != "llava":
            raise ValueError("llava_hf_visionzip currently supports only HF Llava models.")

        if attn_implementation is None:
            attn_implementation = "eager"

        from transformers import LlavaForConditionalGeneration

        self._model = LlavaForConditionalGeneration.from_pretrained(
            pretrained,
            revision=revision,
            torch_dtype=dtype,
            device_map=self.device_map,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )

        self.visionzip_enabled = visionzip_enabled
        image_seq_length = int(getattr(config, "image_seq_length", 576))
        if dominant_ratio is not None:
            visionzip_dominant_ratio = dominant_ratio
        if contextual_ratio is not None:
            visionzip_contextual_ratio = contextual_ratio

        self.visionzip_dominant_ratio = float(visionzip_dominant_ratio) if visionzip_dominant_ratio is not None else None
        self.visionzip_contextual_ratio = float(visionzip_contextual_ratio) if visionzip_contextual_ratio is not None else None

        if self.visionzip_dominant_ratio is not None or self.visionzip_contextual_ratio is not None:
            dominant_ratio_value = self.visionzip_dominant_ratio if self.visionzip_dominant_ratio is not None else (visionzip_dominant / image_seq_length)
            contextual_ratio_value = self.visionzip_contextual_ratio if self.visionzip_contextual_ratio is not None else (visionzip_contextual / image_seq_length)
            self.visionzip_dominant = max(min(int(dominant_ratio_value * image_seq_length), image_seq_length), 1)
            remaining_tokens = max(image_seq_length - self.visionzip_dominant, 0)
            self.visionzip_contextual = max(min(int(contextual_ratio_value * image_seq_length), remaining_tokens), 0)
        else:
            self.visionzip_dominant = max(min(int(visionzip_dominant), image_seq_length), 1)
            remaining_tokens = max(image_seq_length - self.visionzip_dominant, 0)
            self.visionzip_contextual = max(min(int(visionzip_contextual), remaining_tokens), 0)

        if visionzip_enabled:
            patch_hf_llava_with_visionzip(
                self._model,
                dominant=self.visionzip_dominant,
                contextual=self.visionzip_contextual,
            )
        patch_hf_llava_latency_measurement(self._model)

        ratio_log = ""
        if self.visionzip_dominant_ratio is not None or self.visionzip_contextual_ratio is not None:
            ratio_log = (
                f", dominant_ratio={self.visionzip_dominant_ratio if self.visionzip_dominant_ratio is not None else self.visionzip_dominant / image_seq_length:.4f}"
                f", contextual_ratio={self.visionzip_contextual_ratio if self.visionzip_contextual_ratio is not None else self.visionzip_contextual / image_seq_length:.4f}"
            )
        eval_logger.info(
            f"Using VisionZip dominant={self.visionzip_dominant}, contextual={self.visionzip_contextual}{ratio_log}"
        )

        self.pretrained = pretrained
        self._image_processor = AutoProcessor.from_pretrained(pretrained, revision=revision, trust_remote_code=trust_remote_code)
        self._image_processor.tokenizer.padding_side = "left"
        self._tokenizer = self._image_processor.tokenizer
        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self.chat_template = chat_template
        self.use_cache = use_cache

        if accelerator.num_processes > 1 and device_map == "":
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED]
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                ds_kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **ds_kwargs)
            if accelerator.distributed_type in [DistributedType.FSDP, DistributedType.DEEPSPEED]:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._rank = 0
            self._world_size = 1
        else:
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1
        self.accelerator = accelerator

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        res = []
        total_elapsed_time = 0.0
        total_gen_tokens = 0
        total_prefill_time_ms = 0.0
        total_decode_time_ms = 0.0
        total_peak_memory_gb = 0.0
        request_count = 0

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        measured_model = self.model
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
            else:
                task_type = "text"

            gen_kwargs = all_gen_kwargs[0]
            until = [self.tok_decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")
            assert self.batch_size_per_gpu == 1, "Do not support batch_size_per_gpu > 1 for now"
            context = contexts[0]

            messages = [{"role": "user", "content": self._build_message_content(context, task_type, len(visuals))}]
            text = self._image_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=self._resolve_chat_template(),
            )

            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Prompt for doc ID {doc_id[0]}:\n\n{text}\n")

            if task_type == "video":
                try:
                    visuals = [self.load_video(visuals, self.max_frames_num)]
                except Exception as e:
                    res.append(GenerationResult(text="", token_counts=None))
                    eval_logger.info(f"Error {e} when loading video : {visuals}")
                    pbar.update(1)
                    continue

            if task_type == "image":
                inputs = self._image_processor(images=visuals, text=text, return_tensors="pt").to(self._device, self.model.dtype)
            elif task_type == "video":
                inputs = self._image_processor(videos=visuals, text=text, return_tensors="pt").to(self._device, self.model.dtype)
            else:
                inputs = self._image_processor(text=text, return_tensors="pt").to(self._device, self.model.dtype)

            gen_kwargs["image_sizes"] = [visuals[idx].size for idx in range(len(visuals))] if task_type == "image" else []
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            do_sample = True if gen_kwargs["temperature"] > 0 else False

            peak_memory_gb = 0.0
            generated_ids_trimmed = None
            text_outputs = ""
            previous_measure_latency = getattr(measured_model, "measure_latency", False)
            measured_model.measure_latency = True
            measured_model.reset_time()
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)
            try:
                start_time = time.time()
                cont = measured_model.generate(
                    **inputs,
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
                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
                text_outputs = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
                for term in until:
                    if len(term) > 0:
                        text_outputs = text_outputs.split(term)[0]

                total_elapsed_time += end_time - start_time
                total_gen_tokens += sum(len(ids) for ids in generated_ids_trimmed)
                total_prefill_time_ms += float(getattr(measured_model, "prefill_latency", 0.0))
                total_decode_time_ms += float(getattr(measured_model, "decode_latency", 0.0))
                if self.device.type == "cuda" and torch.cuda.is_available():
                    peak_memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024 * 1024)
                total_peak_memory_gb += peak_memory_gb
                request_count += 1
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
            finally:
                measured_model.measure_latency = previous_measure_latency

            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Generated text for doc ID {doc_id[0]}:\n\n{text_outputs}\n")

            token_counts = TokenCounts(output_tokens=len(generated_ids_trimmed[0])) if generated_ids_trimmed is not None else None
            res.append(GenerationResult(text=text_outputs, token_counts=token_counts))
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)

        res = re_ords.get_original(res)

        avg_prefill_time_ms = (total_prefill_time_ms / request_count) if request_count > 0 else 0.0
        avg_decode_time_ms = (total_decode_time_ms / request_count) if request_count > 0 else 0.0
        avg_peak_memory_gb = (total_peak_memory_gb / request_count) if request_count > 0 else 0.0
        avg_speed = (total_gen_tokens / total_elapsed_time) if total_elapsed_time > 0 else 0.0
        log_metrics(
            total_elapsed_time=total_elapsed_time,
            total_gen_tokens=total_gen_tokens,
            avg_speed=avg_speed,
            additional_metrics={
                "prefill_time_ms": avg_prefill_time_ms,
                "decode_time_ms": avg_decode_time_ms,
                "peak_memory_gb": avg_peak_memory_gb,
                "rank": self.rank,
                "total_requests": request_count,
            },
        )

        pbar.close()
        return res
