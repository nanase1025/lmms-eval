from typing import Optional, Union

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.llava import Llava

from loguru import logger as eval_logger


@register_model("llava_visionzip")
class LlavaVisionZip(Llava):
    """
    LLaVA model with VisionZip visual token pruning.

    VisionZip reduces visual tokens by retaining `dominant` dominant tokens
    and `contextual` contextual tokens, significantly reducing prefill cost
    with minimal accuracy loss.

    Example usage:

    python -m lmms_eval \\
        --model llava_visionzip \\
        --model_args pretrained=liuhaotian/llava-v1.5-7b,dominant=54,contextual=10 \\
        --tasks pope \\
        --batch_size 1
    """

    def __init__(
        self,
        dominant: int = 54,
        contextual: int = 10,
        **kwargs,
    ) -> None:
        # Pop visionzip-specific args before forwarding to Llava.__init__,
        # which asserts that no unknown kwargs remain.
        super().__init__(**kwargs)

        eval_logger.info(f"Applying VisionZip: dominant={dominant}, contextual={contextual}")
        try:
            from visionzip import visionzip as apply_visionzip
        except ImportError:
            raise ImportError(
                "VisionZip is not installed. "
                "Install it with: pip install visionzip  "
                "or: cd /path/to/VisionZip && pip install -e ."
            )

        apply_visionzip(self._model, dominant=dominant, contextual=contextual)
        eval_logger.info("VisionZip applied successfully.")
