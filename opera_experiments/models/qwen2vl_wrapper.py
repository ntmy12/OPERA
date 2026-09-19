import torch
import json
import os
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, LogitsProcessorList
from typing import Optional, List, Dict, Any

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opera_core.opera_decoding import OverTrustPenaltyLogitsProcessor, RetrospectionAllocator

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    # Fallback placeholder if not installed
    def process_vision_info(messages):
        return None, None


def resolve_dtype(dtype: Optional[Any] = None) -> torch.dtype:
    """Resolve torch dtype with BF16/FP16 fallback for hardware like T4."""
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        d_lower = dtype.lower()
        if d_lower in ["bf16", "bfloat16"]:
            return torch.bfloat16
        elif d_lower in ["fp16", "float16"]:
            return torch.float16
        elif d_lower in ["fp32", "float32"]:
            return torch.float32
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    return torch.float32


class Qwen2VLWrapper:
    """
    Wrapper for Qwen2-VL-7B-Instruct to support standard generation and OPERA-augmented generation
    with multi-GPU (device_map='auto') and BF16/FP16 precision.
    """
    def __init__(self, model_path: str, device: str = "auto", dtype: Optional[Any] = torch.bfloat16):
        self.device = device
        self.dtype = resolve_dtype(dtype)
        self.model_path = model_path
        self.device_map = "auto" if device == "auto" else device
        
        print(f"Loading Qwen2-VL-7B-Instruct from {model_path} (dtype={self.dtype}, device_map={self.device_map})...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            device_map=self.device_map,
            attn_implementation="eager"
        )
        self.model.eval()

    def _prepare_inputs(self, prompt: str, image) -> Dict[str, Any]:
        # Qwen2-VL requires a specific chat format for inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        target_device = self.model.device if hasattr(self.model, "device") else (
            torch.device(self.device) if self.device != "auto" else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        return {
            k: v.to(target_device, dtype=self.dtype if torch.is_floating_point(v) else None) 
            if isinstance(v, torch.Tensor) else v 
            for k, v in inputs.items()
        }

    def generate_baseline(self, prompt: str, image, max_new_tokens: int = 128, **kwargs) -> str:
        """
        Standard generation without hallucination mitigation.
        """
        inputs = self._prepare_inputs(prompt, image)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            
        input_ids = inputs["input_ids"]
        out_sequences = output_ids.sequences if hasattr(output_ids, "sequences") else output_ids
        generated_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, out_sequences)
        ]
        
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()
        
        return generated_text

    def generate_with_opera(self, prompt: str, image, opera_config: Dict[str, Any], max_new_tokens: int = 128, **kwargs) -> str:
        """
        Generation augmented with OPERA (Over-trust Penalty & Retrospection-Allocation).
        """
        inputs = self._prepare_inputs(prompt, image)
        
        # Initialize OPERA components
        logits_processor = LogitsProcessorList([
            OverTrustPenaltyLogitsProcessor(
                scale_factor=opera_config.get("scale_factor", 50.0),
                num_attn_candidates=opera_config.get("num_attn_candidates", 5),
                penalty_weights=opera_config.get("penalty_weights", 1.0)
            )
        ])
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                logits_processor=logits_processor,
                output_attentions=True,
                return_dict_in_generate=True,
                **kwargs
            )
            
        input_ids = inputs["input_ids"]
        out_sequences = output_ids.sequences if hasattr(output_ids, "sequences") else output_ids
        generated_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, out_sequences)
        ]
        
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()
        
        return generated_text
