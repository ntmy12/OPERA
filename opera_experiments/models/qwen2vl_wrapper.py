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


class Qwen2VLWrapper:
    """
    Wrapper for Qwen2-VL-7B-Instruct to support standard generation and OPERA-augmented generation.
    """
    def __init__(self, model_path: str, device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        
        print(f"Loading Qwen2-VL-7B-Instruct from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
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
        
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

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
            
        generated_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
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
            
        generated_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids.sequences)
        ]
        
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()
        
        return generated_text
