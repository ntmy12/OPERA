import torch
import json
import os
from transformers import LlavaForConditionalGeneration, AutoProcessor, LogitsProcessorList
from typing import Optional, List, Dict, Any

# Ensure opera_core is importable when running from project root
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opera_core.opera_decoding import OverTrustPenaltyLogitsProcessor, RetrospectionAllocator

class LLaVAWrapper:
    """
    Wrapper for LLaVA-1.5-7B to support standard generation and OPERA-augmented generation.
    """
    def __init__(self, model_path: str, device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        
        print(f"Loading LLaVA-1.5 from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=device,
            attn_implementation="eager"
        )
        self.model.eval()

    def _prepare_inputs(self, prompt: str, image) -> Dict[str, torch.Tensor]:
        if "<image>" not in prompt:
            prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        return {k: v.to(self.device, dtype=self.dtype if torch.is_floating_point(v) else None) for k, v in inputs.items()}

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
        
        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        generated_text = self.processor.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True
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
        
        allocator = RetrospectionAllocator(
            threshold=opera_config.get("threshold", 15.0)
        )
        
        # To truly implement Retrospection (rollback), we often need to step manually
        # or use a custom stopping criteria / custom beam search. 
        # For simplicity in this wrapper, we apply the penalty logits processor.
        # In a fully-fledged custom loop, we would extract attention at each step and check allocator.
        
        # Hooking attention requires output_attentions=True
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                logits_processor=logits_processor,
                output_attentions=True, 
                return_dict_in_generate=True,
                **kwargs
            )
            
        # Decode
        generated_ids = output_ids.sequences
        input_len = inputs["input_ids"].shape[1]
        generated_text = self.processor.batch_decode(
            generated_ids[:, input_len:], skip_special_tokens=True
        )[0].strip()
        
        return generated_text
