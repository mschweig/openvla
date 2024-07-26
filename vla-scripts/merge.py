#!/usr/bin/env python3
import os
import torch
import draccus

from dataclasses import dataclass
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForVision2Seq


@dataclass
class MergeConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b" # Path to base model used for training
    adapter_path: Path = None            # Path to the LoRA weights to merge
    output_path: Path = None             # Path to the merged output model

    # fmt: on


@draccus.wrap()
def merge(cfg: MergeConfig) -> None:
    merge_lora(cfg.vla_path, cfg.adapter_path, cfg.output_path)
    
    
def merge_lora(vla_path, adapter_path, output_path):
    # Merge LoRA weights into model backbone for faster inference
    #   =>> Note that merging is slow and can be done post-hoc to speed up training
    print(f"Merging LoRA weights from {adapter_path} into {output_path}")
    base_vla = AutoModelForVision2Seq.from_pretrained(
        vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    )
    merged_vla = PeftModel.from_pretrained(base_vla, adapter_path)
    merged_vla = merged_vla.merge_and_unload(progressbar=True)
    merged_vla.save_pretrained(output_path)
    print(f"Saved merged LoRA weights to {output_path}")


if __name__ == "__main__":
    merge()
