"""
verify_openvla.py

Given an HF-exported OpenVLA model, attempt to load via AutoClasses, and verify forward() and predict_action().
"""

import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# === Verification Arguments
MODEL_PATH = "openvla/openvla-7b"

INSTRUCTION = "put spoon on towel"


def get_openvla_prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


@torch.inference_mode()
def verify_openvla() -> None:
    print(f"[*] Verifying OpenVLAForActionPrediction using Model `{MODEL_PATH}`")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load Processor & VLA
    print("[*] Instantiating Processor and Pretrained OpenVLA")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # === BFLOAT16 + FLASH-ATTN MODE ===
    print("[*] Loading in BF16 with Flash-Attention Enabled")
    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    print("[*] Iterating with Randomly Generated Images")
    for _ in range(10):
        prompt = get_openvla_prompt(INSTRUCTION)
        image = Image.fromarray(np.asarray(np.random.rand(256, 256, 3) * 255, dtype=np.uint8))

        # === BFLOAT16 MODE ===
        inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

        # Run OpenVLA Inference
        start_time = time.time()
        action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        print(f"\t=>> Time: {time.time() - start_time:.4f} || Action: {action}")


if __name__ == "__main__":
    verify_openvla()
