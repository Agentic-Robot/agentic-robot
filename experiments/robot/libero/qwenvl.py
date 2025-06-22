# qwenvl.py (修改版 - 使用图像队列和英文 Prompt)

import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import traceback
import collections 

# 尝试导入 Qwen-VL 官方提供的工具函数 (保持不变)
from qwen_vl_utils import process_vision_info
QWEN_UTILS_AVAILABLE = True
print("Successfully imported qwen_vl_utils.process_vision_info.")


def load_qwen_vl_model(model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: str = "auto"):

    print(f"Attempting to load VLM model: {model_id} to device: {device} (using Flash Attention 2)")
    model = None
    processor = None
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map=device,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print("VLM model and processor loaded successfully.")
        return model, processor
    except Exception as e:
        print(f"Error: Failed to load VLM model/processor: {e}")
        traceback.print_exc()
        return None, None


def check_completion_with_qwen_vl(
    vlm_model,
    vlm_processor,
  
    image_pair_queue: collections.deque, 

    current_subtask_instruction: str
) -> bool:
    """
    Uses the loaded Qwen-VL model to check if the current subtask is complete,
    based on a sequence of image pairs.
    Relies on zero-shot prompting and parsing of "Yes"/"No" responses.
    Uses qwen_vl_utils.process_vision_info for image processing.
    Prompts and parsing are in English.
    """

    if vlm_model is None or vlm_processor is None:
        print("[VLM Check] Error: VLM model or processor not loaded.")
        return False
    if not QWEN_UTILS_AVAILABLE:
        print("[VLM Check] Error: qwen_vl_utils not available.")
        return False
    
    if not image_pair_queue:
        print("[VLM Check] Warning: Image queue is empty. Cannot perform check.")
        return False
    

    try:
        

        object_name = "the object" # Default English names
        location_name = "the target location"
        instruction_lower = current_subtask_instruction.lower()
        verb = ""
        object_part_raw = ""
        if instruction_lower.startswith("pick up"):
            verb = "pick up"
            try: object_part_raw = current_subtask_instruction.split(verb, 1)[1].strip().rstrip('.')
            except: pass
            object_name = object_part_raw if object_part_raw else object_name
        elif instruction_lower.startswith("place"):
            verb = "place"
            try: object_part_raw = current_subtask_instruction.split(verb, 1)[1].strip().rstrip('.')
            except: pass
            split_success = False
            # Use English prepositions
            for separator in [" in ", " on ", " into ", " onto "]:
                if separator in object_part_raw:
                    obj_temp, loc_temp = object_part_raw.split(separator, 1)
                    object_name = obj_temp.strip() if obj_temp.strip() else object_name
                    location_name = loc_temp.strip() if loc_temp.strip() else location_name
                    split_success = True
                    break
            if not split_success: object_name = object_part_raw if object_part_raw else object_name
        
        prompt_text = ""
       
        prompt_prefix = f"Observe the following sequence of {len(image_pair_queue)} image pairs (main view and hand view over time). The instruction for the robot is: '{current_subtask_instruction}'. "

        if verb == "pick up":
            
            prompt_text = f"{prompt_prefix}Based *only* on the image sequence, has '{object_name}' been securely grasped by the gripper and clearly lifted off any surface it was resting on by the end of the sequence? Please answer strictly with 'Yes' or 'No'."
        elif verb == "place":
            
            prompt_text = f"{prompt_prefix}Based *only* on the image sequence, has '{object_name}' been stably placed {location_name} (e.g., 'in the basket', 'on the table'), and does the gripper appear empty or clearly moving away from the object by the end of the sequence? Please answer strictly with 'Yes' or 'No'."
        else:
           
            prompt_text = f"{prompt_prefix}Based *only* on the image sequence and the instruction, has this action been successfully completed by the end of the sequence? Please answer strictly with 'Yes' or 'No'."
        

        content_list = []
        
        for i, (main_img_pil, eih_img_pil) in enumerate(image_pair_queue):
            content_list.append({"type": "text", "text": f"Image Pair {i+1} (Main View):"})
            content_list.append({"type": "image", "image": main_img_pil})
            content_list.append({"type": "text", "text": f"Image Pair {i+1} (Hand View):"})
            content_list.append({"type": "image", "image": eih_img_pil})

        
        content_list.append({"type": "text", "text": prompt_text})

        messages = [
            {
                "role": "user",
                "content": content_list,
            }
        ]
        
        text_template = vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        if image_inputs is None:
             print("[VLM Check] Error: process_vision_info failed to process image inputs.")
             return False

        inputs = vlm_processor(
            text=[text_template],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(vlm_model.device)

        with torch.no_grad():
            generated_ids = vlm_model.generate(
                **inputs,
                max_new_tokens=10, # Still expect short Yes/No
                min_new_tokens=1,
                do_sample=False,
                pad_token_id=vlm_processor.tokenizer.eos_token_id # Corrected access
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = vlm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        if response:
            vlm_response_text = response[0].strip()
            # print(f"[VLM Check] VLM raw response: '{vlm_response_text}'") # Debug

            # Be a bit lenient: check lower case and potential starting variations
            if vlm_response_text.lower().startswith("yes"):
                print(f"[VLM Check] VLM Judgement: COMPLETED (Response: '{vlm_response_text}')")
                return True
            else:
                # print(f"[VLM Check] VLM Judgement: Not Completed (Response: '{vlm_response_text}')") # Debug
                return False

        else:
            # print("[VLM Check] VLM Judgement: Not Completed (No valid response)") # Debug
            return False

    except Exception as e:
        print(f"Error: Exception during VLM check (Instruction: '{current_subtask_instruction}'): {e}")
        traceback.print_exc()
        return False