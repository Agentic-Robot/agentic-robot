# run_libero_eval_pev_v1.py 

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import torch 
import tqdm
from libero.libero import benchmark
from PIL import Image

import wandb
import collections
import time

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
    resize_image, 
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


from qwenvl import load_qwen_vl_model, check_completion_with_qwen_vl
VLM_AVAILABLE = True


@dataclass
class GenerateConfig:

    # fmt: off

    # #################################################################################################################
    # # Model-specific parameters
    # #################################################################################################################
    model_family: str = "openvla"                # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    # #################################################################################################################
    # # LIBERO environment-specific parameters
    # #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 5                  # Number of rollouts per task (原默认 50)

    # #################################################################################################################
    # # Utils
    # #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 42                                    # Random Seed (for reproducibility)

    save_frames: bool = True                        
    frames_save_root_dir: str = "./experiments/saved_frames" 
    verify_frequency: int = 5                       

    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load VLA model (Executor E)
    vla_model = get_model(cfg) # Renamed for clarity

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in vla_model.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla_model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in vla_model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor for VLA
    vla_processor = None
    if cfg.model_family == "openvla":
        vla_processor = get_processor(cfg) # Renamed for clarity

    # Load VLM model (Verifier V) - Load once before the loops
    vlm_model, vlm_processor = None, None

    print("No VLM")
    VLM_ENABLED_EFFECTIVELY = False
    
    hardcoded_plan = [
        "put both the alphabet soup and the tomato sauce in the basket",
    ]
    
    print(f"使用硬编码计划进行测试: {hardcoded_plan}")

    # Initialize local logging
    run_id = f"PEV_V1-EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}" # Added PEV_V1 prefix
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")
    log_file.write(f"PEV V1 Evaluation Run\n")
    log_file.write(f"Hardcoded Plan: {hardcoded_plan}\n")
    log_file.write(f"VLM Verification Enabled: {VLM_ENABLED_EFFECTIVELY}\n")

    frames_run_dir = None # Initialize here
    if cfg.save_frames:
        frames_run_dir = Path(cfg.frames_save_root_dir) / run_id
        os.makedirs(frames_run_dir, exist_ok=True)
        print(f"Will save frames to subdirectories within: {frames_run_dir}")
        log_file.write(f"Saving frames enabled, base directory: {frames_run_dir}\n")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
            config=draccus.encode(cfg) # Log config
        )
        wandb.config.update({"hardcoded_plan": hardcoded_plan, "vlm_enabled": VLM_ENABLED_EFFECTIVELY})


    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc="Tasks"):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and ORIGINAL task description
        env, original_task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task), desc=f"Task {task_id} Episodes", leave=False):
            print(f"\nTask: {original_task_description}")
            log_file.write(f"\nOriginal Task: {original_task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            current_subtask_index = 0
            if not hardcoded_plan:
                print("Warning: Plan is empty, cannot execute subtasks.")
                log_file.write("Warning: Plan is empty.\n")
                current_subtask_instruction = original_task_description # Fallback
            else:
                #current_subtask_instruction = hardcoded_plan[current_subtask_index]
                current_subtask_instruction = original_task_description
            print(f"Starting Episode {episode_idx+1} with Subtask 1: '{current_subtask_instruction}'")
            log_file.write(f"Episode {episode_idx+1} | Subtask {current_subtask_index+1}: {current_subtask_instruction}\n")
            plan_completed_by_vlm = False

            image_pair_queue = collections.deque(maxlen=1)

            episode_frame_save_dir = None
            if cfg.save_frames and frames_run_dir: # Check frames_run_dir exists
                episode_frame_save_dir = frames_run_dir / f"task_{task_id}" / f"episode_{episode_idx}"
                os.makedirs(episode_frame_save_dir, exist_ok=True)
                # log_file.write(f"Saving episode frames to: {episode_frame_save_dir}\n") # Maybe too verbose

            # Setup
            t = 0
            replay_images = []
            # ... (max_steps calculation remains the same) ...
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300
            elif cfg.task_suite_name == "libero_10": #520
                max_steps = 520
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400
            else: # Fallback
                max_steps = 400

            log_file.write(f"Starting episode run (max_steps={max_steps})...\n")
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # Wait steps
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))

                        if VLM_ENABLED_EFFECTIVELY and (t % 5 == 0): 
                             img_main_np_wait = get_libero_image(obs, resize_size)
                             img_eih_np_wait = obs["robot0_eye_in_hand_image"]
                             main_img_pil_wait = Image.fromarray(img_main_np_wait)
                             eih_img_pil_wait = Image.fromarray(img_eih_np_wait)
                             image_pair_queue.append((main_img_pil_wait, eih_img_pil_wait))
                    
                        t += 1
                        continue

                    # --- Get Images (Main and Eye-in-Hand) ---
                    # Get preprocessed main image (likely returns NumPy)
                    img_main_np = get_libero_image(obs, resize_size)
                    # Get eye-in-hand image (NumPy)
                    img_eih_np = obs["robot0_eye_in_hand_image"]

                    # --- Save Frames (Every 20 steps, if enabled) ---
                    # Moved frame saving here to ensure we have the images before VLM check
                    if cfg.save_frames and episode_frame_save_dir and (t % cfg.verify_frequency == 0):
                        try:
                            # Save main image
                            frame_filename = f"frame_{t:04d}.png"
                            pil_img_main = Image.fromarray(img_main_np)
                            pil_img_main.save(episode_frame_save_dir / frame_filename)

                            # Save eye-in-hand image
                            frame_filename_eih = f"eyeinhand_{t:04d}.png"
                            pil_img_eih = Image.fromarray(img_eih_np)
                            pil_img_eih.save(episode_frame_save_dir / frame_filename_eih)
                        except Exception as save_e:
                            print(f"Warning: Failed to save frame {t}. Error: {save_e}")
                            log_file.write(f"Warning: Failed to save frame {t}. Error: {save_e}\n")

                
                    image_collected_this_step = False
                    if VLM_ENABLED_EFFECTIVELY and (t % 5 == 0): 
                        image_collected_this_step = True
                        
                        main_img_pil = Image.fromarray(img_main_np)
                        eih_img_pil = Image.fromarray(img_eih_np)
                        image_pair_queue.append((main_img_pil, eih_img_pil))

                        
                        if cfg.save_frames and episode_frame_save_dir:
                             try:
                                 frame_filename = f"frame_{t:04d}.png"
                                 main_img_pil.save(episode_frame_save_dir / frame_filename)
                                 frame_filename_eih = f"eyeinhand_{t:04d}.png"
                                 eih_img_pil.save(episode_frame_save_dir / frame_filename_eih)
                             except Exception as save_e:
                                 print(f"Warning: Failed to save frame {t}. Error: {save_e}")
                                 log_file.write(f"Warning: Failed to save frame {t}. Error: {save_e}\n")

                    
                    vlm_check_this_step = False
                    if VLM_ENABLED_EFFECTIVELY and (t % cfg.verify_frequency == 0): 
                         vlm_check_this_step = True

                    if vlm_check_this_step and not plan_completed_by_vlm:
                        
                        print(f"[t={t}] Performing VLM check for subtask: '{current_subtask_instruction}' using queue (size {len(image_pair_queue)})")
                        log_file.write(f"[t={t}] VLM Check Start: '{current_subtask_instruction}' (Queue size: {len(image_pair_queue)})\n")

                        is_subtask_complete = check_completion_with_qwen_vl(
                            vlm_model=vlm_model,
                            vlm_processor=vlm_processor,
                            image_pair_queue=image_pair_queue, 
                            current_subtask_instruction=current_subtask_instruction
                        )

                        log_file.write(f"[t={t}] VLM Check Result: {'Complete' if is_subtask_complete else 'Not Complete'}\n")

                        if is_subtask_complete:
                            
                            print(f"[t={t}] VLM Confirmed: Subtask '{current_subtask_instruction}' COMPLETE.")
                            current_subtask_index += 1
                            if current_subtask_index < len(hardcoded_plan):
                                current_subtask_instruction = hardcoded_plan[current_subtask_index]
                                print(f"[t={t}] Switching to next subtask ({current_subtask_index + 1}/{len(hardcoded_plan)}): '{current_subtask_instruction}'")
                                log_file.write(f"[t={t}] VLM SWITCH -> Subtask {current_subtask_index+1}: {current_subtask_instruction}\n")
                            else:
                                print(f"[t={t}] VLM Confirmed: All {len(hardcoded_plan)} subtasks in the plan are complete.")
                                log_file.write(f"[t={t}] VLM: Plan finished.\n")
                                plan_completed_by_vlm = True
                                print(f"[t={t}] Continuing execution with last subtask instruction: '{current_subtask_instruction}' until env done or max_steps.")
                   
                    # Save main image for replay video (use the already processed img_main_np)
                    replay_images.append(img_main_np)

                    # Prepare observations dict for VLA
                    observation = {
                        "full_image": img_main_np, # Use the main image
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    
                    action = get_action(
                        cfg,
                        vla_model, # Use VLA model
                        observation,
                        # Use the current subtask instruction, NOT original_task_description
                        current_subtask_instruction,
                        vla_processor, # Use VLA processor
                    )
                    

                    # Normalize gripper action
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] Invert gripper action if needed
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)
                        # print(f"Action: {action}") # Can be verbose

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())

                    # --- Check Environment Done (Original Success Condition) ---
                    if done:
                        task_successes += 1
                        total_successes += 1
                        print(f"[t={t}] Environment signaled DONE (Success!).")
                        log_file.write(f"[t={t}] Environment DONE signal received.\n")
                        break # Break episode loop on success

                    t += 1 # Increment timestep

                    # Check for timeout
                    if t >= max_steps + cfg.num_steps_wait:
                         print(f"[t={t}] Episode TIMEOUT.")
                         log_file.write(f"[t={t}] Episode TIMEOUT.\n")
                         # Ensure 'done' is False if timeout occurred before env signaled success
                         done = False # Explicitly set done to False on timeout
                         break # Break episode loop on timeout


                except Exception as e:
                    print(f"Caught exception during episode execution: {e}")
                    log_file.write(f"Exception during episode: {e}\n")
                    import traceback
                    traceback.print_exc(file=log_file)
                    done = False # Assume failure on exception
                    break # Break episode loop on error

            # --- Episode End ---
            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=original_task_description, log_file=log_file
            )

            # Log current results
            print(f"Episode End. Success (based on env done signal): {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Episode End. Success (Env Done): {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

            # Log episode result to W&B if enabled
            if cfg.use_wandb:
                 wandb.log({
                     f"episode_success/{original_task_description}": 1 if done else 0,
                     f"plan_completed_by_vlm/{original_task_description}": 1 if plan_completed_by_vlm else 0,
                     "step": total_episodes # Log against total episodes
                 })


        # --- Task End ---
        # Log final task results
        task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
        print(f"Task {task_id} ('{original_task_description}') finished. Task Success Rate: {task_success_rate:.2f} ({task_successes}/{task_episodes})")
        log_file.write(f"\nTask {task_id} Finished. Task Success Rate: {task_success_rate:.2f} ({task_successes}/{task_episodes})\n")

        # Log task summary to W&B
        if cfg.use_wandb:
            wandb.log({
                 f"task_summary/success_rate": task_success_rate,
                 f"task_summary/num_successes": task_successes,
                 f"task_summary/num_episodes": task_episodes,
                 # Log against task_id or a global step if preferred
            }, step=task_id) # Log summary per task_id


    # --- Evaluation End ---
    # Log final total results
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    print(f"\nEvaluation Finished.")
    print(f"Total Success Rate: {total_success_rate:.2f} ({total_successes}/{total_episodes})")
    log_file.write(f"\nEvaluation Finished.\n")
    log_file.write(f"Total Success Rate: {total_success_rate:.2f} ({total_successes}/{total_episodes})\n")
    log_file.flush()

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "total_summary/success_rate": total_success_rate,
                "total_summary/num_successes": total_successes,
                "total_summary/num_episodes": total_episodes,
            }
        )
        wandb.save(local_log_filepath)
        wandb.finish()


if __name__ == "__main__":
    
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.perf_counter() 

    eval_libero()

    end_time = time.perf_counter() 
    duration = end_time - start_time
    print(f"\n end: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"eval_libero total_time: {duration:.2f} seconds")