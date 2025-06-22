import openai
import os
import re
import warnings 

# --- Helper Function ---
def parse_llm_plan(raw_plan_string: str) -> list[str]:

    subtasks = []
    # Regex to find lines starting with a number, period, space, then capture the rest
    pattern = re.compile(r"^\s*\d+\.\s+(.*)")

    for line in raw_plan_string.splitlines():
        stripped_line = line.strip()
        match = pattern.match(stripped_line)
        if match:
            instruction = match.group(1).strip()
    
            if instruction:
                subtasks.append(instruction)
    return subtasks

# --- Main Function ---
def decompose_task_with_llm(
    task_description: str,
    model_name: str = "deepseek-v3",
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
) -> list[str]:
    """
    """
    
    try:
        import openai
    except ImportError:
        raise ImportError("The 'openai' library is required. Please install it using 'pip install openai'.")

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        warnings.warn("Environment variable 'DASHSCOPE_API_KEY' not set. Cannot call LLM.", RuntimeWarning)
        return []

    try:
        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    except Exception as e:
        warnings.warn(f"Failed to initialize OpenAI client: {e}", RuntimeWarning)
        return []

    task_decomposition_prompt = f"""
You are a planning assistant for a fixed robotic arm. Your goal is to break down a high-level task into a sequence of **essential high-level commands**, suitable for a capable Vision-Language-Action (VLA) model to execute directly.

Output Format:
Generate a numbered list of commands. Each command should represent a significant action achieving a clear sub-goal. Stick to the allowed high-level actions.

Example Plan Format (Use **exactly** this level of granularity):
Plan for the robot arm:

Goal: <original instruction>

1. pick up the <object_name_1>
2. place the <object_name_1> in the <target_location>
3. pick up the <object_name_2>
4. place the <object_name_2> in the <target_location>
# --- Example for a different task ---
# Goal: Put the apple in the red bowl
# 1. pick up the apple
# 2. place the apple in the red bowl
# --- Example for another task ---
# Goal: Put the cup in the microwave and close it
# 1. pick up the cup
# 2. place the cup in the microwave
# 3. close the microwave
# --- Example for another task ---
# Goal: Turn on the stove and put the pot on it
# 1. turn on the stove
# 2. pick up the pot
# 3. place the pot on the stove

Instructions:
- Generate **only** high-level commands.
- **Allowed commands are strictly limited to:**
    - `pick up [object]`
    - `place [object] in/on [location]`
    - `open [object/container/drawer/etc.]`
    - `close [object/container/drawer/etc.]`
    - `turn on [device]`
    - `turn off [device]`
- Use the commands above **only when necessary** to achieve the goal. Most tasks will primarily use `pick up` and `place`.
- **Explicitly DO NOT include separate steps for:**
    - `locate` (Assume VLA finds the object as part of executing the command)
    - `move to` or `move towards` (Assume the command includes necessary travel)
    - `lift`, `lower`, `grasp`, `release`, `push`, `pull`, `rotate`, `adjust` (Assume high-level commands handle these internally)
- **Assume the VLA model handles all implicit actions:**
    - "pick up [object]" means: Find the object, navigate to it, grasp it securely, and lift it.
    - "place [object] in [location]" means: Transport the object to the location, position it correctly, and release the grasp.
    - "open/close [container]" means: Find the handle/seam, interact with it appropriately (pull, slide, lift) to change the container's state.
    - "turn on/off [device]" means: Find the correct button/switch, interact with it to change the device's power state.
- Use the descriptive names from the task description (e.g., "alphabet soup", "basket", "stove", "microwave", "bottom drawer").
- Generate the minimal sequence of these high-level commands required to fulfill the Goal. Ensure the sequence logically achieves the task (e.g., you might need to `open` a drawer before `place`ing something inside it, even if 'open' isn't explicitly stated in the goal).

Task: {task_description}
Output:
"""

    answer_content = ""
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": task_decomposition_prompt}
            ],
            stream=True,
        )

        for chunk in completion:

            if chunk.choices:
                delta = chunk.choices[0].delta

                if delta and delta.content is not None:
                    answer_content += delta.content

    except openai.APIConnectionError as e:
        warnings.warn(f"Failed to connect to OpenAI API: {e}", RuntimeWarning)
        return []
    except openai.RateLimitError as e:
        warnings.warn(f"OpenAI API request exceeded rate limit: {e}", RuntimeWarning)
        return []
    except openai.APIStatusError as e:
        warnings.warn(f"OpenAI API returned an error status: {e.status_code} - {e.response}", RuntimeWarning)
        return []
    except Exception as e:
        warnings.warn(f"An unexpected error occurred during LLM call: {e}", RuntimeWarning)
        return []

    if not answer_content:
         warnings.warn("LLM returned an empty response.", RuntimeWarning)
         return []

    parsed_subtasks = parse_llm_plan(answer_content)

    return parsed_subtasks

# --- Example Usage ---
if __name__ == "__main__":
    # export DASHSCOPE_API_KEY='your_actual_api_key'

    print("--- Example 1 ---")
    task1 = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
    print(f"Original Task: {task1}")
    subtasks1 = decompose_task_with_llm(task1)

    if subtasks1:
        print("Decomposed Subtasks:")
        for i, subtask in enumerate(subtasks1):
            print(f"  Step {i+1}: {subtask}")
    else:
        print("Failed to decompose task or no subtasks generated.")

    print("\n--- Example 2 ---")
    task2 = "put the green block into the yellow container"
    print(f"Original Task: {task2}")
    subtasks2 = decompose_task_with_llm(task2)

    if subtasks2:
        print("Decomposed Subtasks:")
        for i, subtask in enumerate(subtasks2):
            print(f"  Step {i+1}: {subtask}")
    else:
        print("Failed to decompose task or no subtasks generated.")

    print("\n--- Example 3 (API Key Missing Test) ---")

    original_key = os.environ.pop("DASHSCOPE_API_KEY", None)
    print(f"Original Task: {task1}")
    subtasks3 = decompose_task_with_llm(task1)
    if not subtasks3:
        print("Correctly handled missing API key.")

    if original_key:
        os.environ["DASHSCOPE_API_KEY"] = original_key