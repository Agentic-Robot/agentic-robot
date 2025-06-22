from openai import OpenAI
import os
import re

# 初始化OpenAI客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# === 输入任务描述 ===
# task_description = "Pick up the alphabet soup and place it in the basket."
task_description = "Grasp the spoon."

# === Prompt 构造 ===
task_decomposition_prompt = f"""
You are a reasoning assistant for a fixed robotic arm that performs tabletop manipulation tasks.

You will:
1. First determine whether the task is **simple** (can be completed in one step like "grasp the cup") or **complex** (requires multiple actions like "pick up the cup and put it in the tray").
2. If the task is simple, just return: SIMPLE TASK: Grasp(object)
3. If the task is complex, decompose it into low-level action steps using the following predicates:

Available actions:
- MoveTo(target)
- Grasp(object)
- Release(object)

Example 1:
Task: "Grasp the apple"
Output: SIMPLE TASK: Grasp(apple)

Example 2:
Task: "Pick up the red cube and place it on the blue tray"
Output:
1. MoveTo(red_cube)
2. Grasp(red_cube)
3. MoveTo(blue_tray)
4. Release(red_cube)

Now reason and respond for:
Task: "{task_description}"
Output:
"""

reasoning_content = ""
answer_content = ""
is_answering = False

# 创建聊天完成请求
completion = client.chat.completions.create(
    model="deepseek-r1",
    messages=[
        {"role": "user", "content": task_decomposition_prompt}
    ],
    stream=True,
)

print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

for chunk in completion:
    if not chunk.choices:
        print("\nUsage:")
        print(chunk.usage)
    else:
        delta = chunk.choices[0].delta
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
            print(delta.reasoning_content, end='', flush=True)
            reasoning_content += delta.reasoning_content
        else:
            if delta.content != "" and is_answering is False:
                print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                is_answering = True
            print(delta.content, end='', flush=True)
            answer_content += delta.content

# ========== ✅ 后处理：解析动作序列 ==========

def parse_low_level_actions(text):
    text = text.strip()
    actions = []

    # Case 1: simple task
    if text.startswith("SIMPLE TASK:"):
        match = re.search(r"SIMPLE TASK:\s*(Grasp\([^)]+\))", text)
        if match:
            actions.append(match.group(1))
    else:
        # Case 2: multi-step task (starts with 1.)
        lines = text.splitlines()
        for line in lines:
            match = re.search(r"\d+\.\s*(\w+\([^)]+\))", line)
            if match:
                actions.append(match.group(1))
    return actions

# 执行解析
print("\n" + "=" * 20 + "解析后的动作序列" + "=" * 20 + "\n")
parsed_actions = parse_low_level_actions(answer_content)
print(parsed_actions)
