<h1 align="center">Agentic Robot: A Brain-Inspired Framework for Vision-Language-Action Models in Embodied Agents</h1>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/python-blue.svg" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/pytorch-orange.svg" alt="PyTorch"></a>
</p>


<p align="center"><a href="https://github.com/Agentic-Robot/agentic-robot">🏠 Project Page</a> • <a href="https://arxiv.org/abs/2505.23450">📄 Paper(Arxiv)</a> • <a href="https://agentic-robot.github.io/">🌐 Website</a> • 
</p>


<p align="center">Zhejian Yang, Yongchao Chen, Xueyang Zhou, Jiangyue Yan, Dingjie Song, Yinuo Liu, Yuting Li, Yu Zhang, Pan Zhou, Hechang Chen*, Lichao Sun</p>

---

![Agentic Robot](./figures/overview.png)

Long-horizon robotic manipulation poses significant challenges for autonomous systems, requiring extended reasoning, precise execution, and robust error recovery across complex sequential tasks. Current approaches, whether based on static planning or end-to-end visuomotor policies, suffer from error accumulation and lack effective verification mechanisms during execution, limiting their reliability in real-world scenarios. We present Agentic Robot, a brain-inspired framework that addresses these limitations through Standardized Action Procedures (SAP)—a novel coordination protocol governing component interactions throughout manipulation tasks. Drawing inspiration from Standardized Operating Procedures (SOPs) in human organizations, SAP establishes structured workflows for planning, execution, and verification phases. Our architecture comprises three specialized components: (1) a large reasoning model that decomposes high-level instructions into semantically coherent subgoals, (2) a vision-language-action executor that generates continuous control commands from real-time visual inputs, and (3) a temporal verifier that enables autonomous progression and error recovery through introspective assessment. This SAP-driven closed-loop design supports dynamic self-verification without external supervision. On the LIBERO benchmark, Agentic Robot achieves state-of-the-art performance with an average success rate of 79.6%, outperforming SpatialVLA by 6.1% and OpenVLA by 7.4% on long-horizon tasks. These results demonstrate that SAP-driven coordination between specialized components enhances both performance and interpretability in sequential manipulation, suggesting significant potential for reliable autonomous systems.

---

## ✨ News ✨

- **[2025/06/22]** 🤖 We open-sourced Agentic Robot v1.0 — we’ll continue improving the project with new ideas and updates, so feel free to follow and ⭐️ Star us to stay in the loop! [Agentic Robot v1.0](https://github.com/Agentic-Robot/agentic-robot)

## Quick Setup
Before running Agentic Robot, please make sure the required environments are properly set up:

🧠 Our project is built on top of [OpenVLA](https://github.com/moojink/openvla-oft?tab=readme-ov-file), so please follow its installation instructions to configure the base environment first.

🧪 Experiments are conducted in the [LIBERO](https://github.com/moojink/openvla-oft/blob/main/LIBERO.md) simulation environment. Make sure to install LIBERO and its dependencies as described in their official documentation.

## ## 🚀 Implementation

Our training framework consists of two stages:

1. Stage I: Decompose the complex task with LRM.  
2. Stage II: Evaluate Agentic Robot with VLA and VLM.

## Task Devision

Here, we take Deepseek-V3 as an example to decompose the complex task.

```python
python experiments/robot/libero/ds.py
```

**For convenience, we provide a test case that calls the LRM once, then generates a hard-coded plan and replaces it in the main.py file. For handling multiple tasks, we can add a function to call the LRM in main.py.**

## 📊 Evaluation

We evaluate the Agentic Robot on LIBERO benchmark with VLM (QwenVL-2.5) and VLA (OpenVLA).

Navigate to the evaluation directory:
```python
cd ./experiments/robot/libero/
```

# Launch LIBERO-Spatial evals

```python
python experiments/robot/libero/main.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True
```

# Launch LIBERO-Object evals

```python
python experiments/robot/libero/main.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --center_crop True
```

# Launch LIBERO-Goal evals

```python
python experiments/robot/libero/main.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --center_crop True
```

# Launch LIBERO-10 (LIBERO-Long) evals

```python
python experiments/robot/libero/main.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True
```

## Support

If you run into any issues, please open a new GitHub issue. If you do not receive a response within 2 business days, please email Zhejian Yang (JLU-Advisor@outlook.com) to bring the issue to his attention.

## Citation

If you use our code in your work, please cite [our paper](https://arxiv.org/abs/2505.23450):

```bibtex
@misc{yang2025agenticrobotbraininspiredframework,
      title={Agentic Robot: A Brain-Inspired Framework for Vision-Language-Action Models in Embodied Agents},
      author={Zhejian Yang and Yongchao Chen and Xueyang Zhou and Jiangyue Yan and Dingjie Song and Yinuo Liu and Yuting Li and Yu Zhang and Pan Zhou and Hechang Chen and Lichao Sun},
      year={2025},
      eprint={2505.23450},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.23450},
}
```
