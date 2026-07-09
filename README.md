# roboticsreachy-mini-neu

Reachy Mini 机器人（东软定制版）项目合集，包含对话应用、意图路由、模块评测及主动沟通方案。

## 目录结构

- **roboticsreachy-mini-chatbox-neusoft** — Reachy Mini 对话应用核心代码，包括级联语音交互（ASR → LLM → TTS）、多模型支持、记忆系统、动作控制等完整功能。
- **Router** — 基于 LoRA 微调（chinese-bert-wwm-ext）的中文意图路由器，用于将用户对话意图分类到对应处理模块，包含训练、测试、数据生成与标注工具。
- **module-evaluation** — 各模块的评测基准及测试代码，涵盖 ASR（jiwer）、LLM（OpenCompass）、TTS（自定义）三个方向的性能评测。
- **机器人主动沟通方案** — 机器人主动发起对话的设计方案文档，描述主动沟通的行为规则与触发机制。
