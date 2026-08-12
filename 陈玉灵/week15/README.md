# week15: Subagent Agent

本目录实现了一个 Master Agent 示例，它可以：

- 将用户指令拆分为多个子任务（subagent）
- 并行下发这些子任务给子代理执行
- 汇总子代理执行结果，返回结构化输出

## 文件说明

- `subagent_agent.py`：主程序，包含 MasterAgent、SubAgent、任务规划与并行执行逻辑。
  - 子任务真实调用 `deepseek-v4-flash` 模型 API，通过 `OpenAI` SDK 向 Deepseek 发送 chat completion 请求。

## 环境准备

- 需要安装 `openai` Python SDK：

```bash
pip install openai
```

- 需要设置 Deepseek API key：

```powershell
$env:DEEPSEEK_API_KEY="sk-xxx"
```

## 运行示例

```bash
python subagent_agent.py
```

如果需要自定义指令：

```bash
python subagent_agent.py --instructions "今天天气真好" --workers 3 --executor process
```

## 说明

本示例使用标准库 `concurrent.futures.ThreadPoolExecutor` 实现并行执行，
并通过 `SubAgent` 类封装真实的子代理模型调用。实际应用中可将 `action` 替换为更多自定义任务、工具链或远程服务调用。