#!/usr/bin/env python3
"""Agent with subagent dispatch and parallel task execution.

本示例展示了如何实现一个 Master Agent，接收用户指令后拆分成多个子任务，
并通过 subagent 并行执行这些任务。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List

from openai import OpenAI

MODEL = os.getenv("AGENT_MODEL", "deepseek-v4-flash")
DEFAULT_MAX_TOKENS = 300


@dataclass
class SubAgentResult:
    task_id: int
    name: str
    goal: str
    status: str
    output: str
    elapsed: float


def get_deepseek_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("环境变量 DEEPSEEK_API_KEY 未设置，请先设置后再运行。")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def deepseek_chat(system_prompt: str, user_prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    client = get_deepseek_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def run_subtask_via_model(goal: str, context: str, output_directive: str) -> str:
    system_prompt = (
        "你是一个专业的中文文本处理助手。"
        "请根据用户给出的任务目标对文本进行处理，输出结果时不要附加额外解释。"
    )
    user_prompt = (
        f"任务目标：{goal}\n"
        "文本内容：\n"
        f"{context}\n\n"
        f"{output_directive}"
    )
    return deepseek_chat(system_prompt, user_prompt)


@dataclass
class SubAgent:
    task_id: int
    name: str
    goal: str
    context: str
    action: Callable[[str, str], str]

    def run(self) -> SubAgentResult:
        start = time.time()
        # 下发子任务给真实的 subagent 接口执行
        output = self.action(self.goal, self.context)
        elapsed = time.time() - start
        return SubAgentResult(
            task_id=self.task_id,
            name=self.name,
            goal=self.goal,
            status="completed",
            output=output,
            elapsed=elapsed,
        )


class MasterAgent:
    """Master Agent 负责任务分解、下发 subagent，并行执行结果汇总。"""

    def __init__(self, max_workers: int = 4, executor_type: str = "thread"):
        self.max_workers = max_workers
        self.executor_type = executor_type

    def plan_subtasks(self, instructions: str) -> List[SubAgent]:
        """根据用户指令生成子任务列表。"""
        text = instructions.strip()
        lower_text = text.lower()
        subtasks: List[SubAgent] = []

        if self._contains(lower_text, ["摘要", "总结", "summary"]):
            subtasks.append(SubAgent(
                task_id=1,
                name="摘要子代理",
                goal="生成一段精炼的文本摘要。",
                context=text,
                action=summarize_text,
            ))

        if self._contains(lower_text, ["关键词", "关键字", "keyword", "keywords"]):
            subtasks.append(SubAgent(
                task_id=2,
                name="关键词子代理",
                goal="抽取文本中的核心关键词。",
                context=text,
                action=extract_keywords,
            ))

        if self._contains(lower_text, ["情感", "情绪", "sentiment"]):
            subtasks.append(SubAgent(
                task_id=3,
                name="情感分析子代理",
                goal="分析文本的情感倾向。",
                context=text,
                action=analyze_sentiment,
            ))

        if self._contains(lower_text, ["翻译", "translate"]):
            subtasks.append(SubAgent(
                task_id=4,
                name="翻译子代理",
                goal="把文本翻译成英文。",
                context=text,
                action=translate_text,
            ))

        if self._contains(lower_text, ["提纲", "目录", "outline"]):
            subtasks.append(SubAgent(
                task_id=5,
                name="提纲子代理",
                goal="基于文本生成一个清晰的提纲。",
                context=text,
                action=generate_outline,
            ))

        if self._contains(lower_text, ["标题", "title"]):
            subtasks.append(SubAgent(
                task_id=6,
                name="标题子代理",
                goal="为文本生成一个吸引人的标题。",
                context=text,
                action=generate_title,
            ))

        if not subtasks:
            subtasks = [
                SubAgent(
                    task_id=1,
                    name="摘要子代理",
                    goal="生成一段简要摘要。",
                    context=text,
                    action=summarize_text,
                ),
                SubAgent(
                    task_id=2,
                    name="关键词子代理",
                    goal="提取文本的关键词。",
                    context=text,
                    action=extract_keywords,
                ),
                SubAgent(
                    task_id=3,
                    name="情感分析子代理",
                    goal="判断文本是否积极、消极或中性。",
                    context=text,
                    action=analyze_sentiment,
                ),
            ]

        return subtasks

    def _contains(self, text: str, patterns: List[str]) -> bool:
        return any(pattern in text for pattern in patterns)

    def dispatch_subagents(self, subtasks: List[SubAgent]) -> List[SubAgentResult]:
        """并行下发子任务给 subagent 执行。"""
        results: List[SubAgentResult] = []

        executor_cls = concurrent.futures.ThreadPoolExecutor
        if self.executor_type == "process":
            executor_cls = concurrent.futures.ProcessPoolExecutor

        with executor_cls(max_workers=self.max_workers) as executor:
            future_to_subagent = {executor.submit(subtask.run): subtask for subtask in subtasks}
            for future in concurrent.futures.as_completed(future_to_subagent):
                subagent = future_to_subagent[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover
                    result = SubAgentResult(
                        task_id=subagent.task_id,
                        name=subagent.name,
                        goal=subagent.goal,
                        status="failed",
                        output=str(exc),
                        elapsed=0.0,
                    )
                results.append(result)
        return sorted(results, key=lambda item: item.task_id)

    def run(self, instructions: str) -> Dict[str, object]:
        subtasks = self.plan_subtasks(instructions)
        started = time.time()
        results = self.dispatch_subagents(subtasks)
        total_time = time.time() - started
        return {
            "instructions": instructions,
            "executor_type": self.executor_type,
            "subagent_count": len(subtasks),
            "total_time_seconds": round(total_time, 3),
            "results": [asdict(result) for result in results],
        }


# ---------- 子代理模型任务执行 ----------

def summarize_text(goal: str, context: str) -> str:
    """调用 Deepseek 生成中文摘要。"""
    return run_subtask_via_model(
        goal,
        context,
        "请输出一段精炼的中文摘要，不超过 80 字，只返回摘要内容。",
    )


def extract_keywords(goal: str, context: str) -> str:
    """调用 Deepseek 提取关键词。"""
    return run_subtask_via_model(
        goal,
        context,
        "请提取文本中的 5 个核心关键词，用逗号分隔，不要解释。",
    )


def analyze_sentiment(goal: str, context: str) -> str:
    """调用 Deepseek 进行情感分析。"""
    return run_subtask_via_model(
        goal,
        context,
        "请判断文本的情感倾向：积极、消极或中性。仅返回一句话结果，不要额外说明。",
    )


def translate_text(goal: str, context: str) -> str:
    """调用 Deepseek 将文本翻译成英文。"""
    return run_subtask_via_model(
        goal,
        context,
        "请将以下中文文本翻译成英文，仅返回译文。",
    )


def generate_outline(goal: str, context: str) -> str:
    """调用 Deepseek 生成中文提纲。"""
    return run_subtask_via_model(
        goal,
        context,
        "请为文本生成一个清晰的中文提纲，每条以 '- ' 开头，最多 5 条。",
    )


def generate_title(goal: str, context: str) -> str:
    """调用 Deepseek 生成中文标题。"""
    return run_subtask_via_model(
        goal,
        context,
        "请为文本生成一个吸引人的中文标题，仅返回标题文本。",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Master Agent with subagent dispatch and parallel execution.")
    parser.add_argument(
        "--instructions",
        "-i",
        default="设计原则：\
1、找出应用中可能需要变化之处，把它们独立出来，不要和那些不需要变化的代码混在一起。（封装变化）。2、针对接口编程，而不是针对实现编程。3、多用组合，少用继承。策略模式：定义了算法族，分别封装起来，是他们之间可以相互替换，此设计模式让算法的变化独立于使用算法的客户。4、为了交互对象之间的松耦合设计而努力。5、类应该对扩展开放，对修改关闭。帮我提取出摘要和英文翻译。",
        help="用户指令，默认会生成摘要、关键词和情感分析。",
    )
    parser.add_argument("--workers", "-w", type=int, default=4, help="最多同时执行多少个 subagent。")
    parser.add_argument("--executor", "-e", choices=["thread", "process"], default="thread", help="并行执行器类型，可选 thread 或 process。")
    args = parser.parse_args()

    agent = MasterAgent(max_workers=args.workers, executor_type=args.executor)
    output = agent.run(args.instructions)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
