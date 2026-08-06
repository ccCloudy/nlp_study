#!/usr/bin/env python3
"""比较两个 SKILL 文件的 token 消耗和 tokenization 性能。

用法：
  python scripts/compare_skill_tokens.py
  python scripts/compare_skill_tokens.py --old skills/SKILL_OLD.md --new skills/SKILL.md
  python scripts/compare_skill_tokens.py --model deepseek-v4-flash --repeat 500
  python scripts/compare_skill_tokens.py --doc compare_report.md

本脚本输出：
  - 字符数
  - 行数
  - 单词数
  - 基于正则的 token 估计
  - 可选的 tiktoken token 计数
  - tokenization 平均耗时
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import time
from pathlib import Path

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None


def read_text(path: Path) -> str:
    """读取文件并以 UTF-8 解码返回文本内容。"""
    return path.read_text(encoding="utf-8")


def normalize_text(text: str) -> str:
    """统一换行格式，便于后续统计。"""
    return text.replace("\r\n", "\n")


def regex_tokenize(text: str) -> list[str]:
    """使用正则表达式对文本进行简单分词。"""
    return re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]+", text)


def regex_token_count(text: str) -> int:
    """返回正则分词后的 token 数量。"""
    return len(regex_tokenize(text))


def tiktoken_encoding(name: str):
    """获取指定模型对应的 tiktoken 编码器。

    对于不在 tiktoken 模型映射中的模型名，会尝试使用常见编码器替代。
    """
    if tiktoken is None:
        raise RuntimeError("未安装 tiktoken。请运行 `pip install tiktoken` 安装。")

    fallback_map = {
        "deepseek-v4-flash": "cl100k_base",
        "gpt-4": "cl100k_base",
        "gpt-4o": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-3.5-turbo-0301": "cl100k_base",
    }

    try:
        if hasattr(tiktoken, "encoding_for_model"):
            return tiktoken.encoding_for_model(name)
        return tiktoken.get_encoding(name)
    except Exception:
        if name in fallback_map:
            return tiktoken.get_encoding(fallback_map[name])
        raise RuntimeError(
            f"无法自动映射模型 {name} 到 tiktoken 编码器。请使用 --model 传入已知编码器名称，或安装更新版本的 tiktoken。"
        )


def tiktoken_token_count(text: str, model: str) -> int:
    """使用 tiktoken 计算文本的 token 数量。"""
    if tiktoken is None:
        raise RuntimeError("未安装 tiktoken。")
    encoder = tiktoken_encoding(model)
    return len(encoder.encode(text))


def measure_performance(func, arg: str, repeat: int) -> float:
    """测量函数执行耗时并返回平均值。"""
    durations = []
    for _ in range(repeat):
        start = time.perf_counter()
        func(arg)
        durations.append(time.perf_counter() - start)
    return statistics.mean(durations)


def summarize(path: Path, text: str, model: str | None, repeat: int) -> dict[str, object]:
    """生成文件统计信息摘要，包括 token 计数和耗时。"""
    normalized = normalize_text(text)
    summary = {
        "path": str(path),
        "chars": len(normalized),
        "lines": normalized.count("\n") + 1,
        "words": len(normalized.split()),
        "regex_tokens": regex_token_count(normalized),
        "regex_token_time_s": measure_performance(regex_token_count, normalized, repeat),
    }
    if model is not None:
        summary["model"] = model
        try:
            summary["tiktoken_tokens"] = tiktoken_token_count(normalized, model)
            summary["tiktoken_token_time_s"] = measure_performance(lambda text: tiktoken_token_count(text, model), normalized, max(10, repeat // 10))
        except Exception as exc:
            summary["tiktoken_error"] = str(exc)
    return summary


def print_summary(summary: dict[str, object]) -> None:
    print(f"路径: {summary['path']}")
    print(f"  字符数: {summary['chars']}")
    print(f"  行数: {summary['lines']}")
    print(f"  单词数: {summary['words']}")
    print(f"  正则 token 估计: {summary['regex_tokens']}")
    print(f"  正则 tokenization 平均耗时: {summary['regex_token_time_s'] * 1e3:.3f} ms")
    if "tiktoken_tokens" in summary:
        print(f"  tiktoken token 数量 ({summary['model']}): {summary['tiktoken_tokens']}")
        print(f"  tiktoken 编码平均耗时: {summary['tiktoken_token_time_s'] * 1e3:.3f} ms")
    if "tiktoken_error" in summary:
        print(f"  tiktoken 错误: {summary['tiktoken_error']}")
    print()


def build_markdown_report(old: dict[str, object], new: dict[str, object], example_word: str = "crazy") -> str:
    """构建包含生成闪卡示例和统计对比的 Markdown 文档内容。"""
    lines = [
        "# SKILL 比较与闪卡生成示例",
        "",
        "## 1. 生成闪卡示例",
        "",
        "以下示例演示如何使用当前仓库中的技能生成一张闪卡：",
        "",
        "```bash",
        f"python src/skill_harness.py --word {example_word}",
        "```",
        "",
        "此命令将：",
        "",
        "- 读取 `skills/SKILL.md` 中的技能定义。",
        "- 查找或生成 `data/<word>.json`。",
        "- 调用 `scripts/make_flashcard.py` 生成 HTML 文件。",
        "- 输出文件为 `output/<word>.html`。",
        "",
        "## 2. 文件统计对比",
        "",
        "| 项目 | 旧版 SKILL | 新版 SKILL | 变化 |",
        "| --- | --- | --- | --- |",
    ]

    def format_delta(old_value, new_value):
        if isinstance(old_value, int) and isinstance(new_value, int):
            delta = new_value - old_value
            pct = (delta / old_value * 100) if old_value else 0.0
            return f"{delta:+} ({pct:+.1f}%)"
        return "-"

    rows = []
    for key, label in [
        ("chars", "字符数"),
        ("lines", "行数"),
        ("words", "单词数"),
        ("regex_tokens", "正则 token 估计"),
    ]:
        old_value = old.get(key, "-")
        new_value = new.get(key, "-")
        delta = format_delta(old_value, new_value)
        rows.append(f"| {label} | {old_value} | {new_value} | {delta} |")

    if "tiktoken_tokens" in old and "tiktoken_tokens" in new:
        old_value = old.get("tiktoken_tokens", "-")
        new_value = new.get("tiktoken_tokens", "-")
        delta = format_delta(old_value, new_value)
        rows.append(f"| tiktoken token 数量 | {old_value} | {new_value} | {delta} |")

    lines.extend(rows)
    lines.extend([
        "",
        "## 3. 性能对比",
        "",
        "以下统计为 tokenization 函数的平均耗时，单位为毫秒。",
        "",
        "| 项目 | 旧版 SKILL | 新版 SKILL | 变化 |",
        "| --- | --- | --- | --- |",
    ])

    perf_rows = []
    for key, label in [
        ("regex_token_time_s", "正则 tokenization 平均耗时"),
    ]:
        old_value = old.get(key, None)
        new_value = new.get(key, None)
        old_ms = f"{old_value * 1e3:.3f}" if isinstance(old_value, float) else "-"
        new_ms = f"{new_value * 1e3:.3f}" if isinstance(new_value, float) else "-"
        delta = "-"
        if isinstance(old_value, float) and isinstance(new_value, float):
            delta = f"{(new_value - old_value) * 1e3:+.3f}"
        perf_rows.append(f"| {label} | {old_ms} | {new_ms} | {delta} |")
    if "tiktoken_token_time_s" in old and "tiktoken_token_time_s" in new:
        old_value = old.get("tiktoken_token_time_s", None)
        new_value = new.get("tiktoken_token_time_s", None)
        old_ms = f"{old_value * 1e3:.3f}" if isinstance(old_value, float) else "-"
        new_ms = f"{new_value * 1e3:.3f}" if isinstance(new_value, float) else "-"
        delta = f"{(new_value - old_value) * 1e3:+.3f}" if isinstance(old_value, float) and isinstance(new_value, float) else "-"
        perf_rows.append(f"| tiktoken 编码平均耗时 | {old_ms} | {new_ms} | {delta} |")
    lines.extend(perf_rows)
    lines.append("")
    lines.append("## 4. 结论")
    lines.append("")
    conclusion = []
    if isinstance(old.get("regex_tokens"), int) and isinstance(new.get("regex_tokens"), int):
        delta = new["regex_tokens"] - old["regex_tokens"]
        conclusion.append(f"- 正则 token 估计减少了 {abs(delta)} 个，表明文本更简洁。" if delta < 0 else f"- 正则 token 估计增加了 {delta} 个。")
    if "tiktoken_tokens" in old and "tiktoken_tokens" in new and isinstance(old["tiktoken_tokens"], int) and isinstance(new["tiktoken_tokens"], int):
        delta = new["tiktoken_tokens"] - old["tiktoken_tokens"]
        conclusion.append(f"- tiktoken token 数量变化：{delta:+}。")
    if isinstance(old.get("regex_token_time_s"), float) and isinstance(new.get("regex_token_time_s"), float):
        delta_ms = (new["regex_token_time_s"] - old["regex_token_time_s"]) * 1e3
        conclusion.append(f"- 正则 tokenization 时间变化：{delta_ms:+.3f} ms。")
    if "tiktoken_token_time_s" in old and "tiktoken_token_time_s" in new and isinstance(old["tiktoken_token_time_s"], float) and isinstance(new["tiktoken_token_time_s"], float):
        delta_ms = (new["tiktoken_token_time_s"] - old["tiktoken_token_time_s"]) * 1e3
        conclusion.append(f"- tiktoken 编码时间变化：{delta_ms:+.3f} ms。")
    if not conclusion:
        conclusion.append("- 未能计算出显著差异，请检查输入文件或 tiktoken 可用性。")
    lines.extend(conclusion)
    lines.append("")
    lines.append("本文档为当前 SKILL 优化前后的 token 统计与性能对比，并给出具体的闪卡生成命令。")
    return "\n".join(lines)


def diff_summary(old: dict[str, object], new: dict[str, object]) -> None:
    """输出旧版与新版统计的对比结果。"""
    print("对比结果：")
    for key in ["chars", "lines", "words", "regex_tokens", "tiktoken_tokens"]:
        if key in old and key in new:
            old_value = old[key]
            new_value = new[key]
            if isinstance(old_value, int) and isinstance(new_value, int):
                delta = new_value - old_value
                pct = (delta / old_value * 100) if old_value else float('inf')
                print(f"  {key}: old={old_value}, new={new_value}, delta={delta} ({pct:+.1f}%)")
    print("性能对比：")
    for key in ["regex_token_time_s", "tiktoken_token_time_s"]:
        if key in old and key in new:
            old_time = old[key]
            new_time = new[key]
            delta = new_time - old_time
            pct = (delta / old_time * 100) if old_time else float('inf')
            print(f"  {key}: old={old_time*1e3:.3f} ms, new={new_time*1e3:.3f} ms, delta={delta*1e3:.3f} ms ({pct:+.1f}%)")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较 SKILL 文件的 token 消耗和 tokenization 性能。")
    parser.add_argument("--old", default="skills/SKILL_OLD.md", help="旧版 SKILL 文件路径。")
    parser.add_argument("--new", default="skills/SKILL.md", help="新版 SKILL 文件路径。")
    parser.add_argument("--model", default="deepseek-v4-flash", help="用于 tiktoken 编码的模型名称。")
    parser.add_argument("--no-tiktoken", action="store_true", help="即使 tiktoken 可用也跳过它。")
    parser.add_argument("--repeat", type=int, default=100, help="性能测量的重复次数。")
    parser.add_argument("--doc", help="将结果写入指定 Markdown 文档。")
    parser.add_argument("--example-word", default="crazy", help="文档生成示例中使用的单词。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    old_path = Path(args.old)
    new_path = Path(args.new)

    if not old_path.exists():
        print(f"未找到旧版 SKILL 文件：{old_path}")
        return 1
    if not new_path.exists():
        print(f"未找到新版 SKILL 文件：{new_path}")
        return 1

    old_text = read_text(old_path)
    new_text = read_text(new_path)
    model = None if args.no_tiktoken else args.model

    old_summary = summarize(old_path, old_text, model, args.repeat)
    new_summary = summarize(new_path, new_text, model, args.repeat)

    print_summary(old_summary)
    print_summary(new_summary)
    diff_summary(old_summary, new_summary)

    if args.doc:
        report = build_markdown_report(old_summary, new_summary, example_word=args.example_word)
        Path(args.doc).write_text(report, encoding="utf-8")
        print(f"已生成文档: {args.doc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
