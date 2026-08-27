"""
附加 50 题验证：在同一批新题上对比 GRPO 前后效果，并输出报告文件。

作用：
  1. 生成 50 道新增验证题（覆盖 6 个难度）
  2. 分别评估基线模型 / GRPO 全量模型 / LoRA 模型
  3. 输出 JSON 报告，记录对比指标和增益

使用方式：
  python src/validate_50_questions.py
  python src/validate_50_questions.py --baseline-model E:\badou\pretrain_models\Qwen2-0.5B-Instruct --grpo-model outputs/grpo_ckpt --lora-model outputs/grpo_lora_ckpt
"""

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_baseline import (
    MODEL_PATH,
    SYSTEM_PROMPT,
    build_prompts,
    generate,
    make_problem,
    parse_output,
    LEVELS,
)

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "outputs_cyl"
DEFAULT_BASELINE_MODEL = MODEL_PATH
DEFAULT_GRPO_MODEL = OUT_DIR / "grpo_ckpt"
DEFAULT_LORA_MODEL = OUT_DIR / "grpo_lora_ckpt"
REPORT_PATH = OUT_DIR / "additional_50_question_report.json"
MARKDOWN_PATH = OUT_DIR / "additional_50_question_report.md"


def build_holdout_questions(seed: int = 2026, total: int = 50):
    """构建 50 题附加验证集，保证难度覆盖较均匀。"""
    rng = random.Random(seed)
    counts = {level: 8 for level in LEVELS}
    for level in ["L3_addsub_3digit", "L5_mul_2x1digit"]:
        counts[level] += 1

    questions = []
    for level, count in counts.items():
        for _ in range(count):
            expr, answer = make_problem(level, rng)
            questions.append({"level": level, "expr": expr, "answer": answer})

    rng.shuffle(questions)
    if len(questions) != total:
        raise ValueError(f"生成题目数不符，期望 {total}，实际 {len(questions)}")
    return questions


@torch.no_grad()
def evaluate_single_model(model_path, questions, k=8):
    """在一批问题上评估一个模型，返回聚合指标。"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if (Path(model_path) / "adapter_config.json").exists():
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            DEFAULT_BASELINE_MODEL,
            dtype=torch.bfloat16,
            device_map="cuda",
        )
        model = PeftModel.from_pretrained(base, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="cuda",
        )
    model.eval()

    prompts = []
    for q in questions:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"计算：{q['expr']} = ?"},
        ]
        prompts.append((q["expr"], q["answer"], msgs))

    texts = []
    for _, _, msgs in prompts:
        texts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    greedy_outputs = generate(model, tokenizer, texts, do_sample=False)
    greedy_format = greedy_loose = 0
    for q, out in zip(questions, greedy_outputs):
        fmt, _, loose = parse_output(out, q["answer"])
        greedy_format += int(fmt)
        greedy_loose += int(loose)

    sample_outputs = generate(model, tokenizer, texts, do_sample=True, k=k)
    sample_loose_sum = 0
    pass_at_k = 0
    mixed_groups = 0
    by_level = {level: {"n": 0, "loose_acc": 0, "pass@8": 0, "mixed_rate": 0} for level in LEVELS}

    for q, outs in zip(questions, sample_outputs):
        results = [parse_output(o, q["answer"]) for o in outs]
        n_loose = sum(r[2] for r in results)
        sample_loose_sum += n_loose
        pass_at_k += int(n_loose > 0)
        mixed_groups += int(0 < n_loose < k)

        lv = q["level"]
        by_level[lv]["n"] += 1
        by_level[lv]["loose_acc"] += n_loose
        by_level[lv]["pass@8"] += int(n_loose > 0)
        by_level[lv]["mixed_rate"] += int(0 < n_loose < k)

    total = len(questions)
    summary = {
        "model": str(model_path),
        "n_questions": total,
        "k": k,
        "greedy_format_rate": round(greedy_format / total, 4),
        "greedy_loose_acc": round(greedy_loose / total, 4),
        "sample_loose_acc": round(sample_loose_sum / (total * k), 4),
        "loose_pass@8": round(pass_at_k / total, 4),
        "informative_group_rate": round(mixed_groups / total, 4),
        "by_level": {
            level: {
                "n": stats["n"],
                "greedy_loose_acc": round((sum(parse_output(out, next(q["answer"] for q in questions if q["level"] == level), 0) for out in []) if False else 0), 4),
                "sample_loose_acc": round(stats["loose_acc"] / (stats["n"] * k), 4) if stats["n"] else 0.0,
                "loose_pass@8": round(stats["pass@8"] / stats["n"], 4) if stats["n"] else 0.0,
                "informative_group_rate": round(stats["mixed_rate"] / stats["n"], 4) if stats["n"] else 0.0,
            }
            for level, stats in by_level.items()
        },
    }
    return summary


def render_markdown(report):
    """生成美观的 Markdown 报告。"""
    base = report["baseline"]
    grpo = report["grpo"]
    lora = report.get("lora")

    def delta(label, a, b):
        return f"{a - b:+.3f}"

    lines = [
        "# GRPO 附加 50 题验证报告",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 题目总数：{report['n_questions']}",
        f"- 难度分布：{dict(report['difficulty_distribution'])}",
        "",
        "## 1. 总体指标对比",
        "",
        "| 模型 | greedy_format | greedy_loose_acc | sample_loose_acc | loose_pass@8 | informative_group_rate |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 基线 | {base['greedy_format_rate']:.3f} | {base['greedy_loose_acc']:.3f} | {base['sample_loose_acc']:.3f} | {base['loose_pass@8']:.3f} | {base['informative_group_rate']:.3f} |",
        f"| GRPO | {grpo['greedy_format_rate']:.3f} | {grpo['greedy_loose_acc']:.3f} | {grpo['sample_loose_acc']:.3f} | {grpo['loose_pass@8']:.3f} | {grpo['informative_group_rate']:.3f} |",
    ]

    if lora:
        lines.append(f"| LoRA | {lora['greedy_format_rate']:.3f} | {lora['greedy_loose_acc']:.3f} | {lora['sample_loose_acc']:.3f} | {lora['loose_pass@8']:.3f} | {lora['informative_group_rate']:.3f} |")

    lines.extend([
        "",
        "## 2. 与基线的增益",
        "",
        "| 指标 | GRPO vs 基线 |",
        "|---|---:|",
        f"| greedy_format_rate | {delta('greedy_format_rate', grpo['greedy_format_rate'], base['greedy_format_rate'])} |",
        f"| greedy_loose_acc | {delta('greedy_loose_acc', grpo['greedy_loose_acc'], base['greedy_loose_acc'])} |",
        f"| sample_loose_acc | {delta('sample_loose_acc', grpo['sample_loose_acc'], base['sample_loose_acc'])} |",
        f"| loose_pass@8 | {delta('loose_pass@8', grpo['loose_pass@8'], base['loose_pass@8'])} |",
        f"| informative_group_rate | {delta('informative_group_rate', grpo['informative_group_rate'], base['informative_group_rate'])} |",
    ])

    if lora:
        lines.extend([
            "",
            "| 指标 | LoRA vs 基线 |",
            "|---|---:|",
            f"| greedy_format_rate | {delta('greedy_format_rate', lora['greedy_format_rate'], base['greedy_format_rate'])} |",
            f"| greedy_loose_acc | {delta('greedy_loose_acc', lora['greedy_loose_acc'], base['greedy_loose_acc'])} |",
            f"| sample_loose_acc | {delta('sample_loose_acc', lora['sample_loose_acc'], base['sample_loose_acc'])} |",
            f"| loose_pass@8 | {delta('loose_pass@8', lora['loose_pass@8'], base['loose_pass@8'])} |",
            f"| informative_group_rate | {delta('informative_group_rate', lora['informative_group_rate'], base['informative_group_rate'])} |",
        ])

    lines.extend([
        "",
        "## 3. 结论",
        "",
        "- 若 GRPO 的 `greedy_loose_acc` 与 `loose_pass@8` 明显高于基线，说明强化学习确实提升了题目解题能力；",
        "- 若格式率提升幅度大于正确率提升，说明模型更快学习到了输出规范；",
        "- 若未训练难度（例如 L4/L6）也有提升，则说明不同程度的迁移能力；",
        "- 若 L6 仍低，表明模型在高难度乘法边界上仍受能力限制。",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-model", type=str, default=str(DEFAULT_BASELINE_MODEL), help="基线模型路径")
    parser.add_argument("--grpo-model", type=str, default=str(DEFAULT_GRPO_MODEL), help="GRPO 完成后的模型路径")
    parser.add_argument("--lora-model", type=str, default=str(DEFAULT_LORA_MODEL), help="LoRA 模型路径（可选）")
    parser.add_argument("--seed", type=int, default=2026, help="附加验证集随机种子")
    parser.add_argument("--n", type=int, default=50, help="新增验证题数量")
    parser.add_argument("--k", type=int, default=8, help="pass@k 采样数")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    questions = build_holdout_questions(seed=args.seed, total=args.n)

    report = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "seed": args.seed,
        "n_questions": args.n,
        "difficulty_distribution": dict(Counter(q["level"] for q in questions)),
        "questions": questions,
        "baseline": evaluate_single_model(args.baseline_model, questions, k=args.k),
        "grpo": evaluate_single_model(args.grpo_model, questions, k=args.k),
    }

    if Path(args.lora_model).exists():
        report["lora"] = evaluate_single_model(args.lora_model, questions, k=args.k)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    markdown = render_markdown(report)
    with open(MARKDOWN_PATH, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"附加验证题已生成：{len(questions)} 题")
    print(f"JSON 报告：{REPORT_PATH}")
    print(f"Markdown 报告：{MARKDOWN_PATH}")
    print("\n总体总结:")
    print(f"基线: greedy_loose_acc={report['baseline']['greedy_loose_acc']:.3f}, loose_pass@8={report['baseline']['loose_pass@8']:.3f}")
    print(f"GRPO : greedy_loose_acc={report['grpo']['greedy_loose_acc']:.3f}, loose_pass@8={report['grpo']['loose_pass@8']:.3f}")
    if "lora" in report:
        print(f"LoRA : greedy_loose_acc={report['lora']['greedy_loose_acc']:.3f}, loose_pass@8={report['lora']['loose_pass@8']:.3f}")


if __name__ == "__main__":
    main()
