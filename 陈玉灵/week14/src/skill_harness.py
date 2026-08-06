#!/usr/bin/env python3
"""渐进式加载并执行 SKILL.md 定义的技能。

可用于解析 skill 的元信息、触发场景、目标单词，并调用本地脚本生成输出。

用法示例：
  python skill_harness.py "帮我做 crazy 的闪卡"
  python skill_harness.py --word resilient
  python skill_harness.py --word thrill --output output.html
  python skill_harness.py --list

参数说明：
  input       可选，用户自然语言输入，用于识别要执行的 skill。
  --skill     指定 SKILL.md 文件路径，默认 skills/SKILL.md。
  --word      指定要生成的单词，跳过输入解析。
  --output    指定 HTML 输出路径，默认 output/<word>.html。
  --no-browser 生成后不自动打开默认浏览器。
  --list      列出当前技能可用的数据词。
"""
import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
import urllib.error
import webbrowser
from pathlib import Path

from openai import OpenAI

# 默认路径与文件名常量
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_SKILL_PATH = REPO_ROOT / "skills" / "SKILL.md"
DEFAULT_DATA_DIR_NAME = "data"
DEFAULT_SCRIPTS_DIR_NAME = "scripts"
DEFAULT_SCRIPT_NAME = "make_flashcard.py"
DEFAULT_OUTPUT_DIR_NAME = "output"
DEFAULT_OUTPUT_FILENAME_TEMPLATE = "{word}.html"

# 用于识别 SKILL.md 中的 YAML front matter 区块
FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)
# 提取用户输入中的英文单词候选项
ENGLISH_WORD_RE = re.compile(r"\b[a-zA-Z]+\b")

# 识别当前 skill 的常见触发关键词
SKILL_KEYWORDS = [
    "闪卡",
    "flash card",
    "flashcard",
    "单词卡",
    "生成",
    "做",
]

STOP_WORDS = {"flash", "card", "word", "words", "flashcard"}


def parse_front_matter(markdown_text: str) -> dict:
    """解析 SKILL.md 文件头部的 YAML front matter，返回键值字典。"""
    match = FRONT_MATTER_RE.search(markdown_text)
    if not match:
        return {}
    # front matter 区块的每一行可能包含普通键值或多行文本块
    front = {}
    lines = match.group(1).splitlines()
    pending_key = None
    pending_style = None
    pending_lines: list[str] = []

    def flush_pending() -> None:
        nonlocal pending_key, pending_style, pending_lines
        if pending_key is None:
            return
        # 处理多行文本块内容，将缩进部分合并成一句话
        if pending_style in {">", ">-", "|"}:
            front[pending_key] = " ".join(line.strip() for line in pending_lines).strip()
        else:
            front[pending_key] = pending_lines[0].strip() if pending_lines else ""
        pending_key = None
        pending_style = None
        pending_lines = []

    for line in lines:
        if pending_key is not None and (line.startswith(" ") or line.startswith("\t")):
            pending_lines.append(line)
            continue
        flush_pending()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        pending_key = key.strip()
        value = value.strip()
        if value in {">", ">-", "|"}:
            pending_style = value
            pending_lines = []
            continue
        front[pending_key] = value.strip().strip('"\'')
        pending_key = None

    flush_pending()
    return front


def extract_trigger_examples(markdown_text: str) -> list[str]:
    """从 SKILL.md 中提取触发场景示例，用于输出展示或调试。"""
    lines = markdown_text.splitlines()
    triggers = []
    collecting = False
    for line in lines:
        if line.strip().startswith("##") and "触发场景" in line:
            collecting = True
            continue
        if collecting:
            if line.strip().startswith("##"):
                break
            stripped = line.strip()
            if stripped.startswith("-"):
                triggers.append(stripped[1:].strip())
    return triggers


def find_english_word_candidates(text: str) -> list[str]:
    """从输入字符串中提取所有英文单词候选项（大小写统一为小写）。"""
    return [w.lower() for w in ENGLISH_WORD_RE.findall(text)]


def normalize_text(text: str) -> str:
    """对用户输入做基础规范化，统一空白和小写处理。"""
    return text.strip().replace("\u3000", " ").lower()


class SkillHarness:
    """技能执行器，负责加载 skill 定义、数据目录、以及执行生成流程。"""

    def __init__(self, skill_file: Path):
        self.skill_file = skill_file
        self.skill_root = self._resolve_workspace_root(skill_file)
        self.data_dir = self.skill_root / DEFAULT_DATA_DIR_NAME
        self.script_path = self.skill_root / DEFAULT_SCRIPTS_DIR_NAME / DEFAULT_SCRIPT_NAME
        self.skill_text = skill_file.read_text(encoding="utf-8")
        self.metadata = parse_front_matter(self.skill_text)
        self.triggers = extract_trigger_examples(self.skill_text)
        self.available_words = self._discover_available_words()

    def _resolve_workspace_root(self, skill_file: Path) -> Path:
        """Resolve workspace root when SKILL.md is stored under skills/ and data/scripts live at root."""
        if skill_file.parent.name == "skills":
            root = skill_file.parent.parent
            if (root / DEFAULT_DATA_DIR_NAME).exists() and (root / DEFAULT_SCRIPTS_DIR_NAME).exists():
                return root
        for ancestor in skill_file.parents:
            if ancestor.name == "skills":
                candidate = ancestor.parent
                if (candidate / DEFAULT_DATA_DIR_NAME).exists() and (candidate / DEFAULT_SCRIPTS_DIR_NAME).exists():
                    return candidate
        return skill_file.parent

    def _discover_available_words(self) -> list[str]:
        """扫描 data 目录，提取所有可用的单词文件名。"""
        if not self.data_dir.exists():
            return []
        return sorted(p.stem for p in self.data_dir.glob("*.json") if p.is_file())

    def is_relevant_input(self, user_input: str) -> bool:
        """判断用户输入是否与本 skill 的触发词或词条相关。"""
        normalized = normalize_text(user_input)
        if any(keyword in normalized for keyword in SKILL_KEYWORDS):
            return True
        return any(word in normalized for word in self.available_words)

    def extract_word(self, user_input: str) -> str | None:
        """从用户输入中提取单词候选，优先匹配已存在的数据词，必要时返回任意英文单词。"""
        normalized = normalize_text(user_input)
        tokens = find_english_word_candidates(normalized)
        candidates = [w for w in tokens if w not in STOP_WORDS]
        known = [w for w in candidates if w in self.available_words]
        if known:
            return known[0]
        if candidates:
            return candidates[0]
        # 尝试用 data 文件名做全词匹配，支持没有英文空格分隔的场景
        for word in self.available_words:
            if re.search(rf"\b{re.escape(word)}\b", normalized):
                return word
        return None

    def load_data(self, word: str) -> dict:
        """载入指定单词对应的 JSON 数据文件。"""
        data_path = self.data_dir / f"{word}.json"
        if not data_path.exists():
            raise FileNotFoundError(f"没有找到数据文件: {data_path}")
        with data_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def fetch_word_from_deepseek(self, word: str) -> dict:
        """调用 Deepseek 大模型接口查询单词资料，并返回可直接保存的 JSON 数据。"""
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        MODEL = os.getenv("AGENT_MODEL", "deepseek-v4-flash")
        prompt = (
            f"你是一个英语单词闪卡生成器。请为英语单词 ‘{word}’ 返回一个 JSON 对象，"
            "严格按照下列结构输出，且不要输出多余的说明文本：\n"
            "{\n"
            "  \"word\": \"...\",\n"
            "  \"phonetic\": \"...\",\n"
            "  \"pos\": \"...\",\n"
            "  \"definition\": \"...\",\n"
            "  \"examples\": [\n"
            "    {\"en\": \"...\", \"zh\": \"...\"},\n"
            "    {\"en\": \"...\", \"zh\": \"...\"},\n"
            "    {\"en\": \"...\", \"zh\": \"...\"}\n"
            "  ],\n"
            "  \"synonyms\": [\"...\", \"...\", \"...\"]\n"
            "}\n"
            "解释为中文，并提供 3 条中英对照例句，例句应包含该单词。"
        )

        payload = {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": "你是一个专业的英语词汇信息提取助手。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 800,
        }
        response = client.chat.completions.create(**payload)
        content = response.choices[0].message.content
        return self._parse_deepseek_response(content, word)

    def _parse_deepseek_response(self, response_text: str, expected_word: str) -> dict:
        """解析 Deepseek 返回内容，并验证为符合 Flash Card 格式的字典。"""
        cleaned = response_text.strip()
        json_block = re.search(r"```json\s*(.*?)\s*```", cleaned, re.S)
        if json_block:
            cleaned = json_block.group(1)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Deepseek 返回的内容无法解析为 JSON: {exc}\n原始内容:\n{cleaned}")

        if not isinstance(data, dict):
            raise RuntimeError("Deepseek 返回的数据不是 JSON 对象。")

        data["word"] = data.get("word", expected_word).strip().lower()
        if data["word"] != expected_word.lower():
            data["word"] = expected_word.lower()

        data.setdefault("phonetic", "")
        data.setdefault("pos", "")
        data.setdefault("definition", "")
        data.setdefault("examples", [])
        data.setdefault("synonyms", [])

        if not isinstance(data["examples"], list):
            raise RuntimeError("Deepseek 返回的 examples 字段不是列表。")
        if not isinstance(data["synonyms"], list):
            raise RuntimeError("Deepseek 返回的 synonyms 字段不是列表。")

        normalized_examples = []
        for example in data["examples"][:3]:
            if not isinstance(example, dict):
                continue
            normalized_examples.append({
                "en": str(example.get("en", "")).strip(),
                "zh": str(example.get("zh", "")).strip(),
            })
        data["examples"] = normalized_examples

        data["synonyms"] = [str(s).strip() for s in data["synonyms"] if str(s).strip()]
        return data

    def save_data_file(self, word_data: dict) -> Path:
        """保存 Deepseek 生成的数据到 data 目录。"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        data_path = self.data_dir / f"{word_data['word']}.json"
        with data_path.open("w", encoding="utf-8") as f:
            json.dump(word_data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        return data_path

    def build_output_path(self, word: str, output: str | None) -> Path:
        """生成最终 HTML 输出路径，默认写入仓库根目录的 output 目录。"""
        if output:
            output_path = Path(output).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return output_path

        output_dir = self.skill_root / DEFAULT_OUTPUT_DIR_NAME
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / DEFAULT_OUTPUT_FILENAME_TEMPLATE.format(word=word)

    def execute(self, word: str, output: Path, open_browser: bool) -> Path:
        """调用 make_flashcard.py 生成 HTML，并可选地自动打开浏览器。"""
        data_path = self.data_dir / f"{word}.json"
        if not data_path.exists():
            raise FileNotFoundError(f"skill data file not found: {data_path}")
        cmd = [sys.executable, str(self.script_path), str(data_path), "-o", str(output)]
        subprocess.run(cmd, check=True, cwd=self.skill_root)
        if open_browser:
            webbrowser.open_new_tab(output.as_uri())
        return output

    def pretty_print(self) -> str:
        """返回技能当前加载状态的描述文本。"""
        return (
            f"技能: {self.metadata.get('name', self.skill_file.stem)}\n"
            f"描述: {self.metadata.get('description', '')}\n"
            f"已知数据词: {', '.join(self.available_words) or '无'}\n"
            f"触发示例: {', '.join(self.triggers) or '无'}"
        )


def main() -> int:
    # 解析命令行参数并建立运行环境
    parser = argparse.ArgumentParser(
        description=(
            "渐进式加载并执行 SKILL.md 技能。"
            " 直接传入用户输入或指定单词，即可生成对应闪卡 HTML。"
        )
    )
    parser.add_argument("input", nargs="?", help="用户输入，用于识别要执行的技能。")
    parser.add_argument("--skill", default=str(DEFAULT_SKILL_PATH), help="技能定义文件路径，默认 skills/SKILL.md。")
    parser.add_argument("--word", help="指定要生成的单词，跳过输入解析。")
    parser.add_argument("--output", help="HTML 输出路径，默认 output/<word>.html。")
    parser.add_argument("--no-browser", action="store_true", help="生成后不自动打开浏览器。")
    parser.add_argument("--list", action="store_true", help="列出当前技能可用的数据词。")
    args = parser.parse_args()

    skill_file = Path(args.skill).expanduser().resolve()
    if not skill_file.exists():
        print(f"技能文件不存在: {skill_file}")
        return 1

    harness = SkillHarness(skill_file)

    if args.list:
        # 仅列出可用单词，不执行生成流程
        print("可用单词:")
        for word in harness.available_words:
            print(f"- {word}")
        return 0

    if args.word:
        # 显式指定单词时直接使用
        word = args.word.strip().lower()
    elif args.input:
        # 通过用户输入识别是否属于当前 skill，并尝试抽取目标单词
        if not harness.is_relevant_input(args.input):
            print("未识别到该输入属于当前技能。请确认输入包含闪卡、flash card、单词卡等关键词。")
            return 1
        word = harness.extract_word(args.input)
        if not word:
            print("未能从输入中提取单词，请使用已存在的数据文件名或加上 --word 参数。")
            return 1
    else:
        print("请提供用户输入，或使用 --word 指定单词。")
        parser.print_help()
        return 1

    if word not in harness.available_words:
        print(f"当前技能没有该单词的数据: {word}，尝试通过 Deepseek 获取词条信息...")
        try:
            word_data = harness.fetch_word_from_deepseek(word)
            saved_path = harness.save_data_file(word_data)
            harness.available_words.append(word)
            print(f"已将词条保存到: {saved_path}")
        except Exception as exc:
            print(f"Deepseek 查询失败: {exc}")
            print("可用单词:")
            for candidate in harness.available_words:
                print(f"- {candidate}")
            return 1

    output_path = harness.build_output_path(word, args.output)
    print("技能元信息:")
    print(harness.pretty_print())
    print(f"正在生成单词: {word}")
    print(f"输出路径: {output_path}")

    harness.execute(word, output_path, open_browser=not args.no_browser)
    print("执行完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
