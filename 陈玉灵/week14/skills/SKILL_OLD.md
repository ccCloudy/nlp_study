---
name: flash-card
description: >-
  为一个英语单词生成静态 HTML 学习闪卡，含音标、词性、释义、3 条中英对照例句和近义词。
  触发词包括“闪卡”、“flash card”、“单词卡”等。
---

# Flash Card 单词闪卡生成

生成英语单词的静态 HTML 学习卡片，输出顺序：单词+音标 → 释义 → 近义词 → 3 条中英对照例句。

## 触发场景

当用户表达类似下面内容时触发本 skill：
- 给我做张 crazy 的闪卡
- 给我做 crazy 的 flash card
- 做一个 resilient 的单词卡
- 帮我生成 meticulous 的闪卡

## 执行流程

1. 从用户话语中提取目标单词，并小写化作为文件名。
2. 生成 `data/<word>.json`，字段：
   - `word`
   - `phonetic`
   - `pos`
   - `definition`
   - `examples`（3 条中英对照例句）
   - `synonyms`
3. 运行脚本生成 HTML：
   ```bash
   python scripts/make_flashcard.py data/<word>.json
   ```
   默认输出 `output/<word>.html`。
4. 自动打开默认浏览器预览结果。

## JSON 格式示例

```json
{
  "word": "resilient",
  "phonetic": "/rɪˈzɪliənt/",
  "pos": "adj.",
  "definition": "能迅速从困难、挫折中恢复过来的；有韧性的，适应力强的",
  "examples": [
    {"en": "She is a resilient child who bounces back quickly from setbacks.", "zh": "她是个有韧性的孩子，遇到挫折能很快恢复过来。"},
    {"en": "The economy proved remarkably resilient during the crisis.", "zh": "在危机期间，经济表现出了惊人的韧性。"},
    {"en": "A resilient mindset helps you cope with life's challenges.", "zh": "一种有韧性的心态能帮你应对生活中的挑战。"}
  ],
  "synonyms": ["tough", "flexible", "strong", "hardy", "buoyant", "springy"]
}
```

## 要点

- `examples` 必须恰好 3 条。
- `synonyms` 建议 4-6 个。
- 数据存放在仓库根目录 `data/`，输出写入 `output/`。
