---
name: flash-card
description: 生成英语单词静态 HTML 闪卡，含音标、词性、释义、3 条中英对照例句和近义词。
---

# Flash Card

触发词：闪卡、flash card、单词卡。

- 数据：`data/<word>.json`
- 渲染：`python scripts/make_flashcard.py data/<word>.json`
- 输出：`output/<word>.html`

字段：
- `word`
- `phonetic`
- `pos`
- `definition`
- `examples`：恰好 3 条中英对照例句
- `synonyms`：4-6 个

注意：
- 数据写入 `data/`
- 输出写入 `output/`
