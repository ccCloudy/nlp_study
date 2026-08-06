# SKILL 比较与闪卡生成示例

## 1. 生成闪卡示例

以下示例演示如何使用当前仓库中的技能生成一张闪卡：

```bash
python src/skill_harness.py --word crazy
```

此命令将：

- 读取 `skills/SKILL.md` 中的技能定义。
- 查找或生成 `data/<word>.json`。
- 调用 `scripts/make_flashcard.py` 生成 HTML 文件。
- 输出文件为 `output/<word>.html`。

## 2. 文件统计对比

| 项目 | 旧版 SKILL | 新版 SKILL | 变化 |
| --- | --- | --- | --- |
| 字符数 | 1272 | 358 | -914 (-71.9%) |
| 行数 | 59 | 25 | -34 (-57.6%) |
| 单词数 | 158 | 45 | -113 (-71.5%) |
| 正则 token 估计 | 274 | 95 | -179 (-65.3%) |
| tiktoken token 数量 | 714 | 188 | -526 (-73.7%) |

## 3. 性能对比

以下统计为 tokenization 函数的平均耗时，单位为毫秒。

| 项目 | 旧版 SKILL | 新版 SKILL | 变化 |
| --- | --- | --- | --- |
| 正则 tokenization 平均耗时 | 0.066 | 0.018 | -0.048 |
| tiktoken 编码平均耗时 | 0.321 | 0.113 | -0.208 |

## 4. 结论

- 正则 token 估计减少了 179 个，表明文本更简洁。
- tiktoken token 数量变化：-526。
- 正则 tokenization 时间变化：-0.048 ms。
- tiktoken 编码时间变化：-0.208 ms。

本文档为当前 SKILL 优化前后的 token 统计与性能对比，并给出具体的闪卡生成命令。