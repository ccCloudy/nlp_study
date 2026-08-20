# GLM-5.2 (`glm_moe_dsa`) 模型实现技术分析报告

> 分析对象:`modeling_glm_moe_dsa.py`(共 827 行)
> 对应配置:`zai-org_GLM-5.2_config.json`(HF 生成文件,头部注释声明由 `modular_glm_moe_dsa.py` 自动生成,见 L1-6)
> 关键配置:hidden=6144 / layers=78 / heads=64 / kv_heads=64 / head_dim=192 / q_lora_rank=2048 / kv_lora_rank=512 / qk_nope=192 / qk_rope=64 / v_head_dim=256 / rope_interleave=true / vocab=154880 / max_pos=1,048,576 / rope_theta=8e6

---

## 0. 总体定位

该文件是 **GLM-5.2** 的 HF Transformers 模块化实现,模型类型为 `glm_moe_dsa`。结构上可以概括为:

- **DeepSeek-V3 风格 MLA 注意力**(低秩 Q/KV 压缩、nope/rope 分离)+ **DSA 稀疏注意力 indexer**(DeepSeek-V3.2 路线,但 RoPE 与 top-k 共享机制不同);
- **DeepSeek 风格 MoE**(sigmoid 无辅助损失路由 + 分组 top-k + 共享专家);
- **GLM 特有扩展:跨层 top-k 共享**(`indexer_types` full/shared 交替,每 4 层仅 1 个 full indexer),代码中反复以 `# MAIN DIFF with DSV3.2` 标注(L616/L629/L732/L741)。

---

## 1. 核心类/函数清单及职责

| 符号 | 行号 | 职责 |
|---|---|---|
| `GlmMoeDsaRMSNorm` | L48-65 | RMSNorm,fp32 累加(先转 fp32 算方差再转回,L57-62);带 `@use_kernel_forward_from_hub("RMSNorm")` 内核覆盖钩子(L47) |
| `GlmMoeDsaRotaryEmbedding` | L68-122 | RoPE 基频计算与 cos/sin 生成;支持 `rope_type` 扩展(YaRN/dynamic 等,L77-81);`inv_freq` 按 `rope_theta` 计算(L98-104);`emb=cat(freqs,freqs)` 产生双倍长度 cos/sin(L118) |
| `apply_rotary_pos_emb_interleave` | L125-161 | **interleaved RoPE 应用函数**:直接对 q/k 的奇偶切片做旋转(L156-160),与 de-interleave 版 `rotate_half` 逐位等价且免拷贝(L129-132) |
| `GlmMoeDsaIndexer` | L164-255 | **DSA 稀疏注意力 indexer**:轻量独立投影(复用 MLA 的 q_resid),输出每查询 top-k token 索引 `[B,S,topk]` int32(L254-255) |
| `repeat_kv` | L258-267 | KV 头重复(eager 路径用,本模型 groups=1 时为空操作) |
| `eager_attention_forward` | L270-292 | eager 注意力(兜底实现,`ALL_ATTENTION_FUNCTIONS` 按实现选择,L453-455) |
| `yarn_get_mscale` / `yarn_apply_mscale` | L295-308 | YaRN mscale 钩子(`rope_type=default` 时直接返回原 scaling,L302) |
| `GlmMoeDsaAttention` | L311-470 | **MLA + DSA indexer 融合层**,含跨层 top-k 共享逻辑(L313-318)与稀疏 mask 构造(L438-451) |
| `GlmMoeDsaMLP` | L473-486 | 标准 SwiGLU 稠密 MLP(gate/up/down,L479-485);前 3 层 dense 用(intermediate=12288) |
| `GlmMoeDsaTopkRouter` | L489-527 | **noaux_tc sigmoid 路由器**:分组 top-k、专家分数修正偏置、权重归一化与缩放 |
| `GlmMoeDsaExperts` | L530-567 | 路由专家集合:3D 参数 `gate_up_proj`(融合 gate+up, L539)与 `down_proj`(L540),按命中的专家循环计算 |
| `GlmMoeDsaMoE` | L570-591 | MoE 模块 = 路由专家 + 共享专家(`shared_experts`,L580-582),输出 = 路由 + 共享(L590) |
| `GlmMoeDsaDecoderLayer` | L594-638 | 解码层:输入/输出 RMSNorm + 注意力 + MLP(按 `mlp_layer_types` 选 MoE 或 dense,L600-603);**返回 `topk_indices` 供下一层复用**(L638) |
| `GlmMoeDsaPreTrainedModel` | L641-670 | 基类:注意力后端支持声明(L648-650)、fp32 模块声明(`indexer.weights_proj` L660、`e_score_correction_bias` L658)、权重初始化(L663-670) |
| `GlmMoeDsaModel` | L673-749 | 主干:embedding + 78 层 + 最终 norm + rotary;主循环内 `topk_indices` 自下而上传递(L732-743);因果 mask 以 `deepseek_sparse_attention` 为 key(L727) |
| `GlmMoeDsaForCausalLM` | L752-824 | LM Head(6144→154880,L763),`logits_to_keep` 切片计算(L811-812);含 TP/PP/FSDP 计划(L755-757) |

---

## 2. 注意力机制:MLA + DSA Indexer

### 2.1 MLA 主体结构(与 DeepSeek-V3 一致)

- **查询低秩分解**(L342-344):
  `q_a_proj: 6144→2048` → `q_a_layernorm(RMSNorm,2048)` → `q_b_proj: 2048→64×256=16384`
  (有 `q_lora_rank` 时走此路径,否则直接 `q_proj`,L339-340)
- **KV 低秩压缩**(L346-356):
  `kv_a_proj_with_mqa: 6144→(512+64)=576`(512 维 latent + 64 维 rope 分量, L346-350)
  → `kv_a_layernorm(RMSNorm,512)`(L351)→ `kv_b_proj: 512→64×(192+256)=28672`(L352-356)
- **K/V 展开 `expand_kv`**(L369-386):`kv_b_proj` 输出切成 `k_nope(192)` 与 `value(256)`(L379),`k_rot(64)` 广播到各头(L380),再拼成完整 key(192+64=256,L383-385)
- **输出投影** `o_proj: 64×256→6144`(L358-362);`num_key_value_groups = 64/64 = 1`(L335),即无 GQA 头重复
- **注意力维度**:query/key 头维 = `qk_nope(192)+qk_rope(64)=256`(L334),value 头维 = 256(L332);scaling = `256^-0.5`(L364,经 `yarn_apply_mscale` 钩子)

### 2.2 DSA Indexer 结构(L164-255)

Indexer 是与主 MLA **完全独立的轻量打分网络**,仅用于"选哪些 token 参与注意力":

- 参数(L189-193):
  - `wq_b: 2048→32×128=4096` —— 直接吃 MLA 的 `q_resid`(q_a_layernorm(q_a_proj(x)) 的输出,L224、L213),复用 MLA 低秩查询信息
  - `wk: 6144→128` —— 单共享 key 头(L190、L228),`k_norm: LayerNorm(128)`(L191)
  - `weights_proj: 6144→32` —— 可学习的逐头权重(L192、L243)
  - `softmax_scale = 128^-0.5`(L193)
- 计算流程(L222-255):
  1. q 切成 `q_rot(64)` / `q_pass(64)`(L226),k 同样切(L229);
  2. **interleaved RoPE** 应用于 q_rot/k_rot(L232,注释明确"GLM-MoE-DSA uses interleaved RoPE in the indexer");
  3. indexer key 写入共享 cache:`past_key_values.update_indexer(k, layer_idx)`(L237);
  4. 打分:`scores = q·kᵀ × scale` → **ReLU**(L239-240);
  5. 跨头加权聚合:每头分数按 `weights_proj(hidden_states) × 32^-0.5` 加权求和,得 `[B,S,T]`(L243-244);
  6. **因果掩码**:优先用外部 attention_mask,否则按 `position_ids` 构造因果掩码(L246-252);
  7. `topk = min(2048, T)`,返回 `top-k` 索引 int32 `[B,S,topk]`(L254-255)。
- 全程 `@torch.no_grad()`(L195):top-k 是离散选择,不参与梯度;`weights_proj` 被声明为 fp32 保持模块(L660)。

### 2.3 Indexer 如何参与注意力计算(L422-466)

- **full 层**:运行自己的 indexer 得到 `topk_indices`(L423-432);
- **shared 层**:直接取上层传入的 `prev_topk_indices`(L433-436,无 indexer 则报错);
- **mask 化(eager/SDPA 路径)**(L439-449):用 `topk_indices` 对 `[B,S,T]` 全 1 布尔张量做 `scatter(...,False)` 得到稀疏掩码,再 `masked_fill` 进 attention_mask 的 -inf(L449)——效果是 **softmax 只在前 top-k(+因果)token 上计算**;
- **kernel 路径(flash-mla 等)**:直接以 `indices=sparse_indices` 传给注意力内核(L451、L464),由内核原生消费 int32 top-k 索引;
- 注意力完成后返回 `topk_indices` 给上层(L470),用于下一层复用。

### 2.4 Indexer 的 full / shared 类型(L366-367,配置 JSON L26-105)

```python
self.skip_topk = config.indexer_types[layer_idx] == "shared"
self.indexer = None if self.skip_topk else GlmMoeDsaIndexer(config, layer_idx)
```

- `"full"`:该层持有完整 indexer 参数(自己计算 top-k);
- `"shared"`:该层 **不实例化任何 indexer 参数**(`self.indexer = None`),完全复用最近一个 full 层的 top-k 选择结果;
- 配置中的 `indexer_types` 共 78 项:第 0-2 层为 full(前 3 层 dense MLP 层),此后**每 4 层一个 full(3 个 shared 夹 1 个 full)**,full 层位于 3,7,11,…,71,共 21 个 full / 57 个 shared(约 73% 层共享);
- 该模式与 `index_topk_freq=4`(每 4 层一个 full)、`index_skip_topk_offset=3`(前 3 层跳过共享)参数自洽(配置 JSON L21-23);
- 传递机制:模型主循环维护 `topk_indices` 变量,逐层传入 `prev_topk_indices` 并接收本层输出(L732-743);shared 层把收到的 top-k 原样传给下一层,由于 shared 层连续成块,同一 full 层的选择被整块复用。

---

## 3. MoE 路由机制(L489-591)

### 3.1 路由器 `GlmMoeDsaTopkRouter`(L489-527)——noaux_tc + sigmoid

- 路由权重 `[256, 6144]` 单矩阵(L495),fp32 计算(L504);`e_score_correction_bias` 专家分数修正偏置(零初始化 buffer,L500,DSV3.2 特性);
- **sigmoid 打分**(L505):`scores = sigmoid(router_logits)`,`topk_method="noaux_tc"`,**无辅助负载均衡损失**(无 load-balancing loss 项);
- **分组 top-k**(L507-520):每组内取 top-2 求和得到组分数(L507-511)→ 组间 top-k(`topk_group=1`)选组(L512)→ 生成组掩码 mask 掉未选专家(L515-520);
  > 注:本配置 `n_group=1`(配置 L194),即 256 个专家为唯一一组,组选必然全选——分组机制在当前规模下是**退化的**(结构保留、效果等价于无分组),但代码逻辑对更大 n_group 是通用的;
- 权重:L521 取 top-k 索引(每 token 8 个专家),L522 用 **原始 sigmoid 分数**(而非加修正偏置的分数)取权重;`norm_topk_prob=true` 时按行归一化(L523-525);最后乘 `routed_scaling_factor=2.5`(L526)。

### 3.2 专家 `GlmMoeDsaExperts`(L530-567)

- 3D 参数:融合 `gate_up_proj [256, 2×2048, 6144]`(L539)+ `down_proj [256, 6144, 2048]`(L540),即每个专家一个 SwiGLU FFN(intermediate=2048);
- forward(L543-567):one-hot 专家掩码(L551-552)→ 命中专家列表(L553)→ 逐专家取 token 计算 `silu(gate)·up → down`,乘路由权重后 `index_add_` 累加(L555-565),稀疏专家激活(每 token 仅 8/256)。

### 3.3 MoE 组合与共享专家(L570-591)

- `GlmMoeDsaMoE` = 路由专家 + `shared_experts`(稠密 MLP,intermediate = 2048×1,L580-582);
- forward:路由输出 + 共享专家输出直接相加(L590)——共享专家每 token 全量计算,提供稳定稠密通道;
- 层内选择(`GlmMoeDsaDecoderLayer` L600-603):`mlp_layer_types[layer_idx]=="sparse"` → MoE,否则稠密 MLP(intermediate=12288);配置前 3 层 dense、第 4-77 层全 sparse(配置 L110-189),对应 `first_k_dense_replace=3`。

---

## 4. "DSA" 的缩写含义与 indexer 共享设计

### 4.1 DSA = DeepSeek Sparse Attention(稀疏注意力)

代码自带定义(L166):

> `DeepSeek Sparse Attention (DSA) indexer for selecting top-k tokens.`

并明确本实现与 DeepSeek-V3.2 的对齐关系(L208-209):

> `Same as DeepseekV32Indexer.forward, but the indexer applies interleaved RoPE rather than the non-interleaved half-split RoPE used by DeepSeek-V3.2.`

即:**DSA 不是"DeepSeek-Architecture",而是 DeepSeek-V3.2 提出的 DeepSeek Sparse Attention 稀疏注意力机制**——用一个轻量 indexer 网络为每个查询选出 top-k 个可注意的 key,把注意力从全量 O(T²) 降为 O(T·k),k=2048 相对于 1M 上下文是 0.2% 量级。GLM-5.2 沿用了这一机制,但做了 GLM 自己的改动(见 4.2、6 节)。

### 4.2 文件内的 indexer 共享:full/shared 跨层复用

- 核心思想:**相邻层的注意力选择高度相似,不必每层都跑 indexer**。`indexer_types` 决定每层类型(L366-367),shared 层零参数、零计算、零 indexer KV 缓存,直接复用最近 full 层的 top-k;
- 该机制是本文件相对 DeepSeek-V3.2 的主要差异,代码以 `# MAIN DIFF with DSV3.2` 四处标注(L616、L629、L732、L741),包括:
  - 解码层 forward 新增 `prev_topk_indices` 入参(L616);
  - 注意力调用透传该参数(L629);
  - 模型主循环维护并传递 `topk_indices`(L732、L741);
- 收益:约 73% 的层省去 indexer 参数(每层约 2048×4096 + 6144×128 + 6144×32 参数量)、省去 indexer key 缓存与打分计算,代价是注意力稀疏选择"略陈旧"(最多相差 3 层)。

### 4.3 `index_share_for_mtp_iteration=true` 的作用

- 该参数**在本文件中未被消费**(全工作区 grep 仅出现在配置 JSON L20);本文件内的共享由 `indexer_types` 驱动;
- 按其命名与 GLM-5.2 的 MTP 配置(`num_nextn_predict_layers=1`,配置 L202)推断,其语义为:**MTP(多 token 预测)模块在多次迭代预测时,复用主模型最后一个 full indexer 层算出的 top-k 索引,而不是每个 MTP 迭代都重新跑一遍 indexer**。这与 4.2 的跨层共享是同一思路的"跨迭代"版本:稀疏选择只算一次,后续迭代直接沿用,进一步削减稀疏注意力的重复开销;
- 由于 MTP 模块文件不在本工作区,以上为基于参数名与配置的推断,确切行为需查阅 MTP 实现(`num_nextn_predict_layers` 相关文件)确认。

---

## 5. 位置编码 / RoPE 处理

### 5.1 基频与 cos/sin 生成(L68-122)

- `rope_type="default"`,基频按 `rope_theta=8,000,000` 计算(L98-104):`inv_freq = base^(-2i/dim)`,`dim = config.head_dim = 192`(L99)——**注意**:这里用的是 `head_dim`(nope 维)而非 `qk_rope_head_dim`,见第 8 节一致性观察;
- forward 生成 `emb = cat(freqs, freqs)` 得到双倍长度 cos/sin(L118),乘 `attention_scaling`(L119-120),并强制 fp32 计算(L116);
- 支持 `dynamic_rope_update` 装饰器(L107),预留长上下文动态 RoPE 能力(`max_position_embeddings=1,048,576`)。

### 5.2 "interleave(交错)"的含义(L125-161)

- 配置 `rope_interleave=true`、`indexer_rope_interleave=true`(配置 L25、L210),注意力与 indexer 均走 `apply_rotary_pos_emb_interleave`;
- **交错布局**:旋转维度按 `(x0,x1),(x2,x3),…` 相邻成对排列,每对共用一个频率(函数 docstring L129-131:"DeepSeek lays the rotary dimensions out in interleaved pairs (x0, x1), (x2, x3), …");
- 实现技巧:取 cos/sin 的前半段(L153-154),把 q/k 切成奇偶切片 `q[...,0::2]` / `q[...,1::2]`(L156-157),直接做 `q1·cos − q2·sin` / `q2·cos + q1·sin`(L159-160)——**不需要 view/transpose/reshape 去交错**,与 de-interleave 的 `rotate_half` 版本逐位等价(L131-132),省一次 contiguity 拷贝;
- 对照:DeepSeek-V3.2 的 indexer 用的是非交错的 half-split RoPE(将 rope 维一分为二分别旋转),本文件在 L208-209 明确点出这是 GLM 版 indexer 与 DSV3.2 的差异之一;
- RoPE 只作用于 rope 分量(注意力 256 维中的 64 维,L403/L406/L412;indexer 128 维中的 64 维,L226/L232),nope 部分(192)不含位置信息。

---

## 6. 与标准 Transformer、DeepSeek MLA 的差异

### 6.1 相对标准 Transformer

| 方面 | 标准 Transformer | 本实现 |
|---|---|---|
| KV 缓存 | 每头完整 W_k/W_v,64×256 每 token | MLA 低秩 latent(512 维)压缩,K/V 展开后缓存(L346-356、L418-420) |
| 注意力范围 | 全量因果 | **top-k 稀疏 + 因果**(L439-449) |
| 位置编码 | 全头维 RoPE | 仅 64/256 维 rope 分量 + interleave(L403-412) |
| FFN | 稠密 | 前 3 层稠密(12288),其余 8/256 专家 MoE + 共享专家(L600-603) |
| 路由 | — | sigmoid 无辅助损失路由(L505) |
| 归一化 | LayerNorm | RMSNorm + fp32 累加(L48-62) |
| 新增组件 | — | DSA indexer(L164-255)、跨层 top-k 共享(L366-367、L732-743) |

### 6.2 相对 DeepSeek-V3 / V3.2 MLA

主体 MLA 结构与 DeepSeek 一致(q_a/q_b 低秩查询、kv_a/kv_b 低秩 KV、nope/rope 分离、expand_kv)。差异点:

1. **indexer 的 RoPE 布局**:本文件用 interleave(L231-232、L412),DSV3.2 用 half-split(L208-209);
2. **跨层 top-k 共享(full/shared)**:DSV3.2 每层独立 indexer;本实现 73% 层零 indexer 参数并复用上层 top-k(L366-367、L422-436),代码自标注为 `MAIN DIFF with DSV3.2`(L616/629/732/741);
3. **稀疏注意力的实现路径**:eager/SDPA 下用布尔 scatter 构造 -inf 加法掩码(L439-449),flash-mla 内核路径则直接消费 int32 索引(L451、L464);DSV3.2 主要依赖 flash-mla 内核(本文件 `_supports_flash_attn=False`,注释称 flash-mla 内核尚需适配,L648);
4. **路由器细节**:新增 `e_score_correction_bias` 专家分数修正(L500、L506),`norm_topk_prob` 归一化(L523-525),`routed_scaling_factor=2.5`(L526);
5. **indexer 梯度策略**:`@torch.no_grad()` 全流程(L195)+ `weights_proj` fp32 保持(L660);
6. **KV 头配置**:`num_key_value_heads = num_attention_heads = 64`,groups=1(L335),无 GQA 头数差异;
7. 额外保留 YaRN mscale 钩子(L295-308,当前 `rope_type=default` 下不生效)。

---

## 7. 最具特色的 3-5 个设计点

1. **跨层 top-k 共享(indexer full/shared 交替)**
   每 4 层仅 1 个 full indexer,57/78 层完全不实例化 indexer 参数(L366-367);靠 `prev_topk_indices` 自下而上传递(L616、L732-743)与 indexer key 的按层缓存(`past_key_values.update_indexer`,L237)实现;同一 top-k 选择在连续 3 个 shared 层被复用。这是文件相对 DeepSeek-V3.2 的**核心创新**,代码多处显式标注。

2. **DSA indexer 的轻量化设计**
   复用 MLA 的 `q_resid`(2048 维)经 `wq_b` 投影(L189、L224),配合**单一共享 key 头**(wk + LayerNorm,L190-191)与**可学习逐头权重** `weights_proj`(L192、L243)聚合 32 头分数,ReLU 后 top-k(L240、L254);全程无梯度(L195)、`weights_proj` 保 fp32(L660)——用最小的额外参数实现稀疏选择。

3. **双路径稀疏注意力(可降级的 kernel 就绪设计)**
   eager/SDPA 路径把 top-k 索引 scatter 成 -inf 加法掩码(L439-449),flash-mla 路径直接传 `indices=` 给内核(L451、L464);同一份 top-k 语义同时服务"参考实现"与"高性能内核",稀疏注意力可随后端能力平滑切换。

4. **无辅助损失的 sigmoid 路由(noaux_tc)+ 共享专家**
   `scores = sigmoid(logits)`(L505)天然无需 aux loss 平衡;分组 top-k(L507-520)、分数修正偏置(L500)、权重归一化 × 2.5 缩放(L523-526)、融合 `gate_up_proj` 3D 专家权重(L539)与每 token 必算的共享专家(L580-582、L590)构成完整的 DeepSeek 风格路由体系,且 `n_group=1` 下分组逻辑自动退化为全量选择,规模可扩展。

5. **Interleaved RoPE 的奇偶切片直接旋转实现**
   用 `q[...,0::2]`/`q[...,1::2]` 奇偶切片 + 前半段 cos/sin 直接完成交错旋转(L153-160),与 de-interleave `rotate_half` 逐位等价但省去 view/transpose/reshape 的拷贝(L129-132);同一函数同时服务注意力(L412)与 indexer(L232),`rope_interleave`/`indexer_rope_interleave` 双配置均为 true。

> 附加特色(次要):缓存设计上稀疏模型缓存**展开后的 K/V**而非压缩 latent(L418-420 注释);indexer key 独立缓存于 `DynamicIndexedLayer`(L172-174);主循环因果 mask 按 `config.layer_types` 映射到 `deepseek_sparse_attention` key(L727、L736);`_keys_to_ignore_on_load_unexpected` 忽略 `model.layers.78.*`(L659,78 层编号 0-77,疑似预留/MTP artifact)。

---

## 8. 附注与一致性观察

1. **RoPE 维度疑点(重要)**:`GlmMoeDsaRotaryEmbedding` 以 `dim = config.head_dim = 192` 计算基频(L99),得到 96 个频率 → cos/sin 长 192(L118),interleave 函数取前半段得 96(L153-154);而注意力/ indexer 的 rope 分量都是 `qk_rope_head_dim = 64`(L403、L226),奇偶切片后每片仅 32 维——96 与 32 无法广播相乘。按本工作区配置数值,该文件在 rope 路径上**与配置不自洽**;若 `dim` 取 `qk_rope_head_dim=64`(即 32 频率 → cos 64 → 切片 32),则完全自洽。推断该生成文件 L99 应取 `qk_rope_head_dim`(参考 DeepSeek 系列 HF 实现),或真实 checkpoint 的 rope 相关配置与此处 JSON 不同;报告结构分析时按代码语义描述,运行正确性需以实际权重/配置为准。

2. **`index_share_for_mtp_iteration`、`index_topk_freq`、`index_skip_topk_offset`、`index_topk_pattern`、`layer_types` 均未被本 modeling 文件直接消费**(仅 `indexer_types` 在 L366 被读取,`mlp_layer_types` 在 L600 被读取;`config.layer_types` 在 L736 被用作 mask 映射 key,应在 configuration 类中由 indexer/mlp 类型推导)。这些参数的作用域在 configuration 类与 MTP 模块(不在本工作区)。

3. 本文件为 HF modular 自动生成产物(L1-6),属 GLM-5.2 的参考实现;`_supports_flash_attn=False`(L648)说明高性能 flash-mla 路径仍在适配中,当前推理主要走 SDPA/eager + 稀疏掩码。
