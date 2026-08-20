# DeepSeek-V3 / DeepSeek-V4 模型结构分析报告

> 分析对象：
> - `dsv3_model.py`（421 行，DeepSeek-V3 推理风格实现）
> - `dsv4_model.py`（827 行，DeepSeek-V4 推理风格实现）
> - 配套配置：`deepseek-ai_DeepSeek-V3_config.json`、`deepseek-ai_DeepSeek-V4-Flash_config.json`、`deepseek-ai_DeepSeek-V4-Pro_config.json`
>
> 行号均指各文件本身。代码依赖 `kernel` 模块（`fp8_gemm`/`fp4_gemm`/`sparse_attn`/`hc_split_sinkhorn` 等，不在本目录），本报告按接口语义推断其行为。

---

## 0. 一句话概括

- **V3**：DeepSeek-V3 的 MLA（低秩 KV 潜变量注意力）+ 无辅助损失 MoE 的**标准实现范本**，是后三个模型（GLM/Kimi MLA 部分）直接照抄的底座。
- **V4**：在 V3 基础上做**长上下文稀疏化**（压缩稀疏注意力 CSA + 重度压缩注意力 HCA）、**残差改造**（Hyper-Connections）、**路由工程化**（哈希路由 + FP4 专家）与 **MTP 多 token 预测**，支撑 1M 上下文。

---

## 1. DeepSeek-V3（`dsv3_model.py`）

### 1.1 核心类/职责

| 类/函数 | 行号 | 职责 |
|---|---|---|
| `ModelArgs` | L19-51 | 超参容器（q/kv lora rank、MLA 头维度、MoE 分组、YaRN 参数） |
| `ParallelEmbedding` | L54-74 | 词表按 rank 切分 + all_reduce |
| `Linear` / `ColumnParallelLinear` / `RowParallelLinear` | L91-137 | 支持 BF16/FP8 权重的线性层（`weight.element_size()==1` 时走 `fp8_gemm`，L77-88） |
| `RMSNorm` | L140-149 | 标准 RMSNorm |
| `precompute_freqs_cis` / `apply_rotary_emb` | L152-192 | YaRN 缩放 RoPE（factor=40） |
| `MLA` | L195-269 | **核心**：Multi-head Latent Attention |
| `MLP` | L272-280 | SwiGLU 稠密层（前 3 层用） |
| `Gate` | L283-318 | noaux_tc MoE 路由 |
| `Expert` | L321-329 | SwiGLU 专家 |
| `MoE` | L332-362 | 稀疏 MoE 块（+1 共享专家） |
| `Block` | L365-376 | 标准 pre-norm 残差块（attn + ffn） |
| `Transformer` | L379-411 | 组装：embed → N 层 → norm → head |

### 1.2 MLA 注意力（L195-269）——V3 的灵魂

**结构**：
- Q 低秩：`wq_a(dim→1536)` → `q_norm` → `wq_b(1536→heads×192)`（L211-213）；192 = nope 128 + rope 64。
- KV 低秩压缩：`wkv_a(dim→512+64)` 一次投影出 **latent 512 + 共享 rope 64**（L214，L240-241 拆分）。
- latent 经 `kv_norm` 后由 `wkv_b(512→heads×256)` 展开为 k_nope 128 + v 128（L216，L245-247）。

**两种推理模式（`attn_impl`）**：
- `naive`：展开完整 K/V 存 cache（L249-251），直观但 cache 大。
- `absorb`（默认，**权重吸收**，L252-267）：把 `wkv_b` 权重按头重组为 `[n_heads, head_dim, kv_lora_rank]`：
  - q 的 nope 分支**先乘** `wkv_b[:, :qk_nope_head_dim]`，即把"展开 K"吸收进 Q 侧（L255）；
  - cache 只存 **512 维 latent**（`kv_cache`）+ **64 维共享 rope**（`pe_cache`）（L227-228）；
  - 分数 = `einsum(q_nope, kv_cache) + einsum(q_pe, pe_cache)`（L258-259）；
  - v 的展开被吸收进输出侧：注意力输出 `einsum(scores, kv_cache)` 后乘 `wkv_b[:, -v_head_dim:]`（L266-267）。
  - **效果：KV cache 从 `heads×(128+128)` 降到 `512+64` 维/位置**，这是 MLA 省显存的核心。

**YaRN 相关**：`softmax_scale = qk_head_dim^-0.5`，超长序列时再乘 `mscale = 0.1·mscale·ln(factor)+1` 两次（L219-221），与 YaRN 的 attention 缩放配套。

### 1.3 MoE 路由（Gate L283-318，noaux_tc）

- 打分：`scores = linear(x, gate_weight)`；`sigmoid`（V3 配置）或 `softmax`（L297-300）。
- **671B 专属 bias**：`self.bias` 仅当 `dim==7168` 时存在（L293），**bias 只影响 topk 选择、不影响路由权重**（L302-303，L314 取原始分数）。
- **分组 topk**：`n_groups=8`、`topk_groups=4` —— 组内 top-2 求和（有 bias 时）或 amax（无 bias）得到组分数 → 选 top-4 组 → 组外专家 mask 置零（L304-312）→ 最终 `topk(8)`。
- 权重：取原始 sigmoid 分数并**归一化**（sigmoid 时 L315-316）再乘 `route_scale=2.5`（L317）。
- `MoE.forward`（L347-362）：逐专家按命中 token 计算，`y += expert(x)·weights`，最后加共享专家（L345，`n_shared_experts×moe_inter_dim` 的稠密 MLP）。

### 1.4 其他

- 前 `n_dense_layers`（配置 `first_k_dense_replace=3`）层用稠密 MLP，其余 MoE（L369）。
- YaRN rope：`factor=40`、`original_max_position_embeddings=4096`、`beta_fast=32/beta_slow=1`（L176-179）。
- FP8：`Linear.dtype = float8_e4m3fn`（L384），权重 128×128 分块缩放（config `weight_block_size=[128,128]`）。
- **注意**：配置 `num_nextn_predict_layers=1`（MTP），但 `dsv3_model.py` **未实现** MTP 层。

---

## 2. DeepSeek-V4（`dsv4_model.py`）

### 2.1 核心类/职责

| 类/函数 | 行号 | 职责 |
|---|---|---|
| `ModelArgs` | L34-80 | 含 compress_ratios、indexer、hc、fp4 参数 |
| `Linear` | L123-152 | **三态权重**：BF16 / FP8(e4m3fn, 128×128 块缩放) / **FP4(e2m1fn_x2, 32 元素块缩放 ue8m0)** |
| `rotate_activation` | L247-251 | Hadamard 旋转（量化前打散维度） |
| `get_window_topk_idxs` / `get_compress_topk_idxs` | L254-276 | 滑窗 / 压缩块的 topk 索引矩阵（lru_cache） |
| `Compressor` | L279-377 | **压缩 KV**：门控软池化 |
| `Indexer` | L380-433 | **稀疏索引**：top-k 压缩块选择 |
| `Attention` | L436-543 | MLA 变体（MQA 化）+ 滑窗 + 压缩 |
| `Gate` | L546-584 | 哈希路由 / 分数路由 |
| `Expert` / `MoE` | L587-644 | FP4 专家 + SwiGLU 截断 |
| `Block` | L647-700 | **Hyper-Connections** 残差 |
| `ParallelHead` | L703-735 | HC 头 |
| `MTPBlock` | L738-766 | 多 token 预测层 |
| `Transformer` | L769-809 | 组装 + HC 扩展/合并 |

### 2.2 注意力：滑窗 + 压缩 + 稀疏选择（L436-543）

**基础形态（MLA→MQA 化）**：
- **单 KV 头**（`num_key_value_heads=1`）、`head_dim=512`、末 64 维为 rope（L448-449，L504）。
- Q：`wq_a(dim→q_lora_rank)` → `q_norm` → `wq_b(→heads×512)`，再做 RMS 归一化（L496-498）。
- KV：`wkv(dim→512)` + `kv_norm`（L502-503），rope 只作用于末 64 维，nope 部分 FP8 模拟量化对齐 QAT（L506）。
- 输出：**分组低秩 O 投影** —— `wo_a` 把每头输出压到 `o_groups×o_lora_rank`（8/16×1024），`wo_b` 投影回 hidden（L462-463，L537-542）。

**两级稀疏（核心）**：
1. **滑窗**：`sliding_window=128`，`get_window_topk_idxs` 生成最近 128 个位置（L507）。
2. **压缩 KV（CSA）**：`compress_ratios[layer]` 按层取 0/4/128（L453）：
   - `Compressor`（L279-377）用**可学习门控软池化**：`wkv` 投影 KV、`wgate` 打分，`(kv·softmax(score)).sum` 把 ratio 个连续 token 压成 1 个（L342，L359）；
   - ratio=4 时用**重叠窗口**（`overlap_transform` L307-314，滑窗重叠平滑边界）；
   - 支持 prefill（start_pos==0 整段压缩，L325-342）与 decode（增量压缩，`kv_state`/`score_state` 状态缓冲，L343-359）；
   - 压缩位置用**独立的压缩 RoPE**（`compress_rope_theta=160000`，L367，L476-477）；
   - 压缩后 KV 再经 Hadamard 旋转 + FP4 模拟量化（`rotate=True` 时，L368-370）。
3. **稀疏选择（HCA）**：ratio=4 的层挂 `Indexer`（L466-471）：
   - 维护**自己的压缩 KV**（Hadamard + FP4，L398，L414-416）；
   - Q 复用注意力低秩投影的输出 `qr`（`wq_b(qr)` L411），分数 = `q·kv_cache` → ReLU → 乘可学习逐头权重 `weights_proj` → 按头求和（L420-421）→ **topk=512/1024** 选出压缩块（L427）；
   - 无 indexer 的压缩层（ratio=128）用确定性 `get_compress_topk_idxs` 等间隔取（L513）。
   - 最终 `topk_idxs = concat(滑窗索引, 压缩块索引)` 交给 `sparse_attn` 内核（L514，L528），并带 `attn_sink` 学习参数（L456）。
4. **输出逆旋转**：`apply_rotary_emb(o, freqs_cis, inverse=True)`（L534）。

**效果**（README）：1M 上下文时单 token 推理 FLOPs 仅 V3.2 的 27%、KV cache 仅 10%。

### 2.3 Hyper-Connections（Block L647-700，mHC）

- 状态从单条变成 **`hc_mult=4` 份拷贝**（`Transformer.forward` 里 `h.unsqueeze(2).repeat(1,1,4,1)`，L805）。
- `hc_pre`（L673-681）：把 `[b,s,hc,d]` 展平，乘可学习 `hc_fn`（尺寸 `(2+hc_mult)·hc_mult × hc·d`），RMS 归一化，**Sinkhorn 迭代（20 次）** 分裂出 pre/post/comb 三组权重（`hc_split_sinkhorn` 内核），pre 加权求和压回 1 份。
- 子层（attn/ffn）在这 1 份上计算，`hc_post`（L683-686）再用 post + comb 把结果与 4 份残差混合回 4 份状态。
- 头部 `ParallelHead.hc_head`（L728-735）与 MTPBlock 同样带 HC 混合。
- README 称其为 **Manifold-Constrained Hyper-Connections（mHC）**：约束信号在低维流形上传播，稳定深网 + 保持表达力。

### 2.4 MoE 路由（Gate L546-584）——两代演进

- **哈希路由（前 `n_hash_layers=3` 层）**：`tid2eid` 是 `[vocab, topk]` 的 int32 查表（L559），**直接按 token id 取专家索引**，完全免去门控计算（L577）；表不参与梯度。
- **分数路由（其余层）**：`sqrtsoftplus` —— `F.softplus(scores).sqrt()`（L571），配合可学习 bias 修正选择（L574-575）。
- 权重统一：取原始分数、非 softmax 时归一化、乘 `route_scale`（L580-583）。
- **FP4 专家**：`expert_dtype='fp4'` 时专家权重为 `float4_e2m1fn_x2`（32 元素块、ue8m0 scale，L623，L131-137）；`swiglu_limit=10` 对 SwiGLU 的 gate/up 幅值截断（L600-602）以适配 4bit 动态范围。
- MoE 每层 6 个激活专家 + 1 共享专家，计算在 float32 累积（L633）。

### 2.5 MTPBlock（L738-766）——多 token 预测

- 结构：`e_proj(embed 下一个 token 的嵌入)` + `h_proj(当前隐藏态)` 相加进入一个完整 Block（L762-764），末尾再出 logits（L765）。
- **共享主模型的 embed 与 head**（`Transformer.__init__` L790-793），训练时可同时预测第 t+1 token，推理时用于投机解码。
- 配置 `num_nextn_predict_layers=1`。

### 2.6 位置编码：双 RoPE 体系（L475-482）

- 压缩层：`compress_rope_theta=160000` + YaRN（factor=16，original=65536）。
- 纯滑窗层（ratio=0）：`rope_theta=10000`、**禁用 YaRN**（`original_seq_len=0`，L478-479）。
- 两种频率都预计算为 `freqs_cis` buffer（L480-482），压缩位置用压缩频率、主位置用主频率。

---

## 3. V3 → V4 演进对照

| 维度 | V3 | V4 |
|---|---|---|
| KV 头 | 128 头共享潜变量（MLA） | **1 头**（MQA 化，head_dim=512） |
| 注意力覆盖 | 全量上下文 | 滑窗 128 + 压缩块 + topk 稀疏选择 |
| KV 压缩 | 低秩潜变量（512/层） | 潜变量 + **跨 token 门控池化压缩**（4/128:1） |
| 残差 | 标准 pre-norm | **Hyper-Connections**（hc_mult=4 + Sinkhorn） |
| 路由 | sigmoid noaux_tc + 分组 topk | **sqrtsoftplus** + **前 3 层哈希路由** |
| 专家精度 | FP8 可选 | **FP4**（e2m1fn_x2 + swiglu_limit） |
| MTP | 配置声明未实现 | **完整实现**（共享 embed/head） |
| 上下文 | 163K（YaRN f=40） | **1M**（滑窗+压缩+稀疏） |

## 4. Top 5 特色设计点

1. **MLA 权重吸收**（V3 L252-267）：把 WKV_b 吸收进 Q 侧与输出侧，KV cache 压到 512+64 维/位置——后续 GLM/Kimi 的 MLA 均继承此设计。
2. **两级稀疏注意力 CSA+HCA**（V4）：滑窗 + 门控池化压缩 + Indexer 学习式 topk，1M 上下文 KV 仅 1/10、FLOPs 仅 27%。
3. **Hyper-Connections**（V4 L647-700）：4 份状态拷贝 + Sinkhorn 约束混合，替代普通残差。
4. **全链路低比特 + QAT 对齐**（V4）：FP8 激活/权重 + FP4 专家，`act_quant`/`fp4_act_quant` 逐处模拟量化（L506、L370、L416），与训练时 QAT 一致。
5. **MoE 路由两代演进**：noaux_tc（sigmoid+分组+归一化）→ sqrtsoftplus + **token-id 哈希路由免算**（V4 L556-579）。

## 5. 边界说明

- `kernel` 模块（`sparse_attn`/`hc_split_sinkhorn`/`fp4_gemm`/`act_quant` 等）不在本目录，接口行为按语义推断。
- V3/V4 代码内 `ModelArgs` 默认值均为小测试配置（7/27 层、64 专家等），真实规模由 config JSON 注入。
- V3 配置的 `num_nextn_predict_layers=1` 在 `dsv3_model.py` 中无对应实现（V4 已完整实现）。
- `num_hash_layers`（MoE 哈希路由）与注意力 `Indexer`（稀疏索引）是**两个独立机制**，勿混淆。
