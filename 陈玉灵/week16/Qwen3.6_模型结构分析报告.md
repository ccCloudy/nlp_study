# Qwen3.5-MoE / Qwen3.6 模型结构分析报告

> 分析对象：`modeling_qwen3_5_moe.py`（2233 行，HuggingFace transformers modular 自动生成）
> 配套配置：`Qwen_Qwen3.6-27B_config.json`（dense 版）、`Qwen_Qwen3.6-35B-A3B_config.json`（MoE 版）
> 文中行号均指本建模文件；配置行号标注为 cfg27B / cfg35B。

---

## 0. 一句话概括

这是一个**多模态、混合注意力、MoE 稀疏化的解码器模型**：主干用"每 4 层 1 个 full attention + 3 个 linear attention"的混合结构，其中 linear attention 是 **Mamba 风格深度可分离因果卷积 + Gated Delta Rule（delta 规则）递归状态**（FLA 库 Gated DeltaNet 家族），full attention 采用 **GQA + QK-Norm + 注意力输出门 + partial RoPE + interleaved M-RoPE**；MoE 为 **256 专家 top-8 softmax 路由 + 门控共享专家 + Switch 式 load-balancing aux loss**；外部套一层 **3D 时空 patch 视觉塔**，以 `Qwen3_5MoeForConditionalGeneration` 为总入口。

## 1. 核心类/函数职责总览

| 类/函数 | 行号 | 职责 |
|---|---|---|
| `Qwen3_5MoeVisionRotaryEmbedding` | L73-82 | 视觉塔简单 2D RoPE（无 mrope） |
| `Qwen3_5MoeTextRotaryEmbedding` | L85-165 | 文本 RoPE：partial rotary + **interleaved M-RoPE**（3 维位置、频段交错） |
| `Qwen3_5MoeRMSNormGated` | L169-185 | 线性注意力输出门控 Norm：`RMSNorm(x)·silu(gate)` |
| `causal_conv1d_update` / `causal_conv1d_fn` | L201-240 | 深度可分离因果卷积（单步更新 / 整段），hub kernel 回退 |
| `torch_chunk_gated_delta_rule` | L250-328 | chunked 分块 Gated Delta Rule kernel（预填充，chunk=64） |
| `torch_recurrent_gated_delta_rule` | L332-381 | 逐 token 递归 Gated Delta Rule kernel（解码，维护矩阵状态） |
| `Qwen3_5MoeGatedDeltaNet` | L385-541 | **线性注意力层本体**（conv + 双门控 + delta rule + 输出门） |
| `Qwen3_5MoeAttention` | L627-701 | full attention 层（GQA + QK-Norm + 输出门 + partial RoPE） |
| `Qwen3_5MoeMLP` | L704-717 | 标准 SwiGLU MLP（兼作 shared expert） |
| `Qwen3_5MoeExperts` | L720-757 | 专家权重集合（3D 参数张量 + one-hot 分发 + index_add） |
| `Qwen3_5MoeTopKRouter` | L760-776 | 线性路由 → softmax → top-8 → 重归一化 |
| `Qwen3_5MoeSparseMoeBlock` | L779-798 | 稀疏 MoE 块（路由专家 + 门控共享专家） |
| `Qwen3_5MoeRMSNorm` | L801-818 | 零初始化 weight、`(1+weight)` 的 RMSNorm |
| `Qwen3_5MoeDecoderLayer` | L821-877 | 解码层：按 layer_types 分派 full/linear 注意力 + MoE MLP |
| `Qwen3_5MoeVisionModel` 及子模块 | L916-1204 | 视觉塔：3D patch embed + 27 层 ViT + 2×2 merger |
| `Qwen3_5MoeTextModel` | L1234-1319 | 文本主干：embed + 分层 + 按层类型分发 mask |
| `Qwen3_5MoeModel` | L1323-1690 | **多模态外壳**：visual + language_model + 3D position ids |
| `load_balancing_loss_func` | L1693-1772 | Switch-Transformer 式路由均衡 aux loss |
| `Qwen3_5MoeForCausalLM` | L1776-1878 | 纯文本入口（含 aux loss 累加） |
| `Qwen3_5MoeForConditionalGeneration` | L1881-2223 | 多模态生成入口（含生成期 4 行 position ids 处理） |

`Qwen3_5MoePreTrainedModel`（L880-913）统一初始化策略：delta-net 的 `dt_bias=1`、`A_log~U(0,16).log()`（L900-902）、RMSNorm 权重置 0（L904-905）、专家权重正态初始化（L906-908），并声明 `_is_stateful = True`（L894，带显式状态缓存）。

## 2. 混合注意力结构（核心特色）

### 2.1 分布模式：`layer_types` + `full_attention_interval=4`

- 两种注意力由配置 `layer_types` 数组逐层指定：27B 的 64 层（cfg27B L21-85）与 35B 的 40 层（cfg35B L19-59）均为同一周期——**连续 3 个 `linear_attention` + 1 个 `full_attention`**，即每 4 层只在第 4 层做全注意力（`full_attention_interval: 4`）。64 层 → 16 个 full 层，40 层 → 10 个 full 层。
- `Qwen3_5MoeDecoderLayer.__init__` 按 `config.layer_types[layer_idx]` 挂载 `self.linear_attn`（GatedDeltaNet）或 `self.self_attn`（Qwen3_5MoeAttention）（L825-829），前向分派（L848-864）。
- **mask 按层类型分别生成**：`Qwen3_5MoeTextModel.forward` 构造 `causal_mask_mapping = {"full_attention": create_causal_mask(...), "linear_attention": create_recurrent_attention_mask(...)}`（L1285-1298），再按 `layer_types[i]` 取对应 mask 传层（L1307）。线性注意力用"递归注意力 mask"（对 padding 掩码，不需因果三角 mask）。

### 2.2 Linear attention：Mamba 风格卷积 + Gated Delta Rule（L385-541）

四阶段：

1. **深度可分离因果卷积**（local 特征）
   - `in_proj_qkv`：hidden → `key_dim*2 + value_dim`（L424）；`conv_dim = key_dim*2 + value_dim`（L402）。
   - `nn.Conv1d(conv_dim, conv_dim, kernel_size=4, groups=conv_dim, padding=3)` —— **分组=通道数的 depthwise 卷积**，kernel=4 即 `linear_conv_kernel_dim=4`（L403-410）。
   - 前向：整段 `causal_conv1d_fn`（L468-474）或解码单 token `causal_conv1d_update` 原地更新（L452-461）；卷积状态存 `cache_params.conv_states`（L464-466）；padding token 隐藏态先置零（L437，Mamba padding 消融技巧）。

2. **双门控时间步（Mamba 式离散化）**
   - `in_proj_b` → `beta = sigmoid(b)`（L426, L495）：**delta 规则写入门**——控制"新 KV 信息写入记忆"的强度。
   - `in_proj_a` + 可学习 `dt_bias`（L414）+ `A_log`（L416-417）：`g = -exp(A_log)·softplus(a + dt_bias)`（L497）——每步指数衰减因子（Mamba 的 Δ·A 离散化，负数保证衰减）。`dt_bias` 初始 1、`A_log` 初始 `U(0,16)` 对数（L900-902）。
   - `mamba_ssm_dtype: float32`：两内核内 q/k/v/beta/g 全部转 float32 计算（L267, L348），避免 fp16 下 A 变 -inf。

3. **Gated Delta Rule 核心（递归矩阵状态 = SSM 状态）**
   - 预填充走 `torch_chunk_gated_delta_rule`（chunk=64，L250-328）；解码单 token 走 `torch_recurrent_gated_delta_rule`（L332-381）。
   - 均 `use_qk_l2norm_in_kernel=True`：**Q、K 先 L2 归一化**（L243-246, L263-265, L344-346），再按 `1/sqrt(d)` 缩放（L279, L353）。
   - **状态是矩阵** `(batch, num_v_heads, k_head_dim, v_head_dim)`（L302-306, L359-363）：`state = state·g + k ⊗ (β·(v − state·k))`（L372-375）——Delta Rule 的"写-读"记忆更新，等价于**线性注意力/SSM 风格的 KV 记忆矩阵**；与 conv 状态两层状态共存，故模型声明 `_is_stateful=True`。状态由 `DynamicCache` 的 `conv_states`/`recurrent_states` 缓存更新（L452-454, L502, L531-532）。

4. **输出门（swish gate）**
   - `in_proj_z`：hidden → value_dim（L425, L446-447）。
   - `Qwen3_5MoeRMSNormGated`（L169-185）：`out = RMSNorm(core_attn_out)·silu(z)`（L183, L537），最后 `out_proj` 回 hidden（L540）。对应 `output_gate_type: swish`（代码硬编码 silu，L174）。

### 2.3 Head 拆分（key/value 头配置）

- `linear_num_key_heads=16, linear_key_head_dim=128`；`linear_num_value_heads`：27B=48（cfg27B L89-91）、35B=32（cfg35B L64）。
- `key_dim = 16×128 = 2048`，`value_dim = 48×128 = 6144`（27B）；conv 输入 = 2048×2 + 6144 = **10240**（27B）/ 8192（35B）。
- Q/K 头按 `num_v_heads // num_k_heads`（27B=3、35B=2）`repeat_interleave` 对齐 V 头数（L498-500）——**线性注意力"KV 共享"方向与 GQA 相反**：多个 Q/K 头共享一个 V 头。

## 3. Full Attention 实现细节（L627-701）

- **GQA**：27B heads=24、kv_heads=4（分组 6）；35B heads=16、kv_heads=2（分组 8）。`repeat_kv` 见 L590-599。
- **QK-Norm**：`q_norm`/`k_norm` 是 head_dim 上的 RMSNorm（L651-654, L672-673，注释明确"unlike olmo, only on the head dim"）。
- **注意力输出门**：`q_proj` 输出 `heads×head_dim×2`（L639-641），`chunk(2)` 拆成 `query_states` 与 `gate`（L667-670）；注意力输出在 `o_proj` 前乘 `sigmoid(gate)`（L698）——**逐头软门控**。
- **partial RoPE**：`apply_rotary_pos_emb`（L552-587）只旋转前 `rotary_dim` 维（L576-586）。`partial_rotary_factor=0.25`：27B head_dim=256 → **rotary dim=64（仅 25%）**，其余 192 维不旋转。
- **interleaved M-RoPE（文本侧）**：
  - `Qwen3_5MoeTextRotaryEmbedding.forward` 把 2D position_ids 扩成 3 行（T/H/W）（L133-138），float32 强制计算（L140-142）。
  - `apply_interleaved_mrope`（L150-165）把 chunked 布局 `[TTT...HHH...WWW]` 重排为 **interleaved `[THWTHW...]`**：H 取索引 1,4,7,...,31（11 个）、W 取 2,5,...,29（10 个）、T 保留其余 11 个 —— 正好 `mrope_section=[11,11,10]`（32 频段）。27B rotary dim=64=32×2，`cat((freqs,freqs))` 后 cos/sin 64 维，与 partial rotary 咬合。
- 位置编码**只作用于 full attention**（L676-677）；linear attention 层完全不用 RoPE。
- 注意力后端经 `ALL_ATTENTION_FUNCTIONS` 支持 eager/SDPA/Flash（L682-695）。

## 4. MoE 路由机制（35B-A3B，L720-798 + L1693-1772）

- 配置：`num_experts=256`、top-8、`moe_intermediate_size=512`、`shared_expert_intermediate_size=512`、`router_aux_loss_coef=0.001`。
- 路由：`Qwen3_5MoeTopKRouter` 单线性层（256, hidden）算 logits（L766, L770）→ **softmax → top-8 → 重归一化**（L771-774）。
- 专家计算：`Qwen3_5MoeExperts` 权重存 3D 参数张量 —— `gate_up_proj (256, 2×512, 2048)`（gate 与 up 融合，L729, L751）与 `down_proj (256, 2048, 512)`（L730）；one-hot 掩码找命中专家（L740-743），`silu(gate)·up` → down → 乘路由权重 → `index_add_` 归位（L745-755）。`@use_experts_implementation` 允许外部替换（L720）。
- **共享专家 + 标量门控**：`shared_expert = Qwen3_5MoeMLP(512)`（L784），输出乘 `sigmoid(shared_expert_gate(x))`——`shared_expert_gate = Linear(hidden, 1)` 每 token 一个可学习标量门（L785, L794）；最终 `路由专家输出 + 门控共享专家输出`（L796）。
- **Aux loss**：`load_balancing_loss_func`（L1693-1772）实现 Switch 式 `num_experts × Σ(f_i·P_i)`（L1771-1772），支持 padding 掩码统计（L1741-1769）；在 `Qwen3_5MoeForCausalLM.forward` 按 `router_aux_loss_coef=0.001` 加权累加（L1859-1868）。

## 5. MTP（Multi-Token Prediction）

- 配置存在：`mtp_num_hidden_layers: 1`、`mtp_use_dedicated_embeddings: false`（cfg27B L95-96 / cfg35B L70-71）。
- **本建模文件无 MTP 模块实现**——仅 `_keys_to_ignore_on_load_unexpected = [r"^mtp.*", ...]`（L888, L1782）显式忽略 `mtp.*` 权重键。MTP 由外部推理引擎承担（如 vLLM `--speculative-config '{"method":"qwen3_next_mtp",...}'`）。

## 6. 视觉编码器结构与融合（L916-1204, L1323-1690）

- 配置：patch=16、temporal_patch_size=2、depth=27、hidden=1152、spatial_merge_size=2、out_hidden_size=5120（27B）/2048（35B）。
- **Patch embed**：`Qwen3_5MoeVisionPatchEmbed` 用 **Conv3d**（kernel/stride=[2,16,16]）一次性时空 patch（L929-946）——2 帧 × 16×16 空间块 → 1 token，视频按帧成组。
- 位置：学习式 `pos_embed`（2304 位置，L1111）+ 按 `grid_thw` 双线性插值重采样（L1168-1175, L1180）；注意力用 2D RoPE（无 mrope，L73-82, L1118）。
- **27 层 ViT block**：LayerNorm + 双向注意力 + SwiGLU MLP（L1062-1089）；视觉注意力 `num_key_value_groups=1`（无 GQA）、`is_causal=False`（L991），`cu_seqlens` 变长打包 Flash Attention（L1016-1033）。
- **Patch Merger**：`Qwen3_5MoeVisionPatchMerger` 把 2×2（spatial_merge_size²=4）个视觉 token 拼成 1152×4=4608 维，LayerNorm→MLP 投影到 `out_hidden_size`（L949-962, L1199）。
- **多模态外壳** `Qwen3_5MoeModel`（L1323-1690）：
  - `self.visual` + `self.language_model`（L1331-1332），`get_image_features` 按每图 token 数 split（L1520-1523）。
  - 占位符替换：`masked_scatter` 把 `image_token_id`/`video_token_id` 嵌入替换为视觉特征（L1645-1665），并校验 token/特征数一致（L1551-1566）。
  - **3D position ids（get_rope_index, L1390-1481）**：按 `mm_token_type_ids` 切成 text/image/video 分段，文本段递增 id，图像/视频段用 T/H/W 三维网格 id（L1375-1388）；**视频按时间戳拆分为多段**（L1425-1427 repeat_interleave 按帧展开、L1449-1474 groupby 分段）。
  - 生成期 `_prepare_position_ids_for_generation` 产出 **`[4, bs, seq]` position_ids**（第 0 行文本 + 3 行 mrope，L2040-2076）；TextModel 内部同样 4 行处理（L1271-1281）。

## 7. 与标准 Transformer / DeepSeek MLA / Qwen3-Next 的差异

**vs 标准 Transformer（Llama 系）**
1. 3/4 层被替换为无 attention mask、带递归矩阵状态的线性注意力 → 长序列显存/延迟大幅下降；
2. QK-Norm 与注意力输出 sigmoid 门（q_proj 双倍输出）；
3. RoPE 只旋转 25% 维度且为 interleaved 三维 M-RoPE；
4. FFN 换 MoE + 门控共享专家 + aux loss。

**vs DeepSeek MLA**
- MLA 用低秩 KV 压缩 + 解耦 RoPE；本模型 full attention 走显式 GQA + QK-Norm + 输出门，无低秩压缩；
- 线性注意力层用 delta-rule 矩阵状态替代 MLA 的 cache —— 两种"压缩 KV"哲学的不同实现；
- 共同点：共享专家 + aux loss（DeepSeek-V3 风格），但本模型共享专家由独立标量 sigmoid 门调制，路由权重是 softmax top-k 重归一化而非 sigmoid。

**vs Qwen3-Next（同宗同源）**
- 本文件 docstring 直接引用 `Qwen/Qwen3-Next-80B-A3B-Instruct`（L1822），Qwen3.5-MoE 是 Qwen3-Next 直接后继；共享 Gated DeltaNet 线性注意力与混合层分布；
- 增量特色：**interleaved（交错）M-RoPE**（非 chunked）、**attn_output_gate**、**QK-Norm**、`use_qk_l2norm_in_kernel=True` 的 QK L2 归一化、完整 3D 视觉塔 + 按时间戳切分视频的 mrope；MoE 256 专家 top-8 规模大于 Qwen3-Next-80B-A3B。

## 8. Top 5 特色设计点

1. **按 4 层周期的混合注意力 + 按层类型分发的双 mask 体系**（L825-864, L1285-1307）：full 层因果 mask、linear 层递归 mask，一套模型两套 token mixing。
2. **Gated DeltaNet 线性注意力 = Mamba 离散化 + delta-rule 矩阵状态 + depthwise conv(k=4)**（L385-541）：beta 控制写入、g 控制衰减、QK L2 归一化、chunked(64)/recurrent 双内核、float32 数值稳定。
3. **Full attention 门控三件套：QK-Norm + partial RoPE(25%) + 注意力输出 sigmoid 门**（L651-654, L667-670, L698），配合 interleaved M-RoPE（11,11,10 交错排布）支撑 262K 长上下文多模态位置编码。
4. **256 专家 top-8 + 融合 gate_up 3D 参数 + 标量门控共享专家**（L720-798）：3D 参数批量 GEMM 切片；共享专家标量 sigmoid 门控；Switch aux loss 加权 0.001。
5. **多模态一体化位置体系**：Conv3d 时空视觉塔 + 2×2 patch merger 直连 LLM hidden，`get_rope_index` 按时间戳拆分视频生成 3D position ids，生成期全流程携带 `[4, bs, seq]` 4 行 position ids。

## 9. 补充说明

- MTP：配置声明 1 层，本建模文件未实现（仅忽略 `mtp.*` 权重键），由外部推理引擎（vLLM `qwen3_next_mtp` 投机解码）使用。
- 配置算术提醒：35B 配置未显式给出 `head_dim`（默认 `hidden//heads=128`），`partial_rotary_factor=0.25` 下 rotary dim 仅 32，与 interleaved mrope 的 32 频段布局（索引上限 31）不匹配；实际发布权重应以 head_dim=256（rotary dim=64）为准。
