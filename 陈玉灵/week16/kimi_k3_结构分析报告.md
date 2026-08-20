# Kimi-K3 模型结构实现分析报告

> 分析对象（Moonshot Kimi-K3 参考实现，代码注释明确说明是"Reference implementation for model architecture"）：
> - `kimi_k3_modeling.py`（1317 行）：多模态壳 `KimiK3ForConditionalGeneration` + 视觉塔 MoonViT-V2
> - `kimi_k3_linear_modeling.py`（1314 行）：文本主干 `KimiLinearForCausalLM`（KDA 线性注意力 + Gated MLA + LatentMoE + SiTU + AttnRes）
> - `kimi_k3_vision_processing.py`（179 行）：图像预处理 `KimiK3VisionProcessor`（NaViT 式动态分辨率）
>
> 对应配置关键参数（`moonshotai_Kimi-K3_config.json`）：hidden=7168、93 层、96 头、MLA(q_lora_rank=1536, kv_lora_rank=512, qk_nope=128, qk_rope=64, v_head=128, output_gate)、situ(beta=4.0/linear_beta=25.0)、24 层 full attention + 69 层 KDA、896 专家/16 topk/2 共享、LatentMoE 3584、patch=14/vt_hidden=1024/27 层、patchmergerv2、sd2_tpool、MXFP4 量化。README 汇总：2.8T 总参数 / 104B 激活参数，1M 上下文，基于 KDA + AttnRes 与 Stable LatentMoE。

---

## 0. 三个文件职责总览

| 文件 | 角色 | 核心类 |
|---|---|---|
| `kimi_k3_modeling.py` | 多模态顶层封装 + 视觉编码器 | `KimiK3ForConditionalGeneration`、`MoonViT3dPretrainedModel`（MoonViT-V2）、`MoonViT3dEncoder`、`MoonViTEncoderLayer`、`PatchMergerMLPV2` |
| `kimi_k3_linear_modeling.py` | 文本主干（全部创新点所在） | `KimiLinearForCausalLM`、`KimiLinearModel`、`KimiDecoderLayer`、`KimiMLAAttention`、`KimiDeltaAttention`（KDA）、`KimiSparseMoeBlock`/`KimiMoEGate`、`SituAndMul`、`KimiDynamicCache`、`_apply_attn_res`（AttnRes） |
| `kimi_k3_vision_processing.py` | 图像预处理（数据侧） | `KimiK3VisionProcessor` |

文本主干头部声明（`kimi_k3_linear_modeling.py` L4-6）：MLA、MoE gating 与 sparse MoE block 改编自 DeepSeek-V3，为 Kimi-Linear 架构做了大量扩展。

---

## 1. 各文件核心类/函数与职责

### 1.1 `kimi_k3_modeling.py`

**注意力/位置编码工具**
- `multihead_attention`（L59-99）：视觉塔的 flash-attention-2 varlen 封装（`flash_attn_varlen_func`，L83-93），支持变长打包，`causal=False`（视觉全可见）。
- `eager_attention`（L102-133）：eager 回退实现，按 `q_cu_seqlens` 逐段构造块对角 mask（L111-119）。
- `apply_rope`（L172-193）：复数域 RoPE 实现，`torch.view_as_complex` 旋转后 `view_as_real` 展平（L187-192）。
- `get_rope_shape`（L162-169）：被 `@torch.compile(dynamic=True)` 装饰的双三次插值，用于把可学习 2D 位置编码插值到任意分辨率（L165-169）。
- `get_1d_sincos_pos_embed(_from_grid)`（L196-230）：时间轴的固定 sincos 位置编码。

**视觉塔（MoonViT-V2）**
- `Learnable2DInterpPosEmbDivided_fixed`（L233-283）：`divided_fixed` 位置编码 = 可学习 2D 空间网格 `self.weight`（L247）+ 固定 sincos 时间编码 `time_weight`（L248-253）。forward（L260-283）按每个图像的 `grid_thws` (t,h,w) 把空间网格插值到 (h,w)（L265-272），再沿时间重复并叠加 `time_weight[0:t]`（L274-278），实现"空间可学习 + 时间固定、动态分辨率"。
- `MoonVision3dPatchEmbed`（L286-338）：`nn.Conv2d(in_dim=3, out_dim=1024, kernel=14, stride=14)`（L308-312），patch 后加位置编码（L335-337）。
- `Rope2DPosEmbRepeated`（L341-434）：2D 旋转位置编码，x/y 两轴用不同频率（L378-404），`get_freqs_cis`（L406-434）按 `grid_thws` 对每个图裁剪 (h,w) 子网格并沿时间 repeat。
- `MoonViTEncoderLayer`（L461-564）：标准 pre-norm Transformer 层；QKV 打包注意力 `attention_qkvpacked`（L507-543），`wqkv` 一次投影后 `unbind` 拆 Q/K/V（L519-528），RoPE（L530），flash/eager 分发（L532-540）。
- `MoonViT3dEncoder`（L567-618）：共享 2D RoPE（L579-580）、27 层 block（L581-586），由 `grid_thws` 计算 `cu_seqlens`/`max_seqlen` 做变长打包（L603-610）。
- `tpool_patch_merger`（L621-646）：`sd2_tpool` 合并策略——先把 t 帧取平均（temporal pooling，L636-640），再把 2×2 空间邻域重排为 `[new_h*new_w, 2*2, d]`（L641-642），即"2×2 空间下采样 + 时间池化"。
- `MoonViT3dPretrainedModel`（L649-717）：组装 patch_embed（L663-674）+ encoder（L676-692）；merge_type 仅支持 `sd2_tpool`（L709-713）。patch 14 → 1024 维、27 层、12 头（`vt_num_attention_heads: 12`）。

**投影器（mm_projector）**
- `IdentityMap`（L725-731）、`MLP`（L734-753）、`PatchMergerMLP`（L756-780，pre-norm 版本）。
- `PatchMergerMLPV2`（L783-815）：**本模型实际使用**（`mm_projector_type=patchmergerv2`）。输入维度 = `mm_hidden_size(1024) × merge_kernel(2×2) = 4096`（L788-789），`Linear(4096→4096, bias=False) → GELU → Linear(4096→7168, bias=False)`（L790-794），末尾 `post_norm = RMSNorm(7168)`（L795），trunc_normal 初始化（L796-801）。它把合并后每 token 的 4×1024 特征展平后一次性映射进文本隐藏维度。

**多模态壳**
- `KimiK3PreTrainedModel`（L818-850）：`_no_split_modules` 列出视觉塔/投影器/解码层（L821-827），便于 offload 与自动切分。
- `VisionTowerConfig`（L853-878）/ `ProjectorConfig`（L881-889）：从 `vision_config` 派生子配置。
- `KimiK3ForConditionalGeneration`（L893-1317）：见第 7 节。

### 1.2 `kimi_k3_linear_modeling.py`

- `SituAndMul`（L64-85）+ `ACT2FN["situ"]` 注册（L85）：SiTU 激活，见第 5 节。
- `KimiDynamicCache`（L120-223）：混合注意力专用动态缓存。按 `is_kda_layer` 生成每层类型（L131-139）；为 KDA 层保存 `conv_states`（q/k/v 短卷积状态）与 `recurrent_states`（L149-150），为 MLA 层保存标准 `key_cache/value_cache`（L151-152）。`has_previous_state`（L218-223）判断是否已有线性层状态（决定生成首步是否重算 prefill）。
- `KimiRMSNorm`（L226-236）：float32 计算的 RMSNorm。
- `KimiBlockSparseMLP`（L242-270）：专家 FFN，w1(gate)/w2(down)/w3(up) 三投影无 bias（L249-251）；situ 时把 gate 与 up 拼接后一次激活（L263-265）。
- `KimiMLP`（L273-301）：稠密 FFN（第 0 层使用），gate/up/down 结构 + situ（L294-301）。
- `KimiMLAAttention`（L335-474）：Gated MLA，见第 3 节。
- `KimiDeltaAttention`（L477-663）：KDA 线性注意力，见第 2.3 节。
- `KimiMoEGate`（L666-759）与 `KimiSparseMoeBlock`（L762-874）：MoE 路由与稀疏计算，见第 4 节。
- `KimiDecoderLayer`（L877-1046）：单层组装 + AttnRes 分支，见第 2.2/2.4 节。
- `_apply_attn_res`（L1075-1088）：AttnRes 核心算子，见第 2.4 节。
- `KimiLinearModel`（L1090-1233）：主干堆叠。embed_tokens（L1096-1097）、93 层（L1098-1099）、末层 RMSNorm（L1100-1101）；强制 flash_attention_2（L1110-1119）；per-layer mask 分发（L1194-1195）；AttnRes 的 block_residual 贯穿（L1188-1192、L1215-1217）。
- `KimiLinearForCausalLM`（L1236-1314）：LM 头 `lm_head`（L1247-1248，不 tie weights，`tie_word_embeddings=False`），`generation_mode` 时只取最后 token logits（L1299-1300）。

### 1.3 `kimi_k3_vision_processing.py`

- `KimiK3VisionProcessor`（L19-179）：
  - `media_tokens_calculator`（L44-51）：估算一张图产生的视觉 token 数（用于 prompt 中的 token 预算）。
  - `make_image_prompt`（L53-57）：生成占位 prompt `"<|media_begin|>image {W}x{H}<|media_content|><|media_pad|><|media_end|>"`。
  - `get_resize_config`（L59-71）：调用 `navit_resize_image`（NaViT 动态分辨率）计算 `new_width/new_height/pad_width/pad_height`，受 `patch_size(14)`、`merge_kernel_size(2,2)` 整除性约束与 `in_patch_limit`/`patch_limit_on_one_side`/`fixed_output_tokens` 上限约束。
  - `resize_image`（L73-88）：resize + 常数零 padding。
  - `preprocess`（L90-156）：逐图 normalize（mean/std，L127-131）→ `navit_patchify` 切 patch 并返回每图的 `(t, h, w)`（L132-135）→ 拼接 `pixel_values` 与 `grid_thws`（L138-146）→ `BatchFeature`。透明背景处理（L30-42，`transparent_bg_fill_stage="before_resize"`）。

---

## 2. 文本注意力混合结构：Full Attention 与 KDA 的分布与切换

### 2.1 层分布（配置驱动）

`text_config.linear_attn_config` 显式列出两类层（1-based 层号）：

- `full_attn_layers`：**24 层** = {4, 8, 12, …, 92, 93} —— 即"每 4 层插入 1 层全局注意力，最后两层(92, 93)也是全局注意力"；
- `kda_layers`：**69 层** = 其余所有层（1,2,3,5,6,7,…）。

与 README 表格"Attention-Layer Composition: **69 KDA + 24 Gated MLA**"（`kimi_k3_readme.md` L75）完全吻合，共 93 层。

> 注：配置列表中出现层号 93 且 0 号层不在任何列表中，说明列表为 1-based 编号（索引 i 对应层号 i+1）；`is_kda_layer`/`is_mla` 方法定义在 `configuration_kimi_k3.py`（不在本目录），解码层只依赖这两个方法做二选一。

### 2.2 切换机制

- **构建期**：`KimiDynamicCache.__init__`（L131-139）用 `config.is_kda_layer(i)` 生成 `layer_types`（`"linear_attention"` / `"full_attention"`）。
- **解码层**：`KimiDecoderLayer.__init__`（L883-892）：

```python
if config.is_kda_layer(layer_idx):
    self.is_linear_attn = True
    self.self_attn = KimiDeltaAttention(config=config, layer_idx=layer_idx)
elif config.is_mla:
    self.is_linear_attn = False
    self.self_attn = KimiMLAAttention(config=config, layer_idx=layer_idx)
else:
    raise NotImplementedError
```

- **前向**：`KimiDecoderLayer.forward` 按 `is_linear_attn` 选择传参形态——KDA 层传 `cache_params=past_key_values`（L952-959），MLA 层传标准 `past_key_values`（L941-950）。
- **Mask 分发**：`KimiLinearModel.forward`（L1194-1195）——KDA 层用 2D 0-1 `linear_attn_mask`（由 `_update_linear_attn_mask` 维护，L1124-1134，左 padding 场景下把首段标记置 None 以清空线性状态），MLA 层用标准 `causal_mask`（L1173-1180）。两类注意力因此共享同一套 dynamic cache（`KimiDynamicCache`）但各自维护状态。

### 2.3 KDA（Kimi Delta Attention）具体实现 —— `KimiDeltaAttention`（L477-663）

KDA 是 DeltaNet 风格**线性注意力（delta-rule 递推）**，用 fla 库的 `chunk_kda` / `fused_recurrent_kda` 内核（L48，import 于 L48-49）实现，核心要素：

1. **短卷积（short conv）**：q/k/v 投影后各自过 `ShortConvolution(kernel_size=4, activation='silu')`（L504-518）。因果短卷积在进入递推前先聚合局部 n-gram 上下文——这是线性注意力补偿"局部归纳偏置"的标配。
2. **学习型指数衰减 `A_log`**（L520-521）：每头一个标量，`log(Uniform(1,16))` 初始化，即衰减率 `exp(-A) ∈ [e⁻¹⁶, e⁻¹]`——控制旧状态遗忘速度（线性注意力的"decay"）。
3. **数据相关写入率 `beta`**（L529-530，L603）：`b_proj` 输出每头标量，核内 `use_beta_sigmoid_in_kernel=True`（L622、L641）经 sigmoid 变为 (0,1) 的"学习率"——delta rule 中决定新信息 (v) 覆盖旧状态的比例。
4. **`dt_bias`**（L526-527）：类似 Mamba 的时间步偏置。
5. **核内门控 `g`**：`f_a_proj(7168→128) → f_b_proj(128→12288)`（L523-524、L601-602），chunk 模式经 `use_gate_in_kernel=True`（L621）送入内核参与扫描。
6. **核内 q/k L2 归一化**：`use_qk_l2norm_in_kernel=True`（L620、L639）——等价于用 cosine 相似度打分，提升长程数值稳定性。
7. **安全门控下限**：`gate_lower_bound=-5.0`（配置），`safe_gate=True` + `lower_bound`（L623-624、L642）把门控值截断到 ≥ -5，防止递推爆炸/除零。
8. **输出门（full-rank gate）**：`use_full_rank_gate=True` → `g_proj(7168→12288)` 全秩门控（L531-537、L651-652；否则走低秩 g_a/g_b 两段式）；随后 `o_norm = FusedRMSNormGated(head_dim=128, activation='sigmoid')`（L539-540、L656）——RMSNorm 归一化 + sigmoid 门控逐元素缩放，再 `o_proj` 投影回 7168。
9. **两种执行模式**（L561）：训练/预填充用 `chunk`（`chunk_kda`，分块并行扫描，L609-627）；单 token 解码用 `fused_recurrent`（`fused_recurrent_kda`，O(1) 状态递推，L628-645）——这正是 1M 长上下文 + 高效解码的关键。训练时强制 chunk（L562-563）。
10. **状态管理**：`recurrent_states` 与 `conv_states` 写入 `KimiDynamicCache`（L646-649）；变长打包经 `get_unpad_data`/`pad_input`（L565-570、L660-661，L98-117）。

即：**KDA = 短卷积 + delta-rule 线性递推（学习衰减 A、数据相关 beta）+ 核内 sigmoid 门控/qk 归一化 + 全秩输出门 + RMSNorm-Gated 输出**。

### 2.4 Attention Residuals（AttnRes）——额外的跨层结构

- `attn_res_block_size=12`（配置）。`KimiDecoderLayer` 检测到该字段后启用 `_forward_attn_residual` 路径（L906-917、L930-934）：
  - 每个 block 起始层（`layer_idx % 12 == 0`）把 `prefix_sum`（本层输入 + 注意力输出）存入 `block_residual`（L995-998）；
  - 每层通过 `_apply_attn_res`（L1075-1088）对"已存 block 残差 + 当前 prefix_sum"做 RMSNorm 归一化（L1082-1083）、学习到的逐维权重打分（`norm.weight * proj.weight`，L1084-1085）、softmax 加权求和（L1086-1087）——即**用可学习门控在块内对多条残差路径做软选择**；
  - 模型末端再对最终输出施加一次（`KimiLinearModel` L1215-1217、L1226-1233）。
  - 效果：为 93 层深网提供跨层短路径，缓解梯度衰减与长上下文信息稀释（README L40/43 将 KDA 与 AttnRes 并列为两大架构创新）。

---

## 3. Gated MLA（Multi-Latent Attention + 输出门）

`KimiMLAAttention`（L335-474），结构上与 DeepSeek-V3 MLA 一致并加了输出门（`mla_use_output_gate=True`）：

- **维度**：`q_head_dim = qk_nope_head_dim(128) + qk_rope_head_dim(64) = 192`（L357）；`num_key_value_heads=96` → `num_key_value_groups=1`（L347）。
- **Q 低秩**：`q_a_proj(7168→1536)` → `KimiRMSNorm(1536)` → `q_b_proj(1536→96×192=18432)`（L364-373）。
- **KV 低秩压缩**：`kv_a_proj_with_mqa(7168→512+64=576)`（L378-382），拆成 latent(512) 与共享 rotary(64)（L426-428）；latent 过 `kv_a_layernorm` 后由 `kv_b_proj(512→96×(128+128))` 展开（L430-431），再拆为 k_nope(128) 与 v(128)（L432-433）。
- **共享 RoPE**：rotary 部分只投影一份（64 维），在头维上 `expand` 到 96 头共享（L435-437），随后 `q = cat(q_pass, q_rot)`、`k = cat(k_pass, k_rot)`（L439-440）。
- **Cache**：解码时把展开后的完整 k(192)/v(128) 写入 `KimiDynamicCache.update`（L442-444；L157-173 沿 seq 维拼接），注意这里缓存的是展开态而非 latent——与官方推理常用做法一致（节省 KV 的效果体现在低秩压缩本身，latent 仅 512+64 维）。
- **Flash-Attention-2 适配**：q_head_dim(192) ≠ v_head_dim(128) 时给 v 补零到 192、输出再截回 128（L446-448、L465-466）。
- **输出门（本模型特色）**：`g_proj(7168→96×128=12288)`，`sigmoid()` 后与注意力输出逐元素相乘，再进 `o_proj`（L470-473，模块定义 L398-401）。即每个 (头, 维度) 都有独立的 (0,1) 门控，让模型学习"关闭"部分注意力通道。README L75/115 称其为 **Gated MLA**。

---

## 4. MoE 路由：896 专家 Stable LatentMoE

### 4.1 门控 `KimiMoEGate`（L666-759）

- **打分**：线性 logits（float32 计算，L707-710）→ `sigmoid` 路由（`moe_router_activation_func="sigmoid"`，L711-712）。sigmoid 路由允许"多专家同时高分"，与 top-16 选取更匹配（相比 softmax 的归一化竞争）。
- **校正偏置**：`e_score_correction_bias`（L693-695）加到分数上**仅用于选专家**，权重仍取原始分数（L723、L750）——解耦"选择"与"加权"。
- **Grouped topk**（L724-746）：`use_grouped_topk=True` 但本检查点 `num_expert_group=1, topk_group=1`，故代码中 `if self.num_expert_group > 1 ...`（L724）分支实际不生效；框架层保留了 DeepSeek-V3 式的"组内 top2 求和 → 选组 → 组内 mask"机制（L725-744），留作通用能力。
- **Renormalize**：`top_k=16 > 1` 且 `moe_renormalize=True` → `topk_weight /= sum(topk_weight)`（L753-755），再乘 `routed_scaling_factor=1.0`（L757）。

### 4.2 稀疏块 `KimiSparseMoeBlock`（L762-874）

- **LatentMoE（低秩专家空间）**：`use_latent_moe` 由 `routed_expert_hidden_size` 是否存在决定（L776）；入口 `routed_expert_down_proj(7168→3584)`（L803-806、L821-822）把 token 压入 3584 维"潜在专家空间"，专家 FFN 在 3584 维上工作（`moe_hidden_size=3584`，L777-780），出口 `routed_expert_norm = KimiRMSNorm(3584)`（`latent_moe_use_norm=True`，L810-813、L829-832）+ `routed_expert_up_proj(3584→7168)`（L807-809）。低秩路由专家显著降低 MoE 参数量与访存。
- **896 个专家**：`ModuleList` 含 896 个 `KimiBlockSparseMLP`（L786-795），每个专家内部 `moe_intermediate_size=3072`（L791）。
- **共享专家**：`num_shared_experts=2` → 单个 `KimiMLP` 的 `intermediate_size = 3072×2 = 6144`（L797-801），输出与路由结果相加（L836-837）。
- **推理分发 `moe_infer`**（L840-874）：先按专家统计 token 数（L842-844）、`argsort` 重排实现 grouped-GEMM 式批处理（L845-846），逐专家前向（L852-860），按 `topk_idx` 回填（L866），乘权重求和（L867-873）。注意训练模式直接 `raise NotImplementedError`（L827）——该文件仅面向推理/微调（L24 注释）。
- **层分布**：`first_k_dense_replace=1` + `moe_layer_freq=1` → 仅第 0 层为稠密 `KimiMLP`，第 1~92 层全部 MoE（L893-900）。与 README"Number of Dense Layers: 1"（L71）一致。

---

## 5. SiTU 激活函数（`SituAndMul`，L64-85）

公式（L79-82，`x` 为拼接的 [gate | up]）：

```
situ_a = beta * tanh(gate / beta) * sigmoid(gate)
up'    = linear_beta * tanh(up / linear_beta)     # 仅当 linear_beta 设置时
out    = situ_a * up'
```

- **与 SiLU 的差异**：SiLU = `gate * sigmoid(gate)`，其中线性项 `gate` 无界；SiTU 把线性项替换为 `beta * tanh(gate/beta)` —— 当 `|gate| << beta` 时 `tanh(gate/beta) ≈ gate/beta`，退化为 SiLU；当 `|gate|` 增大时 tanh 饱和，**门控幅值被软裁剪到 ±beta，整体有界**，抑制激活值爆炸、稳定深网训练（本模型 `beta=4.0`）。
- **`linear_beta=25.0`**：对 `up` 分支做同样的软裁剪（边界远大于 gate，近似恒等，只防极端值），与 README"SiTU-GLU"（L119）一致。
- **实现细节**：`SituAndMul` 前半为 gate、后半为 up（L76-78）；计算在 float32 完成再转回原 dtype（L77-82）。注册进 `ACT2FN["situ"]`（L85），`KimiBlockSparseMLP`（L253-258）与 `KimiMLP`（L285-292）在 `hidden_act=="situ"` 时把 gate 与 up 投影结果 `cat` 后一次调用激活（L263-265、L296-297）。

---

## 6. 视觉编码器结构与图文融合

### 6.1 编码器流水线（`MoonViT3dPretrainedModel`，L649-717）

```
pixel_values(patch化, L×3)
  → MoonVision3dPatchEmbed: Conv2d 14×14 stride 14 (1024通道) + divided_fixed 位置编码
  → MoonViT3dEncoder: 27 × MoonViTEncoderLayer (1024 hidden, 12 heads, 2D RoPE, GELU-Tanh MLP)
  → tpool_patch_merger (sd2_tpool): 时间池化 + 2×2 空间合并 → 每 token 特征 ×4
  → PatchMergerMLPV2 (patchmergerv2): Linear(4096→4096) → GELU → Linear(4096→7168) → RMSNorm
  → 与文本 embedding 拼接（替换 <|media_begin|> 占位 token）
```

- 动态分辨率：每张图独立 (t,h,w)（`grid_thws`），patch embed 的可学习位置网格经 bicubic 插值适配（L265-272），2D RoPE 按子网格裁剪（L427-433）——NaViT 式"变长视觉 token 打包"（processor 侧 `navit_resize_image`/`navit_patchify` 与之配套）。
- 时空维度：`init_pos_emb 64×64×4`，时间用固定 sincos（`divided_fixed`，L248-253），空间可学习。
- 视觉注意力为全可见（`causal=False`，L91），与文本的因果注意力解耦。

### 6.2 融合（`KimiK3ForConditionalGeneration`）

- 图像特征提取与投影：`_extract_image_features`（L1092-1111）→ `mm_projector`（L1154-1155，patchmergerv2）。
- `_merge_input_ids_with_image_features`（L958-1090）：LLaVA 式占位替换——
  1. 以 `media_placeholder_token_id` 为界做"token 占用表"，把每个占位符展开成该图的特征长度（L992-999）；
  2. 用 `cumsum` 计算文本 token 的新位置（L1010-1016），支持左 padding 偏移（L1012-1014）；
  3. 散射写入文本 embedding 与 attention mask（L1048-1053），图像特征填满剩余槽位（L1060-1077）；
  4. 重算 `position_ids`（L1078-1079），图像区 label 置 `ignore_index`（L1054-1057、L1030-1036）。
- 生成阶段：单 token 解码时不再走图像分支（L1170-1207），用首层 cache 中"零向量 token"标记非参与 token 来扩展 attention mask，避免重复计算视觉特征。

---

## 7. `KimiK3ForConditionalGeneration` 的整体组织（L893-1317）

```
KimiK3ForConditionalGeneration
├── vision_tower : MoonViT3dPretrainedModel      (L902-903, 401M, MoonViT-V2)
├── mm_projector : PatchMergerMLPV2              (L905-917, patchmergerv2)
└── language_model : KimiLinearForCausalLM       (L919, 文本主干, 93层混合注意力+MoE)
    ├── embed_tokens (L1096-1097)
    ├── 93 × KimiDecoderLayer (L1098-1099)
    │     ├── layer 3,7,…,91,92 (0-based) → KimiMLAAttention (Gated MLA, 24层)
    │     ├── 其余 → KimiDeltaAttention (KDA, 69层)
    │     └── layer 0 稠密 KimiMLP；layer 1..92 → KimiSparseMoeBlock (896/16/2专家)
    ├── norm (L1100-1101) + AttnRes 末端门控 (L1215-1217)
    └── lm_head (L1247-1248)
```

- `__init__` 末尾把视觉塔/投影器对齐到语言模型 dtype（L922-925）。
- `forward`（L1113-1251）：文本 embedding →（有图时）视觉特征 + 投影 + 合并（L1150-1166）→ 主干（L1209-1218）→ 可选 CE loss（L1222-1239，shift 后按 attention mask 选取）→ `LlavaCausalLMOutputWithPast`（L1245-1251）。
- 生成支持：`prepare_inputs_for_generation`（L1253-1314）处理 cache 长度裁剪、位置 id 重算、首步传 `inputs_embeds` 后续传 `input_ids` 等。
- 量化（配置侧）：`compressed-tensors` `mxfp4-pack-quantized`，`group_size=32`，仅权重量化；`ignore` 列表排除 self_attn / shared_experts / mlp gate·up·down / lm_head / vision_tower / mm_projector —— **即只把 896 个路由专家的 FFN 权重压到 MXFP4**（参数大头），注意力、路由器与投影器保持高精度，兼顾压缩比与质量。

---

## 8. 最关键/最有特色的设计点（Top 5）

1. **KDA（Kimi Delta Attention）混合线性注意力**：DeltaNet 风格 delta-rule 递推（学习型指数衰减 `A_log` + 数据相关写入率 `beta` + 核内 sigmoid 门控与 q/k L2 归一化 + 安全下限 -5.0 + 短卷积 kernel=4），chunk/recurrent 双模式支撑 1M 上下文与 O(1) 解码；与 24 层 Gated MLA（每 4 层 + 最后 2 层）混排，`KimiDynamicCache` 统一管理两类状态——"全局检索能力 + 线性效率"的工程化平衡（L477-663、L120-223、L883-892）。

2. **Gated MLA（DeepSeek-V3 MLA + 输出门）**：低秩 Q(1536)/KV(512) 压缩 + 共享 RoPE(64) + 每(头,维)级 `sigmoid` 输出门调制注意力输出（`mla_use_output_gate`，L398-401、L470-473）——在 MLA 节省 KV 的基础上进一步给注意力注入可学习的"开关"。

3. **Stable LatentMoE（896 专家/16 topk）**：sigmoid 路由器 + 选择/加权解耦（`e_score_correction_bias`）+ topk 权重 renormalize；专家在 3584 维低秩潜在空间运行（down→norm→up 三段），2 个共享专家，只有路由专家 FFN 被 MXFP4 量化——"稀疏性 + 低秩 + 量化"三层压缩（L666-874）。README 称其对 K2 有约 2.5× 规模效率提升。

4. **SiTU-GLU 有界激活**：`beta·tanh(gate/beta)·sigmoid(gate)·up'`——用软饱和 tanh 替换 SiLU 的无界线性门，激活幅值有界（beta=4.0），`linear_beta=25.0` 软裁剪 up 分支，专为超深/超大模型训练稳定性设计（L64-85）。

5. **Attention Residuals（AttnRes）跨层残差门控**：每 12 层一个块，`_apply_attn_res` 用 RMSNorm + 学习投影打分 + softmax 在"历史块残差 + 当前前缀和"之间做软选择，末端再施加一次——为 93 层深网提供可学习的跨层短路径（L1075-1088、L973-1046、L1215-1217）。另有 NaViT 式动态分辨率视觉 tokenization + sd2_tpool 时空合并 + patchmergerv2 投影的视觉侧配套（作为补充亮点）。

---

*报告依据：`kimi_k3_modeling.py`、`kimi_k3_linear_modeling.py`、`kimi_k3_vision_processing.py` 全文逐行阅读；参数与层分布以 `moonshotai_Kimi-K3_config.json` 与 `kimi_k3_readme.md` 交叉验证。*
