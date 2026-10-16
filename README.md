# StreamRAG

## AI-Wife 只读接入（2026-10-16）

新增常驻 `scripts/companion_worker.py`：预热本地 Qwen 后通过 stdin/stdout JSONL 返回带作者归属的事实、互动和原话候选，不写聊天应用的记忆。AI-Wife 新角色启动命令为 `npm run m0:streamrag`，旧角色和旧记忆保留。协议与边界见 [COMPANION_READONLY_WORKER.md](docs/COMPANION_READONLY_WORKER.md)。

本地修订语料和 Gemini 输出继续保留在本机，但 `data/attribution/` 与 `gemini_output/` 不随本次代码提交公开。

**独立的直播文本 → Gemini 网页版 / Astra 本地 / 可选 LLM API 精炼 → LanceDB 检索库工具。**

路径：`D:\project\ai\StreamRAG`。无需安装或导入 AI-Wife，也不读取 AI-Wife 的配置、Persona 或用户记忆。

> 工程基线参考 LAPLACE Eidolon 的公开文档，不是对未公开源码的复制。当前是 RAG **离线构建 + 本地查询**工具；尚无正式直播语料的 LLM 端到端质量验收。

## 目录

```text
StreamRAG/
├── pyproject.toml               # 独立 Python 依赖
├── uv.lock                      # 可复现依赖锁（uv sync 后生成）
├── scripts/
│   └── stream_rag.py            # CLI
├── src/
│   └── streamrag/
│       ├── corpus.py            # TXT/SRT/JSONL、窗口、LLM refine、目录和报告
│       ├── manual.py            # Gemini/Astra 离线文件批次导入导出与证据验证
│       └── retrieval.py         # LanceDB、jieba + BM25、向量、RRF、可选 reranker
├── tests/
│   └── test_stream_rag_pipeline.py
├── docs/
│   └── STREAM_RAG_EIDOLON_BASELINE.md
└── data/
    ├── transcripts/             # 自己提供的原始直播文本（忽略 Git）
    ├── refined/                 # JSONL、摘要、质量报告（忽略 Git）
    └── lancedb/                 # 本地索引库（忽略 Git）
```

**默认 Embedding 完全离线：使用本地 Qwen3-Embedding-0.6B（Python / CUDA 或 CPU 直加载），不使用 SiliconFlow，不需要云端 Embedding API Key。GitHub 只包含代码和下载方法，不包含模型权重；`qwen/` 已被 Git 忽略。**

## 本地 Qwen3-Embedding-0.6B

默认直接加载本项目 `qwen/` 内的官方模型，使用本机 GPU（无 CUDA 时使用 CPU）。无需 API Key，无需启动 Embedding 服务，不读取旧云端密钥，也不会回退到云端。

```powershell
uv sync --python 3.12
uv run python scripts/stream_rag.py embedding-download
uv run python scripts/stream_rag.py embedding-status
uv run python scripts/stream_rag.py embedding-status --live
```

下载命令只需首次运行，权重约 1.2 GB，支持续传；`qwen/` 已被 Git 忽略。`--live` 实际加载本地模型并检查文档和查询的 1024 维向量。之后建库和查询完全离线。本地推理依赖已纳入默认安装。

可以用 `--embed-model-path D:/path/to/model` 或 `STREAM_RAG_EMBED_MODEL_PATH` 复用模型；`--embed-device cpu` 强制 CPU。默认 GPU 内部推理批次为 1，可用 `--embed-local-batch-size` 调整。

查询添加任务指令，文档不加指令；输出单位归一化向量，按 cosine 检索。模型配置和查询编码契约写入索引，配置变化会拒绝混用；旧 BGE/API/hash 索引必须用真实本地模型重建。仅关键词检索不加载模型。

官方模型说明：https://huggingface.co/Qwen/Qwen3-Embedding-0.6B

## 安装与测试（Windows PowerShell）

在本项目目录执行：

```powershell
cd D:\project\ai\StreamRAG
uv sync --python 3.12
uv run python scripts/stream_rag.py --help
uv run python -m unittest discover -s tests -p test_stream_rag_pipeline.py -v
```

依赖保存在本目录自己的 `.venv`，与其他项目完全隔离。

## 推荐方式：Astra 本地 / Gemini 4 网页版，不调用 LLM API

模型只分析文本，StreamRAG 负责原文来源、校验和入库。**当前不仅保存 topic 分类，还保留具体的 details、dialogue_flow、interaction_notes、verbatim_excerpts 和完整 raw_text。** 下游查询附带自然语言 reference_text，不把标签当作聊天内容。

将单主播文字放在 data/transcripts/nailv/，先运行 prepare，再按以下步骤：

```powershell
uv run python scripts/stream_rag.py prepare --input data/transcripts/nailv --speaker-id nailv --output data/refined/nailv/windows.jsonl
uv run python scripts/stream_rag.py export-batches --windows data/refined/nailv/windows.jsonl --outdir data/refined/nailv/exchange
```

随后将 exchange/requests 下的每个 .md 文件交给 **Gemini 4 网页版**（复制回复 JSON 放进 exchange/responses/），或让拥有本地文件权限的 **Astra** 直接生成 JSON 文件。然后：

```powershell
uv run python scripts/stream_rag.py import-refined --windows data/refined/nailv/windows.jsonl --manifest data/refined/nailv/exchange/manifest.json --responses data/refined/nailv/exchange/responses --output data/refined/nailv/refined_gemini4.jsonl --model gemini4-web
uv run python scripts/stream_rag.py validate --windows data/refined/nailv/windows.jsonl --refined data/refined/nailv/refined_gemini4.jsonl --strict
```

**详细逐步说明**：[Astra/Gemini 无 API 工作流](docs/ASTRA_GEMINI_MANUAL_WORKFLOW.md)。Gemini 和 Astra 的输出请使用不同的 refined JSONL 及 response 目录，不混合模型。

## 可选：旧的自动 LLM API 流程

将**同一名主播**的 TXT/SRT/JSONL 放到 `data/transcripts/nailv/`。输入已有文字，无须先运行 ASR。

```powershell
uv run python scripts/stream_rag.py prepare --input data/transcripts/nailv --speaker-id nailv --output data/refined/nailv/windows.jsonl
```

网页批次不需要 LLM API。可选自动精炼需显式指定自己的 Chat Completions 端点；没有默认云端服务。Embedding 始终默认本地 Qwen。执行：

```powershell
$env:STREAM_RAG_LLM_URL = "你选择的 Chat Completions 服务地址"
$env:STREAM_RAG_LLM_MODEL = "你实际可用的模型 ID"
$env:STREAM_RAG_LLM_API_KEY = "从安全密钥管理方式获取，不要存入 Git"

uv run python scripts/stream_rag.py refine --windows data/refined/nailv/windows.jsonl --output data/refined/nailv/refined_streams.jsonl --limit 3

# 人工检查输出的 summary/details 和来源后，再去掉 --limit 处理剩余窗口。
uv run python scripts/stream_rag.py refine --windows data/refined/nailv/windows.jsonl --output data/refined/nailv/refined_streams.jsonl

uv run python scripts/stream_rag.py ingest --windows data/refined/nailv/windows.jsonl --refined data/refined/nailv/refined_streams.jsonl --db data/lancedb/nailv
uv run python scripts/stream_rag.py catalog --refined data/refined/nailv/refined_streams.jsonl --output data/refined/nailv/stream_catalog.json
uv run python scripts/stream_rag.py report --refined data/refined/nailv/refined_streams.jsonl --output data/refined/nailv/quality_report.json
uv run python scripts/stream_rag.py search --db data/lancedb/nailv --query "对工作失误后感到难受的人如何回应？"
```

可选 API 后端只有显式设置 `--embed-mode api` 才会使用；本地默认模式不会读取 API Key。LLM API 精炼需通过 `--llm-url`、`--llm-model` 或对应环境变量指定。

完全不调用 Embedding API 的本地 BM25 检索：

```powershell
uv run python scripts/stream_rag.py search --db data/lancedb/nailv --query "工作失误" --lexical-only
```

## 外部字幕来源索引（不包含第三方原文）

[corpus_sources/README.md](corpus_sources/README.md) 提供明前奶绿、七海、阿梓的 **37 份第三方现成字幕来源**，含逐份官方/原站下载地址、SHA-256 和待复核质量标签：[JSON 清单](corpus_sources/zimu_20261016.json)。

可在有使用权限的本地环境运行：

```powershell
python scripts/fetch_zimu_transcripts.py --speaker nailv
python scripts/fetch_zimu_transcripts.py --verify-only
```

字幕源文件保留在被 Git 忽略的 `data/transcripts/`；未经授权的第三方完整字幕不提交到公开仓库。**仅给 Gemini 一个 GitHub 地址不足以证明它已经读到字幕原文**；需核验能否实际打开每条原站 `source_url`，否则另行提供有权处理的原稿。主播语料与可更换的角色卡相互独立。
## 隔离与限制

- **一个主播一套索引**。不要把团播、观众话语与多名主播直接混为一个说话人。对于 JSONL，必须逐行提供可匹配 `--speaker-id` 的 `speaker_id` / `speaker`；未标注或标注为未知的发言会被跳过。
- 输入和输出是文件；不会修改其他项目数据库或记忆。
- 五分钟窗口遵循**真实字幕时间戳**；纯 TXT 缺少时间戳时以文字长度切片，不虚构直播时间。
- 模型生成自然语言 `summary`、`details`、`dialogue_flow`、`interaction_notes`、逐字引语 `verbatim_excerpts`，`topics` 只是辅助标签；完整 `raw_text` 由程序截取。网页导入标注 `web_import_candidate`，API 精炼标注 `auto_candidate`，都不是人工审核通过。
- `refine` 对每个窗口记录原文与分块配置签名，拒绝把旧分块误用为新的结果；错误窗口实时记录到 `refined_streams.errors.jsonl`，之后重新执行同一命令即可恢复。
- `ingest --windows` 和完整 `pipeline` 会核对所有片段是否齐全且版本一致；索引通过临时目录构建，成功后才替换旧库。
- `eval` 支持 Reranker，并分别报告 Recall@K、HitRate@K 和 MRR；评测时重排出错直接失败，不把 RRF 回退伪装成重排评测。
- `stream_catalog.json` 只供语料来源追溯，不是聊天角色的日程。
- `--embed-mode hash-test --allow-test-embeddings` 为**测试替身**，不可作为生产向量使用。
- 公开的直播字幕、音频、整理稿仍需核对各自许可。
- 具备本地回归测试，包括断点污染复现修复、LLM 故障恢复和索引发布回滚；尚未用真实约 10 万字直播文本完成完整 LLM + 真正 Embedding 的检索质量评测。

更完整的字段和操作说明见 [docs/STREAM_RAG_EIDOLON_BASELINE.md](docs/STREAM_RAG_EIDOLON_BASELINE.md)。


## 归属修订与三层检索

奶绿第二阶段修订入口：`scripts/three_layer_rag.py`。原始 Gemini 输出保持不变。

- 修补清单：`data/attribution/nailv_20261016/semantic_patch.json`，逐窗口语义判断，绑定输入哈希。
- 当前修订：`data/attribution/nailv_20261016/revision_02/`。`records.jsonl` 是检索数据；`author_spans.jsonl` 保存作者边界；`boundary_changes.jsonl` 可对照改动；`v1_audit.jsonl` 保留 160 条旧摘录复核。
- 本地索引：`data/lancedb/nailv_three_layer_v1`，独立于旧的窗口级索引。

```powershell
.venv\Scripts\python.exe -B scripts\three_layer_rag.py validate
.venv\Scripts\python.exe -B scripts\three_layer_rag.py query --scope facts --query "奶绿喜欢什么香味？"
.venv\Scripts\python.exe -B scripts\three_layer_rag.py query --scope style --query "怎样回应受伤求安慰的观众？"
.venv\Scripts\python.exe -B scripts\three_layer_rag.py query --scope attribution --query "谁在达卡打灰三年？"
```

`context` 返回原文与观众情境，`facts` 返回主播自述、观点等，`style` 返回原句片段及互动。`attribution` 对照主播记录与观众记录。过滤先于召回数量限制，使用本地 Qwen + 中文 FTS + RRF。`--epistemic self_report` 可只查自述；`--min-similarity` 可设置向量门槛，默认不猜阈值。

原句放在 `evidence[].text`，分段保留，不拼成连续引语。`trigger_evidence` 是观众触发原句；缺失时用 `trigger_kind=context_inferred`。`text`、`model_analysis` 是分析文字。`author` 表示文字归属；`spoken_by=unverified`、`speaker_verified=false` 表示未做音频核验。上下文通过 `window_id` 另查，不混进事实或表达结果。检索返回候选证据，不直接生成断言。

需要重新生成时，指定新的修订和索引目录：

```powershell
.venv\Scripts\python.exe -B scripts\three_layer_rag.py repair --revision data/attribution/nailv_20261016/revision_new
.venv\Scripts\python.exe -B scripts\three_layer_rag.py index --revision data/attribution/nailv_20261016/revision_new --index data/lancedb/nailv_three_layer_new
.venv\Scripts\python.exe -B scripts\evaluate_three_layer.py --output data/attribution/nailv_20261016/retrieval_evaluation_after.json
```

这批黄金问题用于开发回归；端到端回答质量和独立测试集仍需另测。ai-wife 接入时把这些记录作为外部直播素材，保留作者、证据及类型字段。

当前验收：41 个窗口、450 条记录，189 处边界调整；756 段直接证据通过逐字与归属校验。10 个开发回归问题 Hit@5=100%，MRR@5=0.9333。单主播索引按配置去除查询中的主播全名再检索主题，不改变证据作者。无答案探针仍会返回相似候选，调用方须根据证据决定能否回答；本轮未测试回答生成。
