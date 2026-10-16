# StreamRAG

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

## 免费 Embedding 配置（BGE-M3）

默认使用 SiliconFlow `BAAI/bge-m3`。在本项目根目录执行：

```powershell
uv run python scripts/stream_rag.py embedding-configure
uv run python scripts/stream_rag.py embedding-status
uv run python scripts/stream_rag.py embedding-status --live
```

第一条命令隐藏输入并将 Key 保存在 `.local-secrets/siliconflow-embedding.env`（Git 忽略）；第二条离线检查配置，第三条在授权下发送一条真实 Embedding 请求，不会打印密钥。当前没有发现本机已配置的 Key，必须先输入你自己的 SiliconFlow Key。API 的免费额度仍以服务提供者实际计费规则为准。

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

配置兼容 OpenAI Chat Completions 的 LLM 和 Embedding 端点后，执行：

```powershell
$env:STREAM_RAG_LLM_MODEL = "你实际可用的模型 ID"
$env:STREAM_RAG_LLM_API_KEY = "从安全密钥管理方式获取，不要存入 Git"
$env:STREAM_RAG_EMBED_API_KEY = "从安全密钥管理方式获取，不要存入 Git"

uv run python scripts/stream_rag.py refine --windows data/refined/nailv/windows.jsonl --output data/refined/nailv/refined_streams.jsonl --limit 3

# 人工检查输出的 summary/details 和来源后，再去掉 --limit 处理剩余窗口。
uv run python scripts/stream_rag.py refine --windows data/refined/nailv/windows.jsonl --output data/refined/nailv/refined_streams.jsonl

uv run python scripts/stream_rag.py ingest --windows data/refined/nailv/windows.jsonl --refined data/refined/nailv/refined_streams.jsonl --db data/lancedb/nailv
uv run python scripts/stream_rag.py catalog --refined data/refined/nailv/refined_streams.jsonl --output data/refined/nailv/stream_catalog.json
uv run python scripts/stream_rag.py report --refined data/refined/nailv/refined_streams.jsonl --output data/refined/nailv/quality_report.json
uv run python scripts/stream_rag.py search --db data/lancedb/nailv --query "对工作失误后感到难受的人如何回应？"
```

若使用的 LLM 或 Embedding 不在默认 SiliconFlow 地址，可分别指定 `--llm-url`、`--llm-model`、`--embed-url`、`--embed-model`。API Key 仅从环境变量读取；**不要求调用任何其他人托管的 RAG 服务**。

完全不调用 Embedding API 的本地 BM25 检索：

```powershell
uv run python scripts/stream_rag.py search --db data/lancedb/nailv --query "工作失误" --lexical-only
```

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
- 已通过 39 项本地回归测试，包括断点污染复现修复、LLM 故障恢复和索引发布回滚；尚未用真实约 10 万字直播文本完成完整 LLM + 真正 Embedding 的检索质量评测。

更完整的字段和操作说明见 [docs/STREAM_RAG_EIDOLON_BASELINE.md](docs/STREAM_RAG_EIDOLON_BASELINE.md)。
