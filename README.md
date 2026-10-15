# StreamRAG

**独立的直播文本 → LLM 精炼记忆 → LanceDB 检索库工具。**

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

## 安装与测试（Windows PowerShell）

在本项目目录执行：

```powershell
cd D:\project\ai\StreamRAG
uv sync --python 3.12
uv run python scripts/stream_rag.py --help
uv run python -m unittest discover -s tests -p test_stream_rag_pipeline.py -v
```

依赖保存在本目录自己的 `.venv`，与其他项目完全隔离。

## 最快开始

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

uv run python scripts/stream_rag.py ingest --refined data/refined/nailv/refined_streams.jsonl --db data/lancedb/nailv
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

- **一个主播一套索引**。不要把团播、观众话语与多名主播直接混为一个说话人。
- 输入和输出是文件；不会修改其他项目数据库或记忆。
- 五分钟窗口遵循**真实字幕时间戳**；纯 TXT 缺少时间戳时以文字长度切片，不虚构直播时间。
- 模型生成 `summary`、`details`、`topics`；`raw_text` 由程序截取，数据默认标注 `auto_candidate`。
- `stream_catalog.json` 只供语料来源追溯，不是聊天角色的日程。
- `--embed-mode hash-test --allow-test-embeddings` 为**测试替身**，不可作为生产向量使用。
- 公开的直播字幕、音频、整理稿仍需核对各自许可。
- 尚未用真实约 10 万字直播文本完成完整 LLM + 真正 Embedding 的检索质量评测。

更完整的字段和操作说明见 [docs/STREAM_RAG_EIDOLON_BASELINE.md](docs/STREAM_RAG_EIDOLON_BASELINE.md)。
