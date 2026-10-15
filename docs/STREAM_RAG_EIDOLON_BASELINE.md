# StreamRAG：独立项目的 Eidolon 风格基线

> 状态：已实现离线流水线与基础检索接口；尚未用真实 10 万字语料或付费 LLM 完整跑通。
>
> 不是原 Eidolon 的源码或备份。实现参考其公开架构说明：
> https://laplace.live/eidolon （文档截至 2026-10-15 可访问，原 GitHub 仓库不公开）。

## 原版方法：哪些已经照着做

| 原版 Eidolon（公开文档） | 本地第一版 |
|---|---|
| laplace-transcribe：Whisper / Soniox | **输入阶段已接入现成 TXT / SRT / JSONL**；下载/ASR 留给单独工具 |
| laplace-refine：LLM 二次精炼直播内容 | refine：OpenAI-compatible LLM 输出 summary、details、topics、活动、风险 |
| 约 5 分钟流片段 | SRT/带时间的 JSONL 按 300 秒分组；超大窗口额外按字数分割；无时间 TXT 按字数回退，绝不假造时间戳 |
| laplace-ingest-streams：LanceDB | ingest：本地 LanceDB 的 stream_chunks 表；向量索引按需计算 |
| jieba 中文 BM25 + 向量融合 | jieba 预分词 + LanceDB FTS (BM25) + 向量查询 + RRF |
| reranker 可开关 | search 允许额外指定 Cohere 兼容 /rerank 服务，失败回退 RRF |
| laplace-build-stream-catalog | catalog：无需额外模型，按 source_id 聚合摘要与来源 |
| laplace-search | search：命令行查询实际索引 |
| laplace-eval-retrieval | eval：基于 gold JSONL 计算 recall@k、MRR |
| persona facts / persona.md / Mem0 / RetrievalGate | **暂不实现或接入**。StreamRAG 不储存 AI-Wife 或其他聊天系统的 Persona 和用户记忆 |
| 实时 Pipecat/STT/TTS | 不包含正式聊天运行时，不依赖任何特定聊天项目 |

**注意**：原项目的结构化 refine JSON 实际字段、完整 prompt、数据清洗逻辑和私有仓库没有公开；我们的字段只是**依据公开架构自行实现**，不是声称与原项目逐字段兼容。

## 文件

- scripts/stream_rag.py：CLI，包含 prepare / refine / ingest / catalog / report / search / eval / pipeline
- src/streamrag/corpus.py：TXT/SRT/JSONL → 真实来源窗口 → LLM 二次精炼 JSONL
- src/streamrag/retrieval.py：本地 LanceDB 和混合检索
- tests/test_stream_rag_pipeline.py：回归测试

## 环境

项目根目录为 D:\project\ai\StreamRAG，使用 PowerShell：

    uv sync --python 3.12
    uv run python scripts/stream_rag.py --help
    uv run python -m unittest discover -s tests -p test_stream_rag_pipeline.py -v

`uv sync` 只安装到 **StreamRAG 自己的 .venv**；不修改 AI-Wife、系统 Anaconda、已有 ASR/TTS 环境。下面各处的 `python` 命令前面统一加 `uv run`（或者先激活本项目的 .venv）。

## 输入约定

- **一名主播单独一套索引**；处理混合团播数据前先完成说话人分离。
- TXT：连续讲话；若没有时间戳，不会生成虚假的五分钟位置。
- SRT：每条字幕有真实起止时间，合并为约五分钟窗口。
- JSONL：每行 {"text":"...", "start_ms":1234, "end_ms":3456, "speaker_id":"nailv"}；也可用 start/end **秒**。
- 元数据 JSON（可选）：相对路径映射到来源信息，字段 source_id、bvid、date、title、url。例如：

    {
      "2026-02-20.srt": {
        "source_id": "nailv_20260220",
        "bvid": "BV1MSfVBYEqf",
        "date": "2026-02-20",
        "title": "来辣！！"
      }
    }

这些字段是原始来源的定位信息，不会自动成为 AI-Wife 角色的日程或记忆。上面的元数据示例只说明格式，必须确保实际输入文件确实来自该场录播。

## 运行步骤

先把有权处理的某位主播直播文字放在：

    data\transcripts\nailv\

### 1. 转换成约五分钟的原文窗口

    python scripts\stream_rag.py prepare --input data\transcripts\nailv --speaker-id nailv --output data\refined\nailv\windows.jsonl

如果有每场直播的来源标记：

    python scripts\stream_rag.py prepare --input data\transcripts\nailv --speaker-id nailv --metadata data\transcripts\metadata.json --output data\refined\nailv\windows.jsonl

### 2. LLM 精炼

LLM 必须是 Chat Completions 兼容接口。下面用 SiliconFlow 举例，也可以用本地 OpenAI 兼容模型。

    $env:STREAM_RAG_LLM_API_KEY = "请通过自己的安全配置提供真实 Key"
    $env:STREAM_RAG_LLM_MODEL = "你实际启用的模型名称"
    python scripts\stream_rag.py refine --windows data\refined\nailv\windows.jsonl --output data\refined\nailv\refined_streams.jsonl

建议先限制模型请求数量验证：

    python scripts\stream_rag.py refine --windows data\refined\nailv\windows.jsonl --output data\refined\nailv\refined_streams.jsonl --limit 3

refine 支持断点续跑；已生成的相同 ID 在来源 SHA、说话人、精炼模型与 prompt 签名均匹配时跳过；任一项变化都会阻止重用过时内容。需要重建时，使用一个**新的 workdir**。LLM 输出永远标记 auto_candidate，不能仅凭自动精炼断言“已人工审核”。

### 3. 建立本地 LanceDB 索引

    $env:STREAM_RAG_EMBED_API_KEY = "请通过自己的安全配置提供真实 Key"
    python scripts\stream_rag.py ingest --refined data\refined\nailv\refined_streams.jsonl --db data\refined\nailv\lancedb

默认使用配置好的 BAAI/bge-large-zh-v1.5（SiliconFlow OpenAI 兼容 embedding）；也可以指定 --embed-url、--embed-model、--embed-key-env 连接自托管 embedding 服务。

这**不是外部 RAG API**：文档和 LanceDB 索引都在自己本机。要完全脱离外部服务，请将 embedding、LLM 和 reranker 服务全部换成你控制的本地端点。

### 4. 聚合目录和查看质量

    python scripts\stream_rag.py catalog --refined data\refined\nailv\refined_streams.jsonl --output data\refined\nailv\stream_catalog.json
    python scripts\stream_rag.py report --refined data\refined\nailv\refined_streams.jsonl --output data\refined\nailv\quality_report.json

### 5. 混合搜索

    python scripts\stream_rag.py search --db data\refined\nailv\lancedb --query "有人工作失误很沮丧时主播怎么回应？"

不调用 embedding 服务的本地 **BM25-only** 搜索：

    python scripts\stream_rag.py search --db data\refined\nailv\lancedb --query "工作失误" --lexical-only

如果有 Cohere 兼容 rerank endpoint，可以另外加 --rerank-url、--rerank-key-env；没有就用 RRF。不能把 OpenRouter 的 chat URL 误当成 Cohere /rerank API。

### 6. 检索评测

gold.jsonl 每行至少包含：

    {"query":"怎么安慰工作很沮丧的朋友？","relevant_ids":["nailv_00001__00002"]}

    python scripts\stream_rag.py eval --db data\refined\nailv\lancedb --gold data\refined\nailv\gold.jsonl --top-k 3

输出 recall@k 和 MRR。金标准必须人工选取正确 ID，不能用同一个 LLM 生成的自测答案冒充人工标注。

### 7. 一次全流程（初次批处理）

    python scripts\stream_rag.py pipeline --input data\transcripts\nailv --speaker-id nailv --workdir data\refined\nailv

pipeline 会执行 prepare → refine → ingest → catalog → report。首次使用需正确配置 LLM 和 embedding Key/模型。发生部分精炼失败或设置 --limit 后，流水线会阻止构建不完整索引。

## 调试说明

--embed-mode hash-test --allow-test-embeddings 是仅用于测试数据库和查询代码的**假向量**，绝不是生产 embedding 或模型质量证明。模拟向量建立的索引不能与真实向量混用。

## 风险及下一步

1. 音频 ASR 的错误、多人说话及上下文缺失无法被 refine 自动消除。auto_candidate 未经验证，来源 SHA 只证明文本版本一致，不证明内容正确。
2. 需要授权或明确许可才能对第三方录播字幕进行相应使用；公开下载不等于拥有训练/再分发许可。
3. **目前不接正式 M0/M5 运行时**。未来接入时应使用仅检索证据、禁止写入 Persona/用户事实/私有记忆的受控接口。
4. 官方 Eidolon 有专用的 Retriever Gate、Pipecat 集成、persona facts、Mem0，我们的第一阶段只做 Stream Archive。
5. 同一个 RAG 索引的 embedding 模型和维度必须保持固定，更换模型需重新建库。
6. 初次验收至少需要一次有真实 LLM、真实向量的端到端构建，以及独立的人工检索测试集。
