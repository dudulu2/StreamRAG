# StreamRAG：独立项目的 Eidolon 风格基线

> 状态：已实现离线流水线、Gemini/Astra 手动文件交换、LanceDB 混合检索和基本测试；尚未用真实 10 万字语料完成端到端质量验证。
>
> 不是原 Eidolon 的源码或备份。实现参考其公开架构说明：
> https://laplace.live/eidolon （文档截至 2026-10-15 可访问，原 GitHub 仓库不公开）。

## 原版方法：哪些已经照着做

| 原版 Eidolon（公开文档） | 本地第一版 |
|---|---|
| laplace-transcribe：Whisper / Soniox | **输入阶段已接入现成 TXT / SRT / JSONL**；下载/ASR 留给单独工具 |
| laplace-refine：LLM 二次精炼直播内容 | refine：OpenAI-compatible LLM 输出 summary、details、topics、活动、风险 |
| 约 5 分钟流片段 | SRT/带时间的 JSONL 按 300 秒分组；超大窗口额外按字数分割；无时间 TXT 按字数回退，绝不假造时间戳 |
| laplace-ingest-streams：LanceDB | ingest：本地 LanceDB 的 stream_chunks 表；用临时目录构建、校验后替换旧索引 |
| jieba 中文 BM25 + 向量融合 | jieba 预分词 + LanceDB FTS (BM25) + 向量查询 + RRF |
| reranker 可开关 | search 支持 Cohere 兼容 /rerank 服务，失败回退 RRF；eval 中失败时直接报错 |
| laplace-build-stream-catalog | catalog：无需额外模型，按 source_id 聚合摘要与来源 |
| laplace-search | search：命令行查询实际索引 |
| laplace-eval-retrieval | eval：基于 gold JSONL 计算 recall@k、hit-rate@k、MRR；支持可选 Reranker |
| persona facts / persona.md / Mem0 / RetrievalGate | **暂不实现或接入**。StreamRAG 不储存 AI-Wife 或其他聊天系统的 Persona 和用户记忆 |
| 实时 Pipecat/STT/TTS | 不包含正式聊天运行时，不依赖任何特定聊天项目 |

**注意**：原项目的结构化 refine JSON 实际字段、完整 prompt、数据清洗逻辑和私有仓库没有公开；我们的字段只是**依据公开架构自行实现**，不是声称与原项目逐字段兼容。

## 文件

- scripts/stream_rag.py：CLI，包含 prepare / refine / ingest / catalog / report / search / eval / pipeline
- src/streamrag/corpus.py：TXT/SRT/JSONL → 真实来源窗口 → LLM 二次精炼 JSONL
- src/streamrag/manual.py：无 LLM API 的 Gemini 网页/Astra 本地批次导入导出、签名及原话校验
- src/streamrag/retrieval.py：本地 LanceDB 和混合检索
- tests/test_stream_rag_pipeline.py 和 tests/test_manual_workflow.py：回归测试

## 环境

项目根目录为 D:\project\ai\StreamRAG，使用 PowerShell：

    uv sync --python 3.12
    uv run python scripts/stream_rag.py --help
    uv run python -m unittest discover -s tests -p "test_*.py" -v

`uv sync` 只安装到 **StreamRAG 自己的 .venv**；不修改 AI-Wife、系统 Anaconda、已有 ASR/TTS 环境。下面各处的 `python` 命令前面统一加 `uv run`（或者先激活本项目的 .venv）。

## 无 LLM API 的 Gemini/Astra 操作方式

建议直接使用 [ASTRA_GEMINI_MANUAL_WORKFLOW.md](ASTRA_GEMINI_MANUAL_WORKFLOW.md)，它说明如何把带签名的窗口导出为 Gemini 网页可上传的 .md 文件、接收下载的 JSON、让 Astra 在本地生成同类文件、校验后再进入 LanceDB。原始逐字稿不能由模型重写；具体叙述、话语推进和真实引语是正式 RAG 数据，分类标签只用于辅助检索。下面仍保留传统 LLM API 自动精炼的操作说明。

## 输入约定

- **一名主播单独一套索引**；处理混合团播数据前先完成说话人分离。
- TXT：连续讲话；若没有时间戳，不会生成虚假的五分钟位置。
- SRT：每条字幕有真实起止时间，合并为约五分钟窗口。
- JSONL：每行 {"text":"...", "start_ms":1234, "end_ms":3456, "speaker_id":"nailv"}；也可用 start/end **秒**。**必须逐行提供可确认的目标说话人 ID**；缺失、null、unknown 或其他主播的行会跳过，并断开前后文关联。若没有说话人标签，请先由可靠方式为单主播语料补充标签，或导入纯 TXT（会标记为未核验）。
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

refine 支持断点续跑：逐窗口校验 **原文内容、邻近语境、窗口边界、分块参数、来源、说话人、模型和 prompt** 的签名，出现变化或旧结果多余时会拒绝合并。要改分块参数，请使用一个**新的 workdir**。失败窗口会立刻写到 `refined_streams.errors.jsonl`；模型恢复后重新执行同一 refine 命令，会只重试未完成的窗口并清理已恢复的错误记录。意外中断造成的 JSONL 未完成尾行会单独保存为 `.truncated-tail` 文件，再恢复已完成的行；中间有损坏的完整行则直接报错。LLM 输出仍标注 auto_candidate，不能仅凭自动精炼断言“已人工审核”。

### 3. 建立本地 LanceDB 索引

    $env:STREAM_RAG_EMBED_API_KEY = "请通过自己的安全配置提供真实 Key"
    python scripts\stream_rag.py ingest --windows data\refined\nailv\windows.jsonl --refined data\refined\nailv\refined_streams.jsonl --db data\lancedb\nailv

默认使用配置好的 BAAI/bge-large-zh-v1.5（SiliconFlow OpenAI 兼容 embedding）；也可以指定 --embed-url、--embed-model、--embed-key-env 连接自托管 embedding 服务。向量文本由 summary/details/topics 加上**长度受限的原文摘录**构成。`ingest --windows` 会拒绝不完整或过时的精炼文件。新 LanceDB 在临时目录完成 FTS 建索引后才发布，异常时保留旧数据库；如果新库已发布但旧目录因 Windows 文件占用无法清理，只警告并保留备份目录，不会误报发布失败。

这**不是外部 RAG API**：文档和 LanceDB 索引都在自己本机。要完全脱离外部服务，请将 embedding、LLM 和 reranker 服务全部换成你控制的本地端点。

### 4. 聚合目录和查看质量

    python scripts\stream_rag.py catalog --refined data\refined\nailv\refined_streams.jsonl --output data\refined\nailv\stream_catalog.json
    python scripts\stream_rag.py report --refined data\refined\nailv\refined_streams.jsonl --output data\refined\nailv\quality_report.json

### 5. 混合搜索

    python scripts\stream_rag.py search --db data\lancedb\nailv --query "有人工作失误很沮丧时主播怎么回应？"

不调用 embedding 服务的本地 **BM25-only** 搜索：

    python scripts\stream_rag.py search --db data\lancedb\nailv --query "工作失误" --lexical-only

如果有 Cohere 兼容 rerank endpoint，可以另外加 --rerank-url、--rerank-key-env；没有就用 RRF。若重排服务故障，搜索返回数据会设置 `reranker_status=fallback` 并发出告警；若成功则为 `ok`，未启用为 `disabled`。不能把 OpenRouter 的 chat URL 误当成 Cohere /rerank API。

### 6. 检索评测

gold.jsonl 每行至少包含：

    {"query":"怎么安慰工作很沮丧的朋友？","relevant_ids":["nailv_00001__00002"]}

    python scripts\stream_rag.py eval --db data\lancedb\nailv --gold data\refined\nailv\gold.jsonl --top-k 3

输出真正按相关条目覆盖率计算的 Recall@K、每题至少命中一条的 HitRate@K，以及 MRR。可在 eval 中使用与 search 相同的 `--rerank-url` / `--rerank-model`；评测期间若 Reranker 失败会报错，而普通查询仍可退回 RRF。金标准必须人工选取正确 ID，不能用同一个 LLM 生成的自测答案冒充人工标注。

### 7. 一次全流程（初次批处理）

    python scripts\stream_rag.py pipeline --input data\transcripts\nailv --speaker-id nailv --workdir data\refined\nailv

pipeline 会执行 prepare → refine → ingest → catalog → report。首次使用需正确配置 LLM 和 embedding Key/模型。发生部分精炼失败或设置 --limit 后，流水线会通过窗口内容签名和完整 ID 集校验，阻止构建不完整或新旧混合的索引。

## 调试说明

--embed-mode hash-test --allow-test-embeddings 是仅用于测试数据库和查询代码的**假向量**，绝不是生产 embedding 或模型质量证明。模拟向量建立的索引不能与真实向量混用。

## 风险及下一步

1. 音频 ASR 的错误、多人说话及上下文缺失无法被 refine 自动消除。auto_candidate 未经人工核验；来源 SHA 与窗口签名只证明版本一致，**不证明模型总结的事实正确**。JSONL 明确排除其他说话人后，会断开前后段落，避免连成同一句讨论。
2. 需要授权或明确许可才能对第三方录播字幕进行相应使用；公开下载不等于拥有训练/再分发许可。
3. **目前不接正式 M0/M5 运行时**。未来接入时应使用仅检索证据、禁止写入 Persona/用户事实/私有记忆的受控接口。
4. 官方 Eidolon 有专用的 Retriever Gate、Pipecat 集成、persona facts、Mem0，我们的第一阶段只做 Stream Archive。
5. 同一个 RAG 索引的 embedding 模型和维度必须保持固定，更换模型需重新建库。网页无 LLM API 不等于自动有本地 Embedding：可以连接本机兼容 /v1/embeddings 的模型服务，hash-test 不能用于生产。
6. 初次验收至少需要一次有真实 LLM、真实向量的端到端构建，以及独立的人工检索测试集。
