# Gemini 4 网页版 / Astra 本地文件：无 LLM API 批量精炼

> 工程目的：**人工或 Agent 处理原文，程序负责证据和版本。**
> 不需要订阅第三方 RAG 服务，也不需要调用付费 LLM API。
> 建立真正的语义向量索引仍需要 Embedding：可以接 OpenAI 兼容的本地服务或可用的 Embedding API。测试用 hash-test **不具备真实语义能力**。

## 为什么输出不能只有分类词？

StreamRAG 的知识单元是一个**完整直播文字窗口**，不是标签数据库。每个窗口保存：

| 字段 | 作用 |
| --- | --- |
| raw_text | 本地从原文件截取的**完整原文**，绝不由 LLM 回填/改写 |
| details | 主体：保留谈话具体经过、故事、例子、转折和原观点 |
| dialogue_flow | 按实际发生顺序解释话语如何推进，使用完整句子 |
| interaction_notes | 具体说明主播如何响应、解释、举例或转换角度 |
| verbatim_excerpts | **从原文逐字复制**、可由程序逐条校验的真实讲话片段 |
| trigger_context | 原始素材有明确触发上下文才填写，否则 null |
| topic / topics / activity | 辅助检索与分类，**不代替具体内容** |
| summary / scene_summaries | 提供概览，但不能代替 details 和原始讲话 |

对保留的较长窗口，网页导入要求 **details 至少 60 字、dialogue_flow 非空且逐字引用至少一段真实原话**。这只是一道防止退化成标签表的工程下限，**并不代表质量已经得到验证**。真实含义、歧义、说话人归属、模型总结是否合理，仍然需要检查。

模型绝不可以把主播的日程、经历、关系写成后续 AI 角色的亲身记忆。索引结果的 reference_text 也会明确标记“外部直播对话参考”。

## 一、准备原始资料

在独立 StreamRAG 项目根目录使用 PowerShell：

    cd D:\project\ai\StreamRAG
    uv sync --python 3.12

将**有权使用、单主播、来源明确**的文字放在：

    data/transcripts/host01/live01.txt

如果输入是 JSONL，每行必须包含准确的 speaker_id。SRT 和 TXT 无法自动确立说话人身份，必须从录播来源人工确认。程序不会凭空补充时间戳。

执行：

    uv run python scripts/stream_rag.py prepare --input data/transcripts/host01/live01.txt --speaker-id host01 --source-id live01 --output data/refined/host01/windows.jsonl

SRT 带时间戳时按约 5 分钟分窗，过长再按字数拆分；没有时间戳的 TXT 按字数拆分。

## 二、导出 Gemini/Astra 通用批次

    uv run python scripts/stream_rag.py export-batches --windows data/refined/host01/windows.jsonl --outdir data/refined/host01/exchange --max-windows 2 --max-chars 8000

输出类似：

    data/refined/host01/exchange/
      manifest.json                    # 每个窗口ID/签名和提示词版本
      requests/
        batch_0001_XXXXXXXXXXXX.md    # 可直接上传 Gemini 或让 Astra 读取
        batch_0002_XXXXXXXXXXXX.md
      responses/                      # 接收模型的 JSON 输出

**不要修改**批次文件的 id、签名和 main_source_text。整个 manifest 与 windows.jsonl 必须来自同一次 prepare。批次参数或原文变化时创建新的 exchange 目录，不能复用旧回复。

### Gemini 4 网页版

1. 将 requests 下单个 .md 文件粘贴或上传到 Gemini 网页版；每个批次最好新开一轮清晰会话，避免此前聊天指令影响结果。
2. 要求严格按照文件中的提示词输出一个完整 JSON 对象，必须含 batch_id、prompt_sha256 和该批全部 items；不要让 Gemini 重新生成 raw_text。
3. 将纯 JSON 下载为文件（如 result_01.json），保存到同一 exchange/responses 目录；**不要上传仓库**。
4. 模型省略任一窗口、改动签名、写出不存在的“逐字引语”或只给短标签时，导入会拒绝此批回复。根据本地提示重新让模型修正。
5. 随时可以导入已完成的批次；无需等待全部处理结束。

### Astra / GPT-6 Pro 本地 Agent

直接授权其**仅在 StreamRAG 的 exchange/requests 与 exchange/responses 下读写**，让 Agent 按 .md 输入文件逐批生成 .json 回复；不得编辑原始 windows.jsonl、manifest 或正式 refined 文件。之后让下面的导入器核验。模型即使拥有本地文件权限，依然不能绕过来源校验。

请勿让 Agent 直接填写完整的 refined_streams.jsonl；由 importer 附加真实 raw_text、source_id、窗口签名和来源信息。

## 三、导入并验证（不调用任何 LLM）

Gemini：

    uv run python scripts/stream_rag.py import-refined --windows data/refined/host01/windows.jsonl --manifest data/refined/host01/exchange/manifest.json --responses data/refined/host01/exchange/responses --output data/refined/host01/refined_gemini4.jsonl --model gemini4-web

Astra：处理好相同或单独目录的请求后，建议另建一份输出：

    uv run python scripts/stream_rag.py import-refined --windows data/refined/host01/windows.jsonl --manifest data/refined/host01/exchange/manifest.json --responses data/refined/host01/astra_responses --output data/refined/host01/refined_astra.jsonl --model astra-local

注意：**同一输出文件不能混用不同模型**。要对比 Gemini 和 Astra，使用同一套窗口与提示词，但单独保存两组回复/结果。

查看完整性与粗粒度质量统计：

    uv run python scripts/stream_rag.py validate --windows data/refined/host01/windows.jsonl --refined data/refined/host01/refined_gemini4.jsonl

全部完成才使用严格模式：

    uv run python scripts/stream_rag.py validate --windows data/refined/host01/windows.jsonl --refined data/refined/host01/refined_gemini4.jsonl --strict

导入期间的规则：

- 必须返回**整批**全部窗口；不能插入重复 ID 或混入其他批次
- 所有来源字段只由本地程序从已签名原始窗口恢复
- 原文引语必须是 raw_text 的**字面子串**（不是“意思接近”）
- 精炼长度过短、遗漏话语推进或原话摘录的长窗口拒绝导入
- 多次执行会跳过完全相同的已处理结果，**相同 ID 结果不一致则拒绝覆盖**
- 正式文件使用写入后原子替换，避免因部分失败产生半个批次
- 所有导入结果默认 web_import_candidate，**不是人工事实核验通过**

## 四、先配置真实 Embedding

默认 SiliconFlow `BAAI/bge-m3`，不依赖 AI-Wife 的记忆库或应用配置；Key 不应上传到公开 GitHub。

```powershell
uv run python scripts/stream_rag.py embedding-configure
uv run python scripts/stream_rag.py embedding-status
uv run python scripts/stream_rag.py embedding-status --live
```

第一条命令在终端隐藏输入并保存于 Git 忽略的本地 `.local-secrets/siliconflow-embedding.env`；第二条离线检查 Key 是否配置；第三条显式发送一条联网测试，验证真实服务可用性与向量维度。已有环境变量 `STREAM_RAG_EMBED_API_KEY` 时，环境变量优先于本地文件。没有 Key 时联网测试不会运行。免费额度和速率限制取决于服务商当前政策。

## 五、建立 RAG 索引并搜索

语义向量模型可以是外部 Embedding 接口，也可以是**本机运行的、提供 OpenAI-compatible /v1/embeddings 接口的服务**。例如已在本机正常启动相关服务时：

    $env:STREAM_RAG_EMBED_API_KEY = "local-only-placeholder"
    uv run python scripts/stream_rag.py ingest --windows data/refined/host01/windows.jsonl --refined data/refined/host01/refined_gemini4.jsonl --db data/lancedb/host01_gemini --embed-url http://127.0.0.1:1234/v1 --embed-model LOCAL_MODEL_ID

**端口与 LOCAL_MODEL_ID 必须填写你实际运行的本地服务值，不是本仓库自动启动的服务**。如果用云端 Embedding，则按 README 配置相应密钥。

使用相同 embedding 模型搜索：

    uv run python scripts/stream_rag.py search --db data/lancedb/host01_gemini --query "朋友因为工作失败心情不好时，主播怎么展开话题？" --embed-url http://127.0.0.1:1234/v1 --embed-model LOCAL_MODEL_ID

不用任何 Embedding 调用的本地关键词搜索：

    uv run python scripts/stream_rag.py search --db data/lancedb/host01_gemini --query "工作失败" --lexical-only

注意：BM25 搜索是本地运行的，但 **必须先创建索引**；当前生产级向量建库仍要求真实 Embedding。hash-test 仅用于回归测试。

search 输出包括 details、dialogue_flow、interaction_notes、verbatim_excerpts、完整 raw_text，以及适合直接供下游 LLM 阅读的 reference_text。标签不是最终的回答文本。

## 五、验收和下一阶段

先用 10-20 个同样的直播窗口分别交给 Gemini、Astra 和 GPT-6 Pro，人工盲评：

1. **准确性**：是否出现原文没有的事实、情感或观众提问。
2. **情节/论点细节保留率**：能否覆盖真实例子、前因、转折及结论。
3. **话语推进质量**：dialogue_flow 是否真实、是否符合发生顺序。
4. **引用可验证率**：verbatim_excerpts 是否完全在原文中。
5. **结构与完整性**：JSON 能否直接导入，有无漏窗口或来源错配。

还应对比**实际检索**的 top-k 命中效果以及接入下游聊天后是否产生自然、贴近情境的回答。当前模块未与 AI-Wife 正式聊天/记忆运行时集成，不能以单元测试代替最终对话效果。

**隐私及授权提醒：** 使用网页版模型可能将上传的直播文本发送到服务商。请确认有权上传和处理该资料，并遵守相应服务条款及适用版权要求。data 目录已由 .gitignore 排除，不应将原始语料/模型回复直接提交到公开仓库。
