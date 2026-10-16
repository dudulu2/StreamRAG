# 明前奶绿 · Gemini 4 云端精炼输入批次（2026-10-16）

本目录包含 **Python 确定性预处理生成的任务输入**，不是 AI 已经完成的 RAG 精炼结果。来自仓库中已有的三份原始 SRT：

- [NL04_clip153678.srt](../../raw/nailv/NL04_clip153678.srt)：健身环和情感电台，第 2 段
- [NL05_clip154606.srt](../../raw/nailv/NL05_clip154606.srt)：梦境吱花之国庆回忆，第 2 段
- [NL08_clip151266.srt](../../raw/nailv/NL08_clip151266.srt)：白日奶绿间，第 2 段

这里有 **41 个 5 分钟左右的文字窗口、21 个 Gemini 网页版任务文件**，全部具备不可擅自修改的 `batch_id`、`prompt_sha256`、`id` 与 `window_signature`。来源 SRT 与上传的原文完全一致（上传前已比较 SHA-256）。

## 云端 Gemini 4 Argon 的正确使用方法

1. 阅读 [项目 README](../../../README.md)、[Gemini 手动精炼工作流](../../../docs/ASTRA_GEMINI_MANUAL_WORKFLOW.md) 和 [格式定义](../../../src/streamrag/manual.py)。
2. 打开 [requests](requests/) 目录，按文件名排序，一次读取**一份** `batch_XXXX_*.md` 的完整内容。
3. 执行批次文件中内置的提示词，严格输出**唯一一个完整 JSON 对象**，字段仅限 `batch_id`、`prompt_sha256`、`items` 及 item 允许字段；不得省略窗口或修改签名。
4. 人工保存为 `result_0001.json` 等文件并放到本机的 `data/refined/nailv_gemini_pilot_20261016/exchange/responses/`。不要混合不同批次、不同模型的回复。
5. **区别内容归属**：主播自己的话、观众投稿、念出的故事、影视/游戏原声、其他人发言不能混为主播本人的经历。语料是来源可查的外部参考，不是可更换角色卡的个人记忆。

供 StreamRAG 导入器使用的**只读配套文件**：

- [windows.jsonl](windows.jsonl)：从原始字幕得到的完整 41 窗口原文及指纹
- [manifest.json](manifest.json)：21 个批次的窗口列表、提示词哈希和 windows 文件哈希
- [source_metadata.json](source_metadata.json)：原始直播来源

Gemini 只负责理解、细节保留和结构化输出，**不需要 Embedding**。本地 Qwen3-Embedding-0.6B 在导入严格验证后才用于建库。

## 本机导入校验

在 `D:/project/ai/StreamRAG` 中，先确保已经将 Gemini 返回的 JSON 保存到上文指定的 `responses` 目录。

```powershell
uv run python scripts/stream_rag.py import-refined --windows corpus_sources/gemini_batches/nailv_20261016/windows.jsonl --manifest corpus_sources/gemini_batches/nailv_20261016/manifest.json --responses data/refined/nailv_gemini_pilot_20261016/exchange/responses --output data/refined/nailv_gemini_pilot_20261016/refined_gemini4.jsonl --model gemini4-web
uv run python scripts/stream_rag.py validate --windows corpus_sources/gemini_batches/nailv_20261016/windows.jsonl --refined data/refined/nailv_gemini_pilot_20261016/refined_gemini4.jsonl --strict
```

注意：未复核原文里可能存在 ASR 识别错误、观众投稿和外部语音；程序签名只能证明输入完整，不能证明语义归属正确。当前目录没有任何完成的 Gemini 输出或正式 RAG 索引。
