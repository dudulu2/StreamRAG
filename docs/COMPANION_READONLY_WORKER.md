# 给聊天应用使用的只读检索进程

新增 `scripts/companion_worker.py`：在 StreamRAG 自己的 Python 环境运行本地 Qwen，加载已审核的三层索引，通过 stdin/stdout JSONL 提供只读服务。它不需要公网端口、Embedding API Key 或聊天模型 API。

## 启动

```powershell
.venv\Scripts\python.exe -B -u scripts\companion_worker.py --index data\lancedb\nailv_three_layer_v1 --revision data\attribution\nailv_20261016\revision_02 --windows corpus_sources\gemini_batches\nailv_20261016\windows.jsonl
```

模型默认读取本项目 `qwen/`。启动时校验修订文件、索引清单、源说话人及模型契约，预热 Qwen 和中文全文索引后输出 `event=ready`。同一进程处理多轮查询，避免每次重新加载模型。

## 输入输出

```json
{"id":"request_1","action":"search","query":"怎样回应工作受挫的朋友？","scope":"style","top_k":3}
```

scope 只接受 style、facts、attribution。最多返回 5 条，有效证据文本有边界预算。不会把不同 source_span 拼成假连续引语；原文逐段核验，作者与模型分析保持分开。

`answer_supported=null`：候选不是“这个问题已被证明有答案”。调用方仍需判定是否相关、有无足够证据，以及如何表述来源和不确定性。

可发送 `health`、`shutdown`，或明确的 `context` + `window_id` 回查混合上下文。混合上下文绝不自动并入 facts/style 结果。

## 保守筛选与权限

默认排除 context_only、hearsay、hypothetical、缺少可核对原文的记录。所有资料均标记不能自动变为角色记忆。音频说话人仍未核验，不能把字幕精确匹配当作音频真实性证明。

进程不更新 LanceDB、不修改原始语料、不写 AI-Wife 的数据库。模型推理诊断输出到 stderr，stdout 仅供结构化协议使用。消费者应限制请求量、设置截止时间、取消时丢弃迟到结果，并在失败时回退普通聊天。

## 与 AI-Wife 的关系

AI-Wife 的新角色启动器通过独立子进程连接这里。StreamRAG 不 import AI-Wife，也不读取其用户记忆、Persona 或 API Key；修改聊天角色名称不需要改动来源说话人或重新精炼这批语料。

角色的人生设定与真实新聊天由 AI-Wife 管理；当前奶绿资料仅作为外部表达、来源事实与归属参考。历史日程不得映射成角色现在的日程。

## 验证

```powershell
.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_companion_worker.py -v
.venv\Scripts\python.exe -B -m unittest discover -s tests -p "test_*.py" -q
```

`test_companion_worker.py` 使用合成数据，不依赖私有直播材料。`test_attribution.py` 依赖本地修订资料；公开克隆缺少时会明确 skip，不能把 skip 说成通过真实语料测试。

`data/attribution/` 和 `gemini_output/` 现在由 Git 忽略。本地语料已保留并可使用，但不在这次集成提交中公开。公开仓库里此前已存在的输入文件不在本次变更中自动移除。
