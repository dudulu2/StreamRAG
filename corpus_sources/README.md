# StreamRAG / 2026-10-16 第三方字幕来源目录

这是**现成主播字幕的来源索引，不是字幕正文仓库**。公开发布的 `zimu_20261016.json` 含 37 份字幕的稳定 clip_id、公开原站 SRT 下载链接、本机保存相对路径、原始文件 SHA-256、统计及筛选标记；不公开第三方完整逐字稿。

## 覆盖情况

| 主播 | 标识 | 文件数 | 已下载文本正文字符（约） |
| --- | --- | ---: | ---: |
| 明前奶绿 | `nailv` / NL01–NL11 | 11 | 14.8 万 |
| 七海 Nana7mi | `nana7mi` / NM01–NM11 | 11 | 8.6 万 |
| 阿梓 | `azi` / AZ01–AZ15 | 15 | 2.8 万 |

来源网站：https://std.zimu.live/ 。逐份源地址见 [zimu_20261016.json](zimu_20261016.json)。

### 在有使用权限的本机恢复原始 SRT

先进入 StreamRAG 仓库，运行：

```powershell
python scripts/fetch_zimu_transcripts.py --speaker nailv
python scripts/fetch_zimu_transcripts.py --verify-only
```

默认写入 **已被 .gitignore 排除** 的 `data/transcripts/_candidates_20261016/<speaker_id>/`。不指定 `--speaker` 则取全体 37 份。默认严格对比原始文件 SHA-256，来源数据发生变化时会报告不匹配，不会悄悄把它当作同一版语料。

### 交给 Gemini 网页版前的须知

- GitHub 公开仓库上**只有来源列表，不包含原始 SRT、Gemini 处理批次内容或 Qwen 模型权重**。Gemini 若可以访问外部网页，可尝试打开 manifest 中 `source_url` 的公开原站链接；网页平台是否支持直取 SRT 需要当场验证，不可假定它已经读到全文。
- 若网页无法读取第三方来源，请由有权使用资料的人员另外提供原始文件。上传网页端之前确认有关许可、平台条款和上传范围。
- 这些是未人工核对的 ASR 字幕：可能混入观众投稿、影视/游戏对白和嘉宾发言；**不得把他人第一人称内容写成主播经历**。
- 初选明前奶绿：NL04（情感互动）、NL05（观众来稿/故事回应）、NL08（日常聊天）。这只是先行对照，不代表其余 8 份应跳过。
- 每个主播建独立 RAG；角色卡可能变更，不应决定某位主播的语料是否收集。
- 本地批次须由项目原生 `prepare` → `export-batches` 生成，再由 `import-refined` → `validate --strict` 校验；**不能手写或凭空补签名**。
- Embedding 默认为本地 `Qwen3-Embedding-0.6B`，不需 SiliconFlow；模型权重不在 GitHub。

本目录不授予任何第三方字幕的复制、重新公开发布或商业利用权。