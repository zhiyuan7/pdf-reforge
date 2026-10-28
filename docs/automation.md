# 本机持续迭代与日报

使用已登录的 Codex CLI 和系统 cron，主机需保持开机联网。这些是本机任务，不会出现在 Codex 应用的定时任务列表中。

- 每小时（北京时间每小时00分）检查一次：上一轮智能体或其后台转换任务仍在运行时等待；已结束时按 `iteration-task.md` 接续下一轮，修复、测试并复核。单轮最长4小时40分，共用锁防止重复运行。
- 北京时间 23:00：测试通过后提交并推送 `codex/continuous-pdf-improvement`，随后通过已连接 Gmail 发送真实进展。测试或推送失败也发报告，并注明失败。
- 时间通过 Python `Asia/Shanghai` 解释，不受主机欧洲夏令时影响。cron每小时做轻量检查，运行中的任务不会因检查而中断或重复启动。每天23:00的日报若遇到上一轮未结束，会排队，在之后的检查中优先执行；即使跨过午夜也保留，避免漏报。发信失败或结果不明时保留状态，不盲目重发。
- 工作流状态、执行日志、日报上下文和发信回执位于忽略提交的 `output/automation/`。收件人保存在该目录的 `settings.json`，不进入仓库。
- 最近一次每小时检查的时间和“等待/启动”结果保存在 `output/automation/last-check.json`；当前正在执行的进度与报告入口见 `state.json`。
- 记录以节省token为原则：每小时检查只负责等待或续接，未变化时保持安静。worker以修复和必要验证为主，按需读相关状态/证据；每轮仅保留最多5条、约300字的 `summary.md` 和必要续接状态，不要求长篇 `report.md`、成套手写审计或重复截图。未变样本复用原证据，历史文件保留。每日邮件简短汇总，约600字以内。
- 共用预算账本位于 `output/docker/budget.sqlite3`；账单未知的调用保留最坏预留，不能删除账本规避预算。完整转换与节选回归必须分别报告。
- 只验证排版时，使用已有 source、content、references 调用 `render_saved_book`，避免重新进入提取。要求零 API 的验证使用 `docker run --network none`，并让禁用连接的守卫通过 `sitecustomize.py` 覆盖父进程和 spawn 子进程；仅在主脚本替换 `DeepSeekClient._connect` 不能保证子进程离线。中断且未返回 usage 的请求保留账本预留。
- 保存结果的逐字校勘使用输出目录的 `text_corrections.json`；字形复核可用 `source_style_overrides.json`（`source_sha256`、原图复核 `evidence`、含 `atom/quote/occurrence/italic/bold` 的 `runs`）。两者均绑定当前编辑索引，不改原OCR；字形仅可引用现有文字，不能与校勘区间重叠，不支持将中文楷体记为斜体。复核文件SHA随输入证据保存；换提取签名后旧复核不能直接套用。
- 同一字形复核文件可加 `table_math`：每项指定表格 `atom`、从0开始的 `row/column/occurrence` 与原有 `quote`，仅把该格的原文字节包进数学标记。不接受替换TeX，不重复标注已有数学区，不更改原表HTML；原图确认的变量字形和希腊字母由数学字体排印。
- 表格文字强调用同一文件的 `table_styles`，沿用上述单元格定位，另给布尔 `italic/bold`；仅包裹既有文字，与数学标注不得重叠。绑定原图的正文审校在精确引用范围内覆盖自动字形，其余自动范围保留。
- 原OCR把长图注误作正文时，可加 `caption_continuations`（`asset/group/quote`）：必须是同页紧邻图组、紧随CPU独立定位的图下标签，完整引用现有文字/行内数学组，并避开语义边界。渲染保留源锚点，以两端对齐图注段落随图浮动，不改原OCR或据此声称自动归属正确。
- 本地OCR在检测器输出边界将列表四边形规范化为数组，兼容裁切与结果序列化；已有数组保持原对象和坐标。兼容策略进入OCR观察/图注缓存及公式文字锚点签名，改变后在新运行目录重新验证，旧观察缓存不得冒充新签名结果。
- 扫描公式二次识别把上/下取整误作方括号时，仅在原CPU显示公式与不含已有转录的独立图片复读整式一致、归属唯一时恢复成对取整符号；其他字符、数字、运算符、复合分组、编号不变。原CPU含取整命令的横带签名包含此合并策略；即使局部审校漏报，也会按需请求盲读，不凭数学含义推断括号。
- 节选采用连续的节标题分页，仍保留章标题页和显式锚点分页；整书沿用原分页策略。公式前的段末同样启用孤行保护，嵌套字形切换保留原词间空格。
- 扫描正文漏掉脚注号时，`footnote_overrides.json` 可用 `missing_reference` 记录原子、`offset`、唯一的 `before/after` 原文和原图号标 `bbox`，并绑定编辑索引 `source_sha256` 与原图复核 `evidence`。仅限同页已有编号且未匹配的脚注；渲染添加号标，不改原始OCR，不能据此声称自动识别已修复。普通公理/步骤数字不再直接当作脚注号；原生PDF上标数字须有字形上标标志。
- 基准只追加 `-v /root/pdf-reforge/output/benchmark:/app/output/benchmark` 挂载；不要把宿主机整个 `output` 覆盖到 `/app/output`，那会遮住 Compose 指定的共用账本。其他证据目录按需单独只读挂载。付费前核对容器 `/app/output/budget.sqlite3` 与宿主机指定账本的条数和金额一致。

管理命令：

```bash
cat /etc/cron.d/pdf-reforge
python3 scripts/automation.py preview   # 只检查测试和日报上下文，不提交/推送/发信
python3 scripts/automation.py iterate   # 立即接续一次改进
python3 scripts/automation.py daily     # 立即执行当天提交/推送/日报，按回执防重
```

停止定时执行时移除 `/etc/cron.d/pdf-reforge` 中本项目一行，或停用该 cron 文件；正在执行的任务需要单独停止。外部账户授权由 GitHub CLI 和 Gmail 插件各自管理，和本机 Full access 独立。

2026-10-28更新：提交前测试所用CPU镜像包含Git，避免临时仓库测试因缺依赖而阻断推送。次批5部新作品的下载与抽样版面检查登记于 `benchmarks/corpus-next.json`，当前扫描样本收尾后立即开始。已通过样本的完整原件和冗余旧产物按用户要求删除，仅保留来源、SHA、预算账本、小节选及当前审校成品；历史报告中的旧路径可能已清理，不据此重新下载。
