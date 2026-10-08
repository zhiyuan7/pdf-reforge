# PDF 回炉所 · PDF Reforge

**旧 PDF 进炉，结构清楚的新书稿出炉。**

PDF 回炉所用 MinerU 的识别结果恢复书籍结构，通过 DeepSeek 做少量、简短的判断，再生成带目录、特殊块样式和内部跳转的 PDF。适用于有原始目录的中文数学教材及类似书籍。

仓库只保留当前生成流程的程序、配置与排版资源。原始书籍、识别结果、模型缓存、处理报告和生成的 PDF 均留在本地，不提交到 GitHub。

## 当前流程

```mermaid
flowchart LR
    A[原始 PDF 与 MinerU 结果] --> B[目录标题匹配]
    B --> C[正文清理与特殊块抽取]
    C --> D[本地编号引用映射]
    D --> E[LaTeX 排版与 PDF 检查]
```

| 阶段 | 处理内容 | 模型使用 |
| --- | --- | --- |
| 1. 标题匹配 | 读取原书目录图片，在识别文字中匹配标题、层级与位置 | DeepSeek 读取目录 |
| 2. 正文整理 | 删除边栏、清理段首标点、跨页配对、识别特殊块 | 只返回判断、关键词或块号 |
| 3. 引用映射 | 搜索正文中的类型＋编号，登记候选并生成可确定的跳转 | 本地规则，无模型调用 |
| 4. PDF 生成 | 直接读取保存的前三阶段结果，生成书稿、目录、书签与链接 | 本地排版，无模型调用 |

### 第二步的规则

1. 第一步匹配到的标题优先成为标题，即使 MinerU 原先将其识别为页眉或页脚。
2. 按 MinerU 类型删除 `header`、`footer`、`page_number`。图注、表注和脚注继续保留。
3. 只清理**每个文字块开头**不应出现的句号、逗号、后引号、后括号等；保留前引号、前括号，以及块内部的文字、公式和标点。
4. 将每页最后一个文字块与下一页第一个文字块配对，由 DeepSeek 返回 `{"m":1}`（合并）或 `{"m":0}`（不合并）。
5. 随机抽取六个有正文的标题范围，一次取得原文实际出现的类型词，例如 `{"k":["定理","公式"]}`，不输出编号和解释。默认种子固定，便于复现；不足六个时使用现有范围。
6. 全局查找以关键词开头的正文块，将候选块与随后两个块交给 DeepSeek，只返回 `{"y":0,"b":[]}` 或 `{"y":1,"b":[1,2]}`。
7. 若窗口内所有块都属于该对象，则在已有文字后追加 6、9、12……个块，累计窗口为 3、9、18、30……，直到确定范围或到达标题边界。

特殊块只包含**表述**，不包含证明、推导、解答或后续讨论。证明继续留在普通正文中。第二步不做 OCR 纠错，也不额外推断小标题。无法确定的判断会保留来源并记录问题。

### 名称从哪里来

名称由本地程序从对象声明处的原文提取，不让模型重新起名。例如：

```text
定理4（数列收敛的柯西准则）……
→ 类型：定理；编号：4；名称：数列收敛的柯西准则
```

支持中英文括号、引号、书名号、尖括号、破折号等形式，以及嵌套符号和跨块名称。只有明示且能确定的名称才进入记录；没有明确名称时留空。记录保留名称原文和来源范围，框标题显示原有类型、编号与名称，正文开头不重复名称。

引用映射只使用类型和编号。名称、例题编号和没有明确编号的语义指代不参与匹配；多个候选时保留原引用并记录候选。

## 安装

当前流程在 **Python 3.12 / Linux** 上验证。需要 XeLaTeX、latexmk 和中文字体。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

在 `.env` 中填写自己的 `DEEPSEEK_API_KEY`。在 `book_config.json` 中设置模型、接口地址、并发数和字体；目录读取使用图片输入，因此所配置的接口与模型需要支持图片及 JSON 输出。

已有完整 MinerU 结果时无需安装或运行 MinerU。需要重新识别时，额外安装：

```bash
pip install -r requirements-mineru.txt
```

Linux 用户可用以下脚本在项目的 `.tools/` 中安装排版工具；其他系统可自行安装 TeX Live 与 latexmk：

```bash
python install_book_tex.py
python main.py --check-local
```

默认正文字体为 `Noto Serif CJK SC`，标题字体为 `Noto Sans CJK SC`，需要在系统中安装。图注使用仓库附带的霞鹜文楷字体。`--check` 在本地检查基础上额外检查密钥是否存在，不发送模型请求。

## 准备书籍

将 PDF 放在本地输入目录，例如 `input_pdfs/book.pdf`。复制配置示例并填写书名和作者：

```bash
cp book_metadata.example.yaml input_pdfs/book.book.yaml
```

书名和作者读取 YAML，不自动猜测。`exclude_pages` 使用从 1 开始的原 PDF 物理页码；未知书籍不会默认删除封面或版权页。通常自动找到目录页，也可在 `book_config.json` 的 `toc_pages` 中指定物理页码。

完整 MinerU 结果目录需要包含 `manifest.json`、`middle_json.json` 与识别图片资源。当前适配 MinerU 4.0.10 的 `docvortex.middle` 2.0 格式；输入 PDF 的 SHA-256 必须与清单一致。

## 运行完整流程

复用已有 MinerU 结果，只运行标题匹配及后续步骤：

```bash
python main.py \
  --pdf input_pdfs/book.pdf \
  --mineru-dir output/book/mineru \
  --metadata input_pdfs/book.book.yaml \
  -o output/book/reforged
```

最终 PDF 位于 `output/book/reforged/pdf/latex/main.pdf`。如果不指定 `--mineru-dir`，程序会在输出目录中复用已有完整识别结果，或调用本地 MinerU 生成结果：

```bash
python main.py --pdf input_pdfs/book.pdf -o output/book
```

也可单独完成识别，再运行后续阶段：

```bash
python run_mineru.py input_pdfs/book.pdf -o output/book/mineru --pages all
```

MinerU 推理在本地运行；第二步的文字输入和目录页图片会提交给配置的 DeepSeek 接口。

## 分阶段运行

每个阶段使用独立的输出目录，后续阶段读取已保存结果：

```bash
# 第一步
python main.py --match-headings \
  --pdf input_pdfs/book.pdf --mineru-dir output/book/mineru \
  --metadata input_pdfs/book.book.yaml -o output/book/headings

# 第二步：完整处理所有标题范围
python main.py --process-sections \
  --pdf input_pdfs/book.pdf --mineru-dir output/book/mineru \
  --heading-report output/book/headings -o output/book/content

# 第三步
python main.py --map-references \
  --pdf input_pdfs/book.pdf --mineru-dir output/book/mineru \
  --heading-report output/book/headings \
  --processed-book-dir output/book/content -o output/book/references

# 第四步
python main.py --render-processed-book \
  --pdf input_pdfs/book.pdf --mineru-dir output/book/mineru \
  --metadata input_pdfs/book.book.yaml \
  --processed-book-dir output/book/content \
  --mapping-report output/book/references -o output/book/pdf
```

同一输入与提示会复用经过校验的响应缓存。可通过配置 `shared_cache_dirs` 复用其他本地缓存目录；命中的有效响应会保存到当前阶段缓存，使本次结果不再依赖旧目录。改变来源或标题结果时，请使用新的结果目录。

运行记录提供真实请求数、缓存命中数和 token 用量。配置中的费率只是费用估算参数，需要按实际服务更新；缓存重放不会产生新的模型费用。

## 项目结构

```text
main.py                    当前流程的唯一主入口
run_mineru.py              可选的本地识别入口
mineru_checkpoint.py       MinerU 推理缓存与完整性检查
pipeline_progress.py       MinerU 进度事件
install_book_tex.py        本地排版工具安装
book_config.json           通用运行配置
book_metadata.example.yaml 书籍配置示例
book_pipeline/
  application/             四阶段编排与命令行
  ingestion/               原文、公式与精确来源范围
  ai/                      DeepSeek 请求、缓存和用量记录
  structure/               标题、正文块、名称和引用
  publishing/              LaTeX、编译、页面与导航检查
book_templates/            ElegantBook 模板与书稿样式
models/                    排版所需字体及许可证
output/                    本地结果，Git 忽略
```

结果目录包含标题匹配报告、原文分块、删除记录、跨页判断、关键词采样、特殊块范围、原文名称来源、引用候选、LaTeX 源码、PDF 及检查报告。一级标题独占标题页，二级标题另起一页；特殊块使用对应样式，独立公式保留原生公式排版。

## 已完成的实际验证

以卓里奇《数学分析（第七版）第一卷》完成了整书处理，并在删除旧流程依赖后再次验证完整生成过程：214 个匹配标题、345 个特殊块、51 个原文明示名称，最终 PDF 为 684 页，1,442 个内部链接检查通过。此次验证复用了识别结果与有效缓存，没有重跑 MinerU，没有新增 DeepSeek 请求。

这些书籍数据、报告和 PDF 只保留在本地。仓库不包含旧流程、实验处理脚本或测试脚本。检查确认来源保留、页面和链接可用；原始 OCR 文字与公式没有重新纠错。

## 排版资源来源

- ElegantBook 模板保留上游声明与 LPPL 许可证信息，具体版本和校验值见 [模板来源记录](book_templates/PROVENANCE.md)。
- 霞鹜文楷字体附带 [OFL 许可证](models/font_candidates/LxgwWenkaiGB/OFL.txt)。
