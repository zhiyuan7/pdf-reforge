# 当前数学与 AI 基准

选书范围：只重构现代数学或 AI 教材、专著，优先现代横排、现行数学记号；不再选择古籍或历史数学文献。此范围按用户2026-10-27要求更新，后续迭代同样遵守。

当前正式语料使用 `corpus-v2-real-scans.json`（沿用文件名）：6份原始PDF、4部独立作品、8个节选测试件，覆盖线性代数与深度学习。中英文译本与rasterized变体共用作品身份；预算身份使用各原始完整PDF的SHA，不使用节选SHA。当前8个节选已有质量通过证据；样本缩减不表示识别能力提升，也不表示整书通过。

```bash
docker compose run --rm -v /root/pdf-reforge/output/benchmark:/app/output/benchmark pdf-reforge python scripts/benchmark.py --manifest benchmarks/corpus-v2-real-scans.json --run-id 新版本名
```

`native`是原生文字PDF的原页节选；`rasterized`是程序生成的受控栅格化测试；`scan`是原始真实扫描PDF的直接页拷贝。真实扫描可以带旧OCR层，仍按原图重新OCR。书名、版号、语言和变体不能增加独立作品数。正式语料目前尚无现代真实扫描样本通过；双栏连续正文、表格/代码与长书仍需补覆盖。仅有正文截图和可搜索页脚不算产物成功，编译成功不表示质量通过，节选通过不能外推整本书。

《微積溯源》已按用户要求移除，其原件、命名节选和对应生成目录已删除；1921年《A Treatise on Probability》退出当前任务，历史结果留档。旧报告仅记录当时范围，不作为当前样本表，也不能触发重新下载或继续修复。预算账本和未知用量预留完整保留。

`corpus-supplements.json` 单列现代扫描补缺，不改变上述正式样本计数。复旦公开的《数学分析》第三版上册11页扫描材料中，原第10–11页，以及 Wasserman《All of Statistics》446页原件中的第21–22页，均已生成可搜索PDF和LaTeX，并在绑定原图的人工校勘与字形修正后通过节选复核。原始OCR尚有错误，不能据此宣称全自动或整书通过。复旦原件无文字层；Wasserman原件带不准确的旧OCR，其去层派生对照仍属于同一作品。

Mitchell《Machine Learning》（1997）来自作者明确提供的完整421页免费PDF；原第70–71页作为当前批次补充，覆盖密集公式、并排决策树插图、长图注和表格。两页已转换为三页可搜索PDF与LaTeX；标题边界、顺序、页码与短表断页修复后，通过绑定原图的字形及长图注归属审校，完成节选复核。原件为带旧OCR的真实扫描，插图并排不算双栏连续正文。以上扫描原件均保留在本地，未发现开放再分发许可。来源、SHA、范围、状态与布局特征见补充manifest；最新证据入口见 `output/automation/state.json`，Mitchell出处与页图见 `output/automation/iterations/20261028-180002/mitchell-provenance.json`。

《Parallel and Distributed Computation: Numerical Methods》（Bertsekas、Tsitsiklis，1989；1997/2015再版）取得[作者MIT公开完整原件](https://web.mit.edu/jnt/www/home.html)，共832页：书籍部分PDF1–734为原生无文字层扫描，735–829为有文字层的解答手册，830为空白，831–832为扫描页。仅原30–31页（印刷12–13页）完成CPU预检与三页LaTeX/可搜索PDF产物；修复二次识别丢失上取整后，小标题、命题/证明字形及末页留白仍未通过质量复核。作者公开152页短览作为同作品额外原件保留，不增加作品或样本通过计数，也不替代完整原件SHA预算身份；出处、文字层及页图见 `output/automation/iterations/20261028-220001/pdc-provenance.json`。正文为单栏，连续双栏正文仍待补。
