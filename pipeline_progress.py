"""Dependency-free MinerU progress events for opt-in callers."""
import json
import os
import re

PREFIX = "@@PDF_PROGRESS@@"


def emit(phase, done=0, total=None, unit="页", detail="", **context):
    if os.environ.get("PDF_PIPELINE_PROGRESS") == "1":
        print(PREFIX + json.dumps(dict(phase=phase, done=done, total=total, unit=unit, detail=detail, **context),
                                 ensure_ascii=False), flush=True)


class MinerUProgress:
    """Read SDK progress without confusing per-window blocks with document pages."""
    WINDOW = re.compile(r'Hybrid processing window (\d+)/(\d+): pages (\d+)-(\d+)/(\d+)')
    BAR = re.compile(r'(Layout Predict|VLM Predict|OCR-det|OCR-rec|MFR Predict):\s*[\d.]+%\|[^|]*\|\s*(\d+)\s*/\s*(\d+)')
    PHASES = {'Layout Predict': ('版面分析', '页'), 'VLM Predict': ('内容块识别', '块'),
              'OCR-det': ('文字检测', '块'), 'OCR-rec': ('文字识别', '条'),
              'MFR Predict': ('公式识别', '式')}
    COMPLETE_PHASES = {'持久保存 MinerU 推理', '复用已完成的 MinerU 推理', '保存 MinerU 结果',
                       '逐页保存 MinerU 结果', '复用 MinerU 页面缓存'}

    def __init__(self, source_total):
        self.source_total = source_total
        self.completed_pages = 0
        self.window = None
        self.pagewise = False

    def event(self, detail, done=None, total=None, unit=''):
        return dict(detail=detail, done=done, total=total, unit=unit,
                    completed_pages=self.completed_pages, source_total=self.source_total,
                    window=self.window)

    def consume(self, line):
        if PREFIX in line:
            try:
                value, _ = json.JSONDecoder().raw_decode(line.split(PREFIX, 1)[1].lstrip())
                phase, done, total = value['phase'], value['done'], value['total']
                if not isinstance(phase, str) or type(done) is not int or done < 0:
                    return None
                if total is not None and (type(total) is not int or total < 1 or done > total):
                    return None
                if phase in self.COMPLETE_PHASES:
                    self.completed_pages = min(done, self.source_total)
                    self.window = None
                    return self.event(phase, done, total, value.get('unit', '页'))
                if phase == '识别单页':
                    page = value.get('page_number')
                    if type(page) is not int or not 1 <= page <= self.source_total or total is None:
                        return None
                    self.pagewise = True
                    self.completed_pages = done
                    self.window = dict(index=done+1,total=total,start=page,end=page)
                    return self.event(f'识别第 {page} 页',done,total,'页')
                if phase == 'MinerU 加载与识别':
                    return self.event('加载模型与识别')
                return self.event(phase, done, total, value.get('unit', ''))
            except (ValueError, KeyError, TypeError):
                return None
        window = self.WINDOW.search(line)
        if window:
            if self.pagewise:
                return None
            index, windows, start, end, total = map(int, window.groups())
            if not 1 <= index <= windows or not 1 <= start <= end <= total:
                return None
            self.window = dict(index=index, total=windows, start=start, end=end)
            self.source_total = total
            # MinerU enters the next window only after finishing the preceding one.
            self.completed_pages = start - 1
            return self.event('准备页面')
        bar = self.BAR.search(line)
        if bar:
            name, raw_done, raw_total = bar.groups()
            done, total = int(raw_done), int(raw_total)
            if not 0 <= done <= total or total < 1:
                return None
            detail, unit = self.PHASES[name]
            return self.event(detail, done, total, unit)
        return None
