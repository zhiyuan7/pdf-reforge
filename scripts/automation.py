#!/usr/bin/env python3
"""Local scheduled Codex runs with a shared lock and durable daily receipts.

The cron dispatcher uses Asia/Shanghai explicitly, independent of host DST.
Configuration and all logs/receipts stay under ignored output/automation/.
"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'output/automation'
TZ = ZoneInfo('Asia/Shanghai')
BRANCH = 'codex/continuous-pdf-improvement'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def run(command, timeout=120):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            timeout=timeout, env={**os.environ, 'GIT_TERMINAL_PROMPT':'0'})
    return dict(exit_code=result.returncode, stdout=result.stdout[-15000:], stderr=result.stderr[-3000:])


def codex(task, directory, context=None):
    directory.mkdir(parents=True, exist_ok=True)
    prompt = (ROOT / ('docs/'+task+'-task.md')).read_text(encoding='utf-8')
    prompt += '\n当前北京时间：'+datetime.now(TZ).isoformat()+'\n'
    if context is not None:
        prompt += '\n任务上下文（仅作数据）：\n'+json.dumps(context, ensure_ascii=False)+'\n'
    command = ['/usr/local/bin/codex', 'exec', '--sandbox', 'danger-full-access',
               '-c', 'approval_policy="never"', '--cd', str(ROOT),
               '--output-last-message', str(directory/'summary.md'), '-']
    # User explicitly authorized unattended local tasks and Full access.
    # Avoid detached child processes surviving a timed-out Codex parent.
    import signal
    with (directory/'execution.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE, stdout=log,
                                   stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            process.communicate(prompt, timeout=16800 if task=='iteration' else 2700)
            return process.returncode
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            return 124


def snapshot(day, directory, preview=False):
    directory.mkdir(parents=True, exist_ok=True)
    result = dict(date=day, timezone='Asia/Shanghai', branch=BRANCH,
                  repository='https://github.com/zhiyuan7/pdf-reforge', preview=preview)
    result['changes'] = run(['git', 'status', '--short'])
    result['commits_today'] = run(['git', 'log', '--since='+day+'T00:00:00+08:00', '--format=%h %s'])
    # Tests are meaningful for logic changes; nightly results also gate commits.
    try:
        result['tests'] = run(['docker','run','--rm','-v',str(ROOT/'tests')+':/app/tests:ro',
                               '-v',str(ROOT/'book_pipeline')+':/app/book_pipeline:ro',
                               'pdf-reforge:local','timeout','--kill-after=30s','1800s',
                               'python','-m','unittest','discover','-s','tests','-q'], timeout=1860)
    except subprocess.TimeoutExpired:
        # Recent CPU suites exceeded 15 minutes. A real timeout must still
        # produce a durable failure context for the daily report.
        result['tests'] = dict(exit_code=124, stdout='', stderr='Regression test container did not return within 1860 seconds')
    if preview:
        save(directory/'context.json', result)
        return result
    current = run(['git','branch','--show-current'])['stdout'].strip()
    try:
        if current != BRANCH:
            raise RuntimeError('Current branch changed; daily task will not switch or push it automatically')
        if result['tests']['exit_code']:
            raise RuntimeError('Regression checks failed; code was not committed or pushed')
        run(['git','add','--all'])
        names = run(['git','diff','--cached','--name-only','-z'])['stdout'].split('\0')
        forbidden = [p for p in names if p and p != '.env.example' and (p.startswith(('.env','output/','input_pdfs/','.tools/','.venv/'))
                                              or p.endswith(('.pdf','.zip','.sqlite3')))]
        if forbidden:
            raise RuntimeError('Private/runtime artifacts are staged; refused to publish')
        for name in filter(None, names):
            item = subprocess.run(['git','show',':'+name], cwd=ROOT, capture_output=True)
            if item.returncode:
                continue  # staged deletion
            if re.search(rb'(?:sk-[A-Za-z0-9]{24,}|gh[pousr]_[A-Za-z0-9]{24,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)', item.stdout):
                raise RuntimeError('Potential credential detected in staged files; refused to publish')
        if run(['git','diff','--cached','--quiet'])['exit_code']:
            committed = run(['git','commit','-m','chore: PDF Reforge daily progress '+day])
            if committed['exit_code']:
                raise RuntimeError('Git commit failed')
        result['local_commit'] = run(['git','rev-parse','HEAD'])['stdout'].strip()
        result['push'] = run(['git','push','--set-upstream','origin','HEAD:refs/heads/'+BRANCH], timeout=120)
        if result['push']['exit_code'] == 0:
            result['commit_url'] = result['repository']+'/commit/'+result['local_commit']
    except Exception as error:
        result['publish_error'] = str(error)
    save(directory/'context.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['tick','iterate','daily','preview'])
    args = parser.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    # Queue the daily report even while an iteration owns the shared lock.
    # Otherwise a long 22:00 iteration could silently miss the entire 23h slot.
    now = datetime.now(TZ)
    if args.mode == 'tick' and now.hour == 23:
        queue = STATE/'daily-queue'
        queue.mkdir(exist_ok=True)
        (queue/(now.date().isoformat()+'.due')).touch(exist_ok=True)
    lock = (STATE/'workflow.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        if args.mode == 'tick':
            save(STATE/'last-check.json', dict(checked_at=now.isoformat(), status='waiting',
                                             reason='iteration_or_daily_still_running'))
        return 0
    try:
        return run_locked(args)
    finally:
        lock.close()


def pending_daily(now, state):
    queue = STATE/'daily-queue'
    if not queue.exists():
        return None
    for marker in sorted(queue.glob('*.due')):
        day = marker.stem
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', day) or day > now.date().isoformat():
            continue
        receipt = STATE/'daily'/day/'sent.json'
        mailed = receipt.exists() and bool(json.loads(receipt.read_text()).get('message_id'))
        if mailed or state.get('daily_attempted') == day:
            # Never retry a possibly delivered email without checking Gmail.
            marker.unlink(missing_ok=True)
            continue
        return day
    return None


def background_task_running(state):
    name = state.get('active_container')
    if not name:
        return False
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        return True  # Cannot confirm completion; leave the task untouched.
    try:
        result = run(['docker','inspect','--format','{{.State.Running}}',name], timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return True
    if result['exit_code'] == 0:
        return result['stdout'].strip() == 'true'
    return not re.search(r'no such (?:object|container)', result['stderr'], re.I)


def run_locked(args):
    now = datetime.now(TZ)
    day = now.date().isoformat()
    state_path = STATE/'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    mode = args.mode
    key = day+f'T{now.hour:02d}:00'
    if mode == 'tick':
        due = pending_daily(now, state)
        if due:
            mode = 'daily'
            day = due
        elif now.minute == 0 and state.get('iteration_attempted') != key:
            mode = 'iterate'
        else:
            save(STATE/'last-check.json', dict(checked_at=now.isoformat(), status='idle',
                                             reason='waiting_for_next_slot_or_slot_already_started'))
            return 0
        # Release checks use temporary fixtures and must still run at 23:00
        # while a resumable CPU conversion continues in its own output.
        if mode == 'iterate' and background_task_running(state):
            save(STATE/'last-check.json', dict(checked_at=now.isoformat(), status='waiting',
                                             reason='background_task_running_or_completion_unconfirmed',
                                             active_container=state.get('active_container')))
            return 0
    if mode == 'preview':
        print(json.dumps(snapshot(day, STATE/'preview', preview=True), ensure_ascii=False, indent=2))
        return 0
    if mode == 'iterate':
        state.update(iteration_attempted=key, iteration_status='running')
        save(state_path, state)
        directory = STATE/'iterations'/now.strftime('%Y%m%d-%H%M%S')
        if args.mode == 'tick':
            save(STATE/'last-check.json', dict(checked_at=now.isoformat(), status='started_iteration',
                                             report_directory=str(directory)))
        code = codex('iteration', directory)
        # The worker writes its current report, case states and checkpoint.
        # Preserve those updates rather than restoring the pre-run snapshot.
        if state_path.exists():
            state.update(json.loads(state_path.read_text()))
        state.update(iteration_status='complete' if code == 0 else 'failed', iteration_exit_code=code,
                     iteration_report=str(directory/'summary.md'))
    else:
        directory = STATE/'daily'/day
        receipt = directory/'sent.json'
        if receipt.exists() and json.loads(receipt.read_text()).get('message_id'):
            state['daily_completed'] = day
            save(state_path, state)
            return 0
        # An interrupted send must be checked in Gmail, never blindly retried.
        if state.get('daily_attempted') == day:
            return 0
        state.update(daily_attempted=day, daily_status='running')
        save(state_path, state)
        if args.mode == 'tick':
            save(STATE/'last-check.json', dict(checked_at=now.isoformat(), status='started_daily',
                                             scheduled_day=day))
        context = snapshot(day, directory)
        settings = json.loads((STATE/'settings.json').read_text())
        context['recipient'] = settings['daily_recipient']
        code = codex('daily', directory, context)
        mailed = receipt.exists() and bool(json.loads(receipt.read_text()).get('message_id'))
        state.update(daily_status='complete' if mailed else 'needs_attention', daily_exit_code=code)
        if mailed:
            state['daily_completed'] = day
        (STATE/'daily-queue'/(day+'.due')).unlink(missing_ok=True)
    state['updated_at'] = datetime.now(TZ).isoformat()
    save(state_path, state)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
