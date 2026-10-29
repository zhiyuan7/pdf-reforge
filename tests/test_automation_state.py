"""Returning from a scheduled worker preserves its resumable checkpoint."""
import json
from datetime import datetime
import fcntl
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import automation


class AutomationPublishTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def publish(self, name, content=b'EXAMPLE_SETTING=placeholder\n'):
        # Exercise real staging, content inspection and commit in an isolated repo.
        repository = self.root / str(len(list(self.root.iterdir())))
        repository.mkdir()
        for args in (['init', '-b', automation.BRANCH],
                     ['config', 'user.name', 'Publish Test'],
                     ['config', 'user.email', 'publish@example.test']):
            subprocess.run(['git', *args], cwd=repository, check=True, capture_output=True)
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        real_run = automation.run

        def local_run(command, **kwargs):
            if command[0] == 'docker' or command[:2] == ['git', 'push']:
                return dict(exit_code=0, stdout='', stderr='')
            return real_run(command, **kwargs)

        with patch.object(automation, 'ROOT', repository), \
             patch.object(automation, 'run', side_effect=local_run) as runner:
            result = automation.snapshot('2026-10-27', repository / 'receipt')
        commands = [call.args[0] for call in runner.call_args_list]
        return repository, result, commands

    def test_exact_env_example_can_be_committed_and_reach_push(self):
        repository, result, commands = self.publish('.env.example')
        self.assertNotIn('publish_error', result)
        self.assertEqual(result['push']['exit_code'], 0)
        self.assertTrue(any(command[:2] == ['git', 'push'] for command in commands))
        committed = subprocess.run(['git', 'show', 'HEAD:.env.example'], cwd=repository,
                                   check=True, capture_output=True)
        self.assertEqual(committed.stdout, b'EXAMPLE_SETTING=placeholder\n')

    def test_private_and_runtime_files_still_stop_before_commit(self):
        for name in ('.env', '.env.local', '.env.production', '.env.example.local',
                     '.env.example.bak', 'output/result.txt', 'input_pdfs/source.txt',
                     '.tools/tool', '.venv/config', 'book.pdf', 'archive.zip', 'budget.sqlite3'):
            with self.subTest(name=name):
                _, result, commands = self.publish(name)
                self.assertIn('Private/runtime artifacts', result['publish_error'])
                self.assertFalse(any(command[:2] in (['git', 'commit'], ['git', 'push'])
                                     for command in commands))

    def test_env_example_still_undergoes_credential_inspection(self):
        for credential in (b'sk-' + b'A' * 32, b'ghp_' + b'B' * 32,
                           b'-----BEGIN ' + b'PRIVATE KEY-----'):
            with self.subTest(prefix=credential[:8]):
                _, result, commands = self.publish('.env.example', credential)
                self.assertIn('Potential credential', result['publish_error'])
                self.assertFalse(any(command[:2] in (['git', 'commit'], ['git', 'push'])
                                     for command in commands))

    def test_test_timeout_saves_failure_context_without_publishing(self):
        commands = []

        def timed_run(command, **kwargs):
            commands.append(command)
            if command[0] == 'docker':
                raise subprocess.TimeoutExpired(command, kwargs['timeout'])
            return dict(exit_code=0, stdout=automation.BRANCH, stderr='')

        receipt = self.root/'receipt'
        with patch.object(automation, 'run', side_effect=timed_run):
            result = automation.snapshot('2026-10-29', receipt)
        self.assertEqual(result['tests']['exit_code'], 124)
        self.assertIn('Regression checks failed', result['publish_error'])
        self.assertEqual(json.loads((receipt/'context.json').read_text()), result)
        self.assertFalse(any(c[:2] in (['git', 'add'], ['git', 'commit'], ['git', 'push'])
                             for c in commands))


class AutomationStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patcher = patch.object(automation, 'STATE', self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        clock_patcher = patch.object(automation, 'datetime')
        self.clock = clock_patcher.start()
        self.addCleanup(clock_patcher.stop)
        self.at(25, 10, 0)

    def at(self, day, hour, minute):
        self.clock.now.return_value = datetime(2026, 10, day, hour, minute, tzinfo=automation.TZ)

    def tick(self):
        with patch('sys.argv', ['automation', 'tick']):
            return automation.main()

    def state(self):
        return json.loads((self.root/'state.json').read_text())

    def test_next_iteration_starts_once_per_hour_after_completion(self):
        automation.save(self.root/'state.json', dict(iteration_attempted='2026-10-25T06', iteration_status='complete'))
        with patch.object(automation, 'codex', return_value=0) as worker:
            self.assertEqual(self.tick(), 0)
            self.assertEqual(self.state()['iteration_attempted'], '2026-10-25T10:00')
            self.assertEqual(self.tick(), 0)
            self.assertEqual(worker.call_count, 1)
            self.at(25, 10, 30)
            self.assertEqual(self.tick(), 0)
            self.assertEqual(worker.call_count, 1)
            self.at(25, 11, 0)
            self.assertEqual(self.tick(), 0)
            self.assertEqual(worker.call_count, 2)
            self.assertEqual(self.state()['iteration_attempted'], '2026-10-25T11:00')

    def test_busy_worker_is_left_running_without_starting_another(self):
        with (self.root/'workflow.lock').open('a') as lock, patch.object(automation, 'codex') as worker:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(self.tick(), 0)
            worker.assert_not_called()
            self.assertEqual(json.loads((self.root/'last-check.json').read_text())['status'], 'waiting')

    def test_running_background_conversion_is_waited_for(self):
        automation.save(self.root/'state.json', dict(active_container='pdf-reforge-benchmark'))
        with patch.object(automation, 'run', return_value=dict(exit_code=0, stdout='true', stderr='')), \
             patch.object(automation, 'codex') as worker:
            self.assertEqual(self.tick(), 0)
            worker.assert_not_called()
            self.assertNotIn('iteration_attempted', self.state())
        self.at(25, 11, 0)
        with patch.object(automation, 'run', return_value=dict(exit_code=0, stdout='false', stderr='')), \
             patch.object(automation, 'codex', return_value=0) as worker:
            self.assertEqual(self.tick(), 0)
            worker.assert_called_once()

    def test_unknown_docker_state_waits_instead_of_starting_duplicate_work(self):
        automation.save(self.root/'state.json', dict(active_container='pdf-reforge-benchmark'))
        with patch.object(automation, 'run', return_value=dict(exit_code=1, stdout='', stderr='Cannot connect to Docker daemon')), \
             patch.object(automation, 'codex') as worker:
            self.assertEqual(self.tick(), 0)
            worker.assert_not_called()

    def test_daily_release_runs_while_background_ocr_is_preserved(self):
        self.at(29, 23, 0)
        automation.save(self.root/'state.json', dict(active_container='pdf-reforge-cpu-ocr'))
        automation.save(self.root/'settings.json', dict(daily_recipient='me@example.test'))
        with patch.object(automation, 'background_task_running', return_value=True) as background, \
             patch.object(automation, 'snapshot', return_value=dict(date='2026-10-29')) as release, \
             patch.object(automation, 'codex', return_value=0) as worker:
            self.assertEqual(self.tick(), 0)
            release.assert_called_once()
            self.assertEqual(worker.call_args.args[0], 'daily')
            background.assert_not_called()
        self.assertEqual(self.state()['active_container'], 'pdf-reforge-cpu-ocr')
        self.assertEqual(self.state()['daily_attempted'], '2026-10-29')

    def test_checks_outside_hour_do_not_start_work(self):
        with patch.object(automation, 'codex') as worker:
            for minute in (1, 15, 30, 59):
                self.at(25, 10, minute)
                self.assertEqual(self.tick(), 0)
            worker.assert_not_called()

    def test_daily_queued_during_long_iteration_survives_midnight(self):
        self.at(25, 23, 0)
        with (self.root/'workflow.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(self.tick(), 0)
        self.assertTrue((self.root/'daily-queue/2026-10-25.due').exists())
        self.at(26, 0, 0)
        automation.save(self.root/'settings.json', dict(daily_recipient='me@example.test'))
        def worker(task, directory, context):
            self.assertEqual(task, 'daily')
            self.assertEqual(context['date'], '2026-10-25')
            automation.save(directory/'sent.json', dict(message_id='test-receipt'))
            return 0
        with patch.object(automation, 'snapshot', side_effect=lambda day,*args: dict(date=day)), \
             patch.object(automation, 'codex', side_effect=worker):
            self.assertEqual(self.tick(), 0)
        self.assertEqual(self.state()['daily_completed'], '2026-10-25')
        self.assertFalse((self.root/'daily-queue/2026-10-25.due').exists())

    def test_unknown_mail_delivery_does_not_trigger_a_second_send(self):
        self.at(25, 23, 30)
        automation.save(self.root/'state.json', dict(daily_attempted='2026-10-25', daily_status='needs_attention'))
        with patch.object(automation, 'codex', return_value=0) as worker:
            self.assertEqual(self.tick(), 0)
            worker.assert_not_called()
            self.at(26, 0, 0)
            self.assertEqual(self.tick(), 0)
            self.assertEqual(worker.call_args.args[0], 'iteration')

    def test_worker_report_and_quality_states_survive_dispatcher_completion(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            automation.save(root/'state.json', dict(active_report='old.json', current_batch='math'))
            def worker(*args):
                state = json.loads((root/'state.json').read_text())
                state.update(active_report='new.json', quality_passes=0, unresolved=['captions'])
                automation.save(root/'state.json', state)
                return 0
            with patch.object(automation, 'STATE', root), patch.object(automation, 'codex', side_effect=worker), \
                 patch('sys.argv', ['automation', 'iterate']):
                self.assertEqual(automation.main(), 0)
            saved = json.loads((root/'state.json').read_text())
            self.assertEqual(saved['active_report'], 'new.json')
            self.assertEqual(saved['unresolved'], ['captions'])
            self.assertEqual(saved['iteration_status'], 'complete')
