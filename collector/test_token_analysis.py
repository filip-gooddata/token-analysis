"""Tests for token_analysis.py. Run: python3 -I collector/test_token_analysis.py."""
import io
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_collect as tc  # noqa: E402
import token_analysis as ta  # noqa: E402


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        cl = os.path.join(self.tmp, 'claude', 'proj')
        tc.write_jsonl(os.path.join(cl, 's1.jsonl'), [
            tc.claude_user('prompt ' + tc.SECRET, 2), tc.claude_asst('m1', 2, 3, 0, 2000, 50)])
        tc.write_jsonl(os.path.join(self.tmp, 'codex', '2026', 'rollout.jsonl'), tc.codex_lines(26, role='drone'))
        self.out_dir = os.path.join(self.tmp, 'out')
        self.argv = ['--out-dir', self.out_dir, '--end', tc.END, '--jobs', '1',
                     '--claude-root', os.path.join(self.tmp, 'claude'),
                     '--codex-root', os.path.join(self.tmp, 'codex')]


class Run(Fixture):
    def run_main(self, *extra):
        err, out = io.StringIO(), io.StringIO()
        with mock.patch.object(ta.webbrowser, 'open') as wb, \
                mock.patch('sys.stderr', err), mock.patch('sys.stdout', out):
            rc = ta.main(self.argv + list(extra))
        return rc, wb, err.getvalue(), out.getvalue()

    def test_collects_reports_progress_and_opens_browser(self):
        rc, wb, err, out = self.run_main()
        self.assertEqual(rc, 0)
        html = os.path.join(self.out_dir, 'token-drilldown.html')
        self.assertTrue(os.path.exists(html))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, 'token-data.json')))
        wb.assert_called_once_with('file://' + html)
        for label in ta.PHASES.values():
            self.assertIn(label, err)
        self.assertIn('Dashboard: ' + html, out)
        with open(html) as fh:
            self.assertNotIn(tc.SECRET, fh.read())

    def test_no_browser(self):
        rc, wb, _err, _out = self.run_main('--no-browser')
        self.assertEqual(rc, 0)
        wb.assert_not_called()

    def test_collect_failure_skips_browser(self):
        with mock.patch.object(ta.collect, 'main', return_value=2):
            rc, wb, _err, out = self.run_main()
        self.assertEqual(rc, 2)
        wb.assert_not_called()
        self.assertNotIn('Dashboard', out)

    def test_passes_unknown_options_to_collect(self):
        with mock.patch.object(ta.collect, 'main', return_value=2) as cm:
            self.run_main('--lookback', '7')
        argv, progress = cm.call_args[0]
        self.assertEqual(argv[argv.index('--lookback') + 1], '7')
        self.assertEqual(argv[argv.index('--template') + 1], os.path.join(ta.resource_dir(), 'template.html'))
        self.assertIsInstance(progress, ta.Progress)

    def test_relative_out_dir_opens_absolute_uri(self):
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(self.tmp)
        self.argv[1] = 'rel'
        rc, wb, _err, _out = self.run_main()
        self.assertEqual(rc, 0)
        wb.assert_called_once_with('file://' + os.path.join(os.path.realpath(self.tmp), 'rel', 'token-drilldown.html'))

    def test_rejects_collect_destination_options(self):
        for opt in ('--html', '--out=x.json', '--templ', '--out'):
            with mock.patch.object(ta.collect, 'main') as cm, self.assertRaises(SystemExit) as cx:
                self.run_main(opt, 'x')
            self.assertEqual(cx.exception.code, 2)
            cm.assert_not_called()

    def test_exception_in_collect_ends_progress_line(self):
        def boom(_argv, progress):
            progress('parse', 1, 3)
            raise RuntimeError('boom')
        err = FakeTTY()
        with mock.patch.object(ta.collect, 'main', boom), mock.patch.object(ta.webbrowser, 'open') as wb, \
                mock.patch('sys.stderr', err), self.assertRaises(RuntimeError):
            ta.main(self.argv)
        wb.assert_not_called()
        self.assertTrue(err.getvalue().endswith('1/3\n'))


class CollectProgress(Fixture):
    def test_run_reports_each_phase_to_completion(self):
        calls = []
        tc.c.main(['--out', os.path.join(self.tmp, 'o.json')] + self.argv[2:],
                  lambda *a: calls.append(a))
        phases = [p for p, _d, _t in calls]
        self.assertEqual(list(dict.fromkeys(phases)), ['parse', 'check', 'attribute'])
        for phase in ('parse', 'check'):
            self.assertIn((phase, 0, 2), calls)
            self.assertIn((phase, 2, 2), calls)
        self.assertEqual(calls[-1], ('attribute', 2, 2))


class Rendering(unittest.TestCase):
    def test_tty_draws_bar_and_ends_lines(self):
        s = FakeTTY()
        p = ta.Progress(s, clock=iter(range(100)).__next__)
        p('parse', 0, 4)
        p('parse', 2, 4)
        p('parse', 4, 4)
        p('check', 4, 4)
        p.finish()
        text = s.getvalue()
        self.assertIn('\rReading session logs   [' + '#' * 15 + '.' * 15 + '] 2/4', text)
        self.assertIn('#' * 30 + '] 4/4\n', text)
        self.assertTrue(text.endswith('\n'))

    def test_tty_throttles_but_always_draws_completion(self):
        s = FakeTTY()
        p = ta.Progress(s, clock=lambda: 5.0)
        p('parse', 0, 100)
        for i in range(1, 101):
            p('parse', i, 100)
        self.assertEqual(s.getvalue().count('\r'), 2)  # first draw and the final 100/100
        self.assertIn('100/100', s.getvalue())

    def test_pipe_prints_one_line_per_phase(self):
        s = io.StringIO()
        p = ta.Progress(s)
        for phase in ('parse', 'check'):
            for i in range(3):
                p(phase, i, 2)
        p.finish()
        self.assertEqual(s.getvalue(), 'Reading session logs (2)...\nCross-checking usage (2)...\n')

    def test_finished_bar_ends_its_line_once(self):
        s = FakeTTY()
        p = ta.Progress(s, clock=iter(range(100)).__next__)
        p('attribute', 0, 2)
        p('attribute', 2, 2)
        self.assertTrue(s.getvalue().endswith('2/2\n'))
        p.finish()
        self.assertEqual(s.getvalue().count('\n'), 1)

    def test_empty_phase(self):
        s = FakeTTY()
        p = ta.Progress(s)
        p('attribute', 0, 0)
        p.finish()
        self.assertIn('0/0\n', s.getvalue())


if __name__ == '__main__':
    unittest.main()
