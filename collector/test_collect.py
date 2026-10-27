"""Tests for collect.py on synthetic fixtures. Run: python3 -I collector/test_collect.py."""
import datetime
import json
import os
import shutil
import sys
import tempfile
import unittest
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect as c  # noqa: E402

END = '2026-10-20T12:00:00+00:00'
END_TS = datetime.datetime.fromisoformat(END).timestamp()
SECRET = 'FIXTURE_SECRET_TEXT'
ROLE_TAG = '<t3-subagent-profile role="%s">profile body</t3-subagent-profile>'

def assert_label_or_unparsed(t, got, want, msg=None):
    """r18 labels a Codex exec script only through an allowlist grammar (README, "Exec scripts"). A script r17
    labelled may fall outside it and be Unparsed: a missed label. Any other label or result is still wrong."""
    if got not in (c.CMD_UNPARSED, (c.CMD_UNPARSED,)):
        t.assertEqual(got, want, msg)


def exec_attribution(js, cwd):
    """The (category, path) the exec script js attributes, or Unparsed."""
    res = c.result_label_for(None, 'exec', js, cwd)
    return c.CMD_UNPARSED if res == (c.CMD_UNPARSED,) else res[0][1:]


def ts(hours_before):
    return (datetime.datetime.fromtimestamp(END_TS - hours_before * 3600, datetime.timezone.utc)
            .strftime('%Y-%m-%dT%H:%M:%S.000Z'))


def claude_user(text, h):
    return {'type': 'user', 'timestamp': ts(h), 'message': {'role': 'user', 'content': text}}


def claude_asst(mid, h, inp, cr, cw, out, text='ok', model='claude-opus-5-5', cw1=0):
    return {'type': 'assistant', 'timestamp': ts(h), 'requestId': 'req-' + mid,
            'message': {'id': mid, 'model': model, 'content': [{'type': 'text', 'text': text}],
                        'usage': {'input_tokens': inp, 'cache_read_input_tokens': cr,
                                  'cache_creation_input_tokens': cw, 'output_tokens': out,
                                  'cache_creation': {'ephemeral_1h_input_tokens': cw1}}}}


def codex_lines(h, role=None):
    users = [{'type': 'input_text', 'text': 'first message ' + SECRET}]
    out = [
        {'timestamp': ts(h), 'type': 'session_meta', 'payload': {'base_instructions': {'text': 'B' * 700}}},
        {'timestamp': ts(h), 'type': 'turn_context', 'payload': {'model': 'gpt-6-luna'}},
        {'timestamp': ts(h), 'type': 'response_item', 'payload': {'type': 'message', 'role': 'user', 'content': users}},
    ]
    if role:
        out.append({'timestamp': ts(h), 'type': 'response_item', 'payload': {
            'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': ROLE_TAG % role}]}})
    out += [
        {'timestamp': ts(h), 'type': 'response_item', 'payload': {'type': 'reasoning', 'summary': []}},
        {'timestamp': ts(h), 'type': 'response_item', 'payload': {
            'type': 'function_call', 'name': 'exec_command', 'call_id': 'c1', 'arguments': '{"cmd":"x"}'}},
        {'timestamp': ts(h), 'type': 'token_usage_record', 'payload': {
            'response_id': 'resp1', 'usage': {'input_tokens': 500, 'cached_input_tokens': 200,
                                              'output_tokens': 60, 'reasoning_output_tokens': 40}}},
        # duplicate record for the same response must be ignored
        {'timestamp': ts(h), 'type': 'token_usage_record', 'payload': {
            'response_id': 'resp1', 'usage': {'input_tokens': 500, 'cached_input_tokens': 200,
                                              'output_tokens': 60, 'reasoning_output_tokens': 40}}},
        {'timestamp': ts(h), 'type': 'response_item', 'payload': {
            'type': 'function_call_output', 'call_id': 'c1', 'output': 'R' * 350}},
        {'timestamp': ts(h), 'type': 'token_usage_record', 'payload': {
            'response_id': 'resp2', 'usage': {'input_tokens': 700, 'cached_input_tokens': 500, 'output_tokens': 20}}},
    ]
    return out


def write_jsonl(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        for o in lines:
            fh.write((o if isinstance(o, str) else json.dumps(o)) + '\n')
    os.utime(path, (END_TS, END_TS))


def ratios(r=1.0):
    return {'claude': {'chars_per_token': r}, 'codex': {'chars_per_token': r}}


def replay(events, windows=None, role='top', src='claude'):
    acc, recon, counters, basis = {}, {}, defaultdict(int), defaultdict(set)
    windows = windows or [(0, 1e12)]
    c.attribute({'src': src, 'role': role, 'events': events}, ratios(), windows, acc, recon, counters, basis)
    return acc, recon, counters


def call(C, out=0, cr=0, outs=None, t=1000.0, model='opus-5.5', reas=None, src='claude'):
    return ('C', t, model, C, cr, 0, 0, out, reas, outs or [], src)


TYPED = (c.USER, 'Typed prompts')


class Apportion(unittest.TestCase):
    def test_exact_integer_total(self):
        for total in (0, 1, 7, 1000, 12345):
            parts = c.apportion([0.3, 1.7, 2.0, 0.0001], total)
            self.assertEqual(sum(parts), total)
            self.assertTrue(all(p >= 0 for p in parts))

    def test_zero_weights_go_to_last(self):
        self.assertEqual(c.apportion([0, 0], 5), [0, 5])


class Attribution(unittest.TestCase):
    def test_rereads_counted_each_call_and_reset_at_compaction(self):
        acc, recon, _ = replay([
            ('B', TYPED, 100, 0, c.B_EST),
            call(100, out=10, outs=[(c.P_TEXT, 5, 0)]),
            call(110),
            ('R',),
            call(50),
        ])
        get = lambda p: acc.get((p, 0, 'opus-5.5', 'top'), [0, 0, 0])
        self.assertEqual(get(TYPED)[0], 200)        # billed on both calls before compaction
        self.assertAlmostEqual(get(TYPED)[1], 100)  # unique only once
        self.assertEqual(get(c.P_TEXT)[0], 20)      # output once + re-read once
        self.assertEqual(get(c.P_TEXT)[1], 10)
        self.assertEqual(get(c.UNATTRIB)[0], 50)    # after reset nothing is attributed
        self.assertEqual(recon[(0, 'opus-5.5')], 100 + 10 + 110 + 50)

    def test_prefix_and_residual(self):
        acc, _, counters = replay([('P', [((c.INSTR, 'System prompt', 'Harness base prompt'), 300)]),
                                   ('B', TYPED, 100, 0, c.B_EST), call(1000)])
        get = lambda p: acc[(p, 0, 'opus-5.5', 'top')][0]
        self.assertEqual(get((c.INSTR, 'System prompt', 'Harness base prompt')), 300)
        self.assertEqual(get(TYPED), 100)
        self.assertEqual(get(c.UNATTRIB), 600)
        self.assertEqual(counters['clamp_events'], 0)

    def test_prefix_scaled_when_it_does_not_fit(self):
        acc, _, counters = replay([('P', [((c.INSTR, 'X'), 300)]), ('B', TYPED, 100, 0, c.B_EST), call(250)])
        self.assertEqual(acc[(TYPED, 0, 'opus-5.5', 'top')][0], 100)
        self.assertEqual(acc[((c.INSTR, 'X'), 0, 'opus-5.5', 'top')][0], 150)
        self.assertEqual(counters['prefix_scaled_events'], 1)
        self.assertEqual(counters['clamp_events'], 0)

    def test_remainder_clamp_never_negative_and_counted(self):
        acc, recon, counters = replay([('P', [((c.INSTR, 'X'), 300)]), ('B', TYPED, 900, 0, c.B_EST),
                                       ('B', (c.USER, 'System reminders'), 100, 0, c.B_EST), call(500)])
        self.assertEqual(counters['clamp_events'], 1)
        cells = {k[0]: v for k, v in acc.items()}
        self.assertTrue(all(v[0] >= 0 and v[1] >= 0 and v[2] >= 0 for v in cells.values()))
        self.assertEqual(sum(v[0] for v in cells.values()), 500)
        self.assertEqual(cells[TYPED][0], 450)
        self.assertEqual(cells[((c.INSTR, 'X'))][0], 0)

    def test_cache_read_priced_on_oldest_tokens(self):
        acc, _, _ = replay([('P', [((c.INSTR, 'X'), 100)]), ('B', TYPED, 100, 0, c.B_EST), call(200, cr=100)])
        usd = lambda p: acc[(p, 0, 'opus-5.5', 'top')][2]
        self.assertAlmostEqual(usd((c.INSTR, 'X')), 100 * 0.20 / 1e6)
        self.assertAlmostEqual(usd(TYPED), 100 * 4 / 1e6)

    def test_codex_reasoning_is_measured(self):
        acc, _, _ = replay([call(100, out=60, reas=40, outs=[((c.CALLS, 'Built-in tools', 'x'), 10, 0)],
                                 model='gpt-6-luna', src='codex')], src='codex')
        self.assertEqual(acc[(c.P_THINK, 0, 'gpt-6-luna', 'top')][0], 40)
        self.assertEqual(acc[((c.CALLS, 'Built-in tools', 'x'), 0, 'gpt-6-luna', 'top')][0], 20)

    def test_range_filtering(self):
        windows = [(900, 2000), (0, 2000)]  # "today" starts at 900
        acc, recon, _ = replay([('B', TYPED, 10, 0, c.B_EST), call(10, t=500), call(10, t=1500), call(10, t=2500)],
                               windows=windows)
        self.assertEqual(recon[(0, 'opus-5.5')], 10)
        self.assertEqual(recon[(1, 'opus-5.5')], 20)
        self.assertEqual(acc[(TYPED, 0, 'opus-5.5', 'top')][0], 10)
        self.assertEqual(acc[(TYPED, 1, 'opus-5.5', 'top')][0], 20)


    def test_thinking_dropped_on_new_user_turn(self):
        acc, _, _ = replay([('B', TYPED, 100, 0, c.B_EST),
                            call(100, out=30, outs=[(c.P_THINK, 30, 0)]),
                            ('T',), call(130)])
        get = lambda p: acc.get((p, 0, 'opus-5.5', 'top'), [0, 0, 0])[0]
        self.assertEqual(get(c.P_THINK), 30)     # generated once, never re-read
        self.assertEqual(get(c.UNATTRIB), 30)


class Pricing(unittest.TestCase):
    @staticmethod
    def usd(C, out=0, cr=0, model='opus-5.5', src='claude'):
        acc, _, _ = replay([call(C, out=out, cr=cr, model=model, src=src)], src=src)
        return sum(v[2] for v in acc.values())

    def test_opus_5_5_rates(self):
        M = 1e6
        self.assertEqual(c.rates('opus-5.5', 'claude', 0, 0, 1000), (4 / M, 0.20 / M, 4 / M, 20 / M))
        self.assertAlmostEqual(c.rates('opus-5.5', 'claude', 10, 0, 1000)[2], 5 / M)
        self.assertAlmostEqual(c.rates('opus-5.5', 'claude', 0, 10, 1000)[2], 8 / M)
        self.assertAlmostEqual(self.usd(1000, out=100, cr=600), (400 * 4 + 600 * 0.20 + 100 * 20) / M)

    def test_haiku_5_5_tier_boundary(self):
        M = 1e6
        self.assertAlmostEqual(self.usd(100000, out=1000, model='haiku-5.5'), (100000 * 0.10 + 1000 * 0.50) / M)
        self.assertAlmostEqual(self.usd(100001, out=1000, model='haiku-5.5'), (100001 * 0.50 + 1000 * 2.50) / M)
        # cache reads count toward the prompt length
        self.assertAlmostEqual(self.usd(100001, cr=100000, model='haiku-5.5'), (1 * 0.50 + 100000 * 0.05) / M)
        self.assertAlmostEqual(c.rates('haiku-5.5', 'claude', 10, 0, 100001)[2], 0.625 / M)
        self.assertAlmostEqual(c.rates('haiku-5.5', 'claude', 0, 10, 100000)[2], 0.20 / M)

    def test_gpt_tier_boundary(self):
        M = 1e6
        for m, short, long in (('gpt-6.1-sol', (2.00, 0.10, 10.0), (4.00, 0.20, 15.0)),
                               ('gpt-6-luna', (0.10, 0.01, 0.50), (0.20, 0.02, 0.75)),
                               ('gpt-6-astra', (10.0, 1.00, 50.0), (20.0, 2.00, 75.0))):
            for C, (i, r, o) in ((272000, short), (272001, long)):
                self.assertAlmostEqual(self.usd(C, out=100, cr=200000, model=m, src='codex'),
                                       ((C - 200000) * i + 200000 * r + 100 * o) / M, msg='%s %d' % (m, C))

    def test_new_models_priced_and_aliased(self):
        M = 1e6
        self.assertEqual(c.model_id('claude-fable-5-1'), 'fable-5.1')
        self.assertEqual(c.model_id('claude-haiku-4-5-20251001'), 'haiku-4.5')
        self.assertEqual(c.model_id('claude-opus-5-5[1m]'), 'opus-5.5')
        self.assertEqual(c.model_id('gpt-6.1-sol'), 'gpt-6.1-sol')
        self.assertEqual(c.model_id('claude-unknown-0'), 'claude-unknown-0')
        self.assertEqual(c.rates('fable-5.1', 'claude', 0, 0, 10 ** 6), (10 / M, 0.25 / M, 10 / M, 50 / M))
        self.assertAlmostEqual(c.rates('fable-5.1', 'claude', 10, 0, 1)[2], 12.50 / M)
        self.assertAlmostEqual(c.rates('fable-5.1', 'claude', 0, 10, 1)[2], 20 / M)
        self.assertAlmostEqual(self.usd(1000, out=10, cr=500, model='haiku-4.5'), (500 * 1 + 500 * 0.10 + 10 * 5) / M)


class Parsing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_claude_role_detection(self):
        p = os.path.join(self.tmp, 'a.jsonl')
        write_jsonl(p, [claude_user('hello ' + ROLE_TAG % 'drone', 1), claude_asst('m1', 1, 10, 0, 0, 5)])
        sess, _ = c.parse_claude(p, False)
        self.assertEqual(sess['role'], 'drone')
        write_jsonl(p, [claude_user('plain', 1), claude_asst('m1', 1, 10, 0, 0, 5)])
        self.assertEqual(c.parse_claude(p, False)[0]['role'], 'top')
        self.assertEqual(c.parse_claude(p, True)[0]['role'], 'native')

    def test_claude_dedupe_uses_last_usage(self):
        p = os.path.join(self.tmp, 'a.jsonl')
        first = claude_asst('m1', 1, 10, 0, 0, 1)
        last = claude_asst('m1', 1, 10, 0, 0, 9)
        write_jsonl(p, [claude_user('hi', 1), first, last, 'not json'])
        sess, st = c.parse_claude(p, False)
        calls = [e for e in sess['events'] if e[0] == 'C']
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][7], 9)
        self.assertEqual(st['unparseable_lines'], 1)

    def test_claude_split_profile_and_reminder(self):
        text = 'A' * 10 + ROLE_TAG % 'review' + '<system-reminder>' + 'r' * 20 + '</system-reminder>'
        parts = dict(c.split_text(text, 'Typed prompts'))
        self.assertEqual(parts[(c.USER, 'Typed prompts')], 10)
        self.assertEqual(parts[(c.INSTR, 'Subagent role profile', 'Review')], len(ROLE_TAG % 'review'))
        self.assertEqual(sum(parts.values()), len(text))

    def test_snapshot_without_tools_keeps_previous_schemas(self):
        p = os.path.join(self.tmp, 's.jsonl')
        snap = lambda tools: {'type': 'attachment', 'timestamp': ts(1), 'attachment': dict(
            {'type': 'prompt_snapshot', 'systemPrompt': ['S' * 50]}, **({'tools': tools} if tools is not None else {}))}
        tool = {'name': 'Read', 'description': 'd' * 40, 'schema': {}}
        write_jsonl(p, [snap([tool]), claude_user('hi', 1), claude_asst('m1', 1, 10, 0, 0, 1),
                        snap(None), claude_asst('m2', 1, 10, 0, 0, 1)])
        sess, st = c.parse_claude(p, False)
        prefixes = [e[1] for e in sess['events'] if e[0] == 'P']
        self.assertEqual(len(prefixes), 2)
        self.assertEqual(sorted(prefixes[0]), sorted(prefixes[1]))
        self.assertNotIn('_last_tools', st)

    def test_codex_role_in_later_message_and_dedupe(self):
        p = os.path.join(self.tmp, 'r.jsonl')
        write_jsonl(p, codex_lines(1, role='review'))
        sess, st = c.parse_codex(p)
        self.assertEqual(sess['role'], 'review')
        self.assertEqual(len([e for e in sess['events'] if e[0] == 'C']), 2)
        self.assertEqual(st['duplicate_usage_skipped'], 1)
        write_jsonl(p, codex_lines(1))
        self.assertEqual(c.parse_codex(p)[0]['role'], 'top')


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        cl = os.path.join(self.tmp, 'claude', 'proj')
        write_jsonl(os.path.join(cl, 's1.jsonl'), [
            claude_user('prompt ' + SECRET, 2),
            claude_asst('m1', 2, 3, 0, 2000, 50, cw1=500),
            claude_user('again', 1),
            claude_asst('m2', 1, 3, 2050, 40, 30),
            claude_asst('m3', 31 * 24, 3, 0, 100, 5),            # older than 30 days: before the 30-day preset
            claude_asst('m4', 3, 3, 0, 100, 5, model='claude-unknown-0'),
        ])
        write_jsonl(os.path.join(cl, 's1', 'subagents', 'agent-1.jsonl'), [
            claude_user('sub task', 5), claude_asst('m5', 5, 10, 0, 500, 20, model='claude-haiku-5-5')])
        write_jsonl(os.path.join(self.tmp, 'codex', '2026', 'rollout.jsonl'), codex_lines(26, role='drone'))
        self.out = os.path.join(self.tmp, 'out.json')
        self.html = os.path.join(self.tmp, 'out.html')
        c.main(['--out', self.out, '--html', self.html, '--end', END, '--jobs', '1',
                '--claude-root', os.path.join(self.tmp, 'claude'), '--codex-root', os.path.join(self.tmp, 'codex')])
        with open(self.out) as fh:
            self.data = json.load(fh)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_reconciles_per_day_and_model(self):
        rec = {r['model']: r for r in self.data['meta']['reconciliation']}
        self.assertTrue(self.data['meta']['reconciled'])
        self.assertEqual(self.data['meta']['reconciliation_mismatches'], [])
        self.assertEqual(rec['opus-5.5']['logged_total'], (2003 + 50) + (2093 + 30) + (103 + 5))
        self.assertEqual(rec['gpt-6-luna']['logged_total'], 560 + 720)
        self.assertEqual(rec['haiku-5.5']['logged_total'], 530)
        for m, r in rec.items():
            self.assertEqual(sum(day_billed(self.data, m).values()), r['leaf_sum'], m)
        self.assertEqual(last_days(self.data, 30, 'opus-5.5'), (2003 + 50) + (2093 + 30))
        self.assertEqual(last_days(self.data, 1, 'gpt-6-luna'), 0)  # codex calls were 26h before end
        self.assertIn('claude-unknown-0', self.data['meta']['unpriced_models'])
        self.assertEqual([m['id'] for m in self.data['models']][:4], c.PRIMARY_MODELS)

    def test_roles_and_days(self):
        self.assertNotIn('ranges', self.data)
        self.assertEqual(self.data['days'], {'first': '2026-09-19', 'last': '2026-10-20', 'count': 32})
        self.assertIsNone(self.data['meta']['collected_from'])
        self.assertEqual([r['id'] for r in self.data['roles']], ['top', 'drone', 'native'])
        self.assertEqual(self.data['meta']['sessions_by_role'], {'claude:native': 1, 'claude:top': 1, 'codex:drone': 1})

    def test_privacy_and_html(self):
        with open(self.out) as fh:
            raw = fh.read()
        self.assertNotIn(SECRET, raw)
        self.assertNotIn(self.tmp, raw)
        with open(self.html) as fh:
            html = fh.read()
        self.assertNotIn('/*__DATA__*/null', html)
        self.assertIn('Claude + Codex · generated', html)
        self.assertNotIn(SECRET, html)

    def test_top_level_categories_and_basis(self):
        labels = [n['label'] for n in self.data['tree']]
        self.assertIn('Unattributed prefix', labels)
        self.assertIn('Assistant output', labels)

        def leaves(n):
            return [n] if 'children' not in n else [x for ch in n['children'] for x in leaves(ch)]
        for leaf in [x for n in self.data['tree'] for x in leaves(n)]:
            self.assertTrue(leaf['basis'])


def run_e2e(tmp, claude=None, codex=None, end=END, extra=()):
    """Writes {relative path: lines} fixtures, runs main, returns (exit code, data or None)."""
    for root, files in (('claude', claude or {}), ('codex', codex or {})):
        os.makedirs(os.path.join(tmp, root), exist_ok=True)
        for rel, lines in files.items():
            write_jsonl(os.path.join(tmp, root, rel), lines)
    out, html = os.path.join(tmp, 'out.json'), os.path.join(tmp, 'out.html')
    for f in (out, html):
        if os.path.exists(f):
            os.remove(f)
    rc = c.main(['--out', out, '--html', html, '--end', end, '--jobs', '1', *extra,
                 '--claude-root', os.path.join(tmp, 'claude'), '--codex-root', os.path.join(tmp, 'codex')])
    if not os.path.exists(out):
        return rc, None
    with open(out) as fh:
        return rc, json.load(fh)


def day_billed(data, model=None):
    """{local day: billed} summed over every leaf row, optionally for one model."""
    first = datetime.date.fromisoformat(data['days']['first'])
    mi = None if model is None else [m['id'] for m in data['models']].index(model)
    out = defaultdict(int)
    for n in data['tree']:
        for (d, m, k), v in node_cells(n).items():
            if mi is None or m == mi:
                out[(first + datetime.timedelta(days=d)).isoformat()] += v[0]
    return {d: b for d, b in out.items() if b}


def last_days(data, n, model=None):
    """Billed over the last n local days of the timeline, as the n-day preset selects them."""
    last = datetime.date.fromisoformat(data['days']['last'])
    return sum(b for d, b in day_billed(data, model).items()
               if (last - datetime.date.fromisoformat(d)).days < n)


class LocalTZ:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.old = os.environ.get('TZ')
        os.environ['TZ'] = self.name
        __import__('time').tzset()

    def __exit__(self, *a):
        if self.old is None:
            os.environ.pop('TZ')
        else:
            os.environ['TZ'] = self.old
        __import__('time').tzset()


def tc_event(h, last, total):
    usage = lambda inp, out: {'input_tokens': inp, 'cached_input_tokens': 0, 'output_tokens': out,
                              'reasoning_output_tokens': 0}
    return {'timestamp': ts(h), 'type': 'event_msg', 'payload': {'type': 'token_count', 'info': {
        'last_token_usage': usage(*last), 'total_token_usage': usage(*total)}}}


class ReviewFixes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_dedup_keys_on_message_id_even_when_request_id_appears_later(self):
        p = os.path.join(self.tmp, 'a.jsonl')
        first = claude_asst('m1', 1, 7, 0, 0, 2)
        del first['requestId']
        write_jsonl(p, [claude_user('hi', 1), first, claude_asst('m1', 1, 9, 0, 0, 4)])
        calls = [e for e in c.parse_claude(p, False)[0]['events'] if e[0] == 'C']
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][7], 4)

    def test_dedup_falls_back_to_request_id_then_uuid(self):
        o, m = {'requestId': 'r', 'uuid': 'u'}, {}
        self.assertEqual(c.claude_call_key(o, m, 3), 'r')
        self.assertEqual(c.claude_call_key({'uuid': 'u'}, m, 3), 'u')
        self.assertEqual(c.claude_call_key({}, {'id': 'msg'}, 3), 'msg')

    def test_claude_role_marker_in_later_user_message(self):
        p = os.path.join(self.tmp, 'a.jsonl')
        write_jsonl(p, [claude_user('plain first', 2), claude_user('next ' + ROLE_TAG % 'review', 2),
                        claude_asst('m1', 1, 10, 0, 0, 5)])
        self.assertEqual(c.parse_claude(p, False)[0]['role'], 'review')
        meta = dict(claude_user(ROLE_TAG % 'drone', 1), isMeta=True)
        write_jsonl(p, [claude_user('plain', 2), meta, claude_asst('m1', 1, 10, 0, 0, 5)])
        self.assertEqual(c.parse_claude(p, False)[0]['role'], 'top')
        write_jsonl(p, [claude_user('a ' + ROLE_TAG % 'envoy', 2), claude_user(ROLE_TAG % 'drone', 2)])
        self.assertEqual(c.parse_claude(p, False)[0]['role'], 'envoy')

    def test_timestamps_honour_offsets_and_z(self):
        utc = c.ts_epoch('2026-10-20T01:00:00Z')
        self.assertEqual(c.ts_epoch('2026-10-20T01:00:00+02:00'), utc - 7200)
        self.assertEqual(c.ts_epoch('2026-10-20T01:00:00.123456789Z'), utc + 0.123456)  # truncated to microseconds
        self.assertEqual(c.ts_epoch('2026-10-20T01:00:00'), utc)  # naive log stamps are UTC
        self.assertEqual(c.ts_epoch('2026-10-19T20:00:00-0500'), utc)
        self.assertIsNone(c.ts_epoch('yesterday'))
        self.assertIsNone(c.ts_epoch(None))

    def _day_hits(self, end, stamps):
        lines = [claude_user('hi', 1)]
        for i, t in enumerate(stamps):
            a = claude_asst('m%d' % i, 1, 1, 0, 0, 10 ** i)
            a['timestamp'] = t
            lines.append(a)
        rc, data = run_e2e(self.tmp, claude={'p/s.jsonl': lines}, end=end)
        self.assertEqual(rc, 0)
        self.assertTrue(data['meta']['oracle']['matched'])
        self.assertTrue(data['meta']['reconciled'])
        return data, day_billed(data)

    def test_days_are_local_and_end_is_exclusive(self):
        with LocalTZ('UTC'):
            data, hits = self._day_hits('2026-10-20T12:00:00+00:00', [
                '2026-10-13T00:00:00Z',        # local midnight: first second of the 13th
                '2026-10-12T23:59:59.999Z',    # last instant of the 12th
                '2026-10-20T12:00:00Z',        # exactly end: excluded
                '2026-10-20T13:00:00+01:00'])  # same instant as end, other offset: excluded
        self.assertEqual(hits, {'2026-10-13': 1 + 1, '2026-10-12': 10 + 1})
        self.assertEqual(last_days(data, 7), 0)
        self.assertEqual(last_days(data, 8), 2)
        self.assertEqual(last_days(data, 9), 2 + 11)

    def test_today_is_the_local_calendar_day(self):
        with LocalTZ('Europe/Prague'):
            data, hits = self._day_hits('2026-10-19T22:30:00Z', [  # 00:30 on 20 Oct in Prague (CEST)
                '2026-10-19T22:10:00Z',          # 00:10 local: today
                '2026-10-20T00:00:00+02:00',     # local midnight: today (inclusive)
                '2026-10-19T21:50:00Z'])         # 23:50 local on the 19th: not today
        self.assertEqual(data['days']['last'], '2026-10-20')
        self.assertEqual(hits, {'2026-10-20': 1 + 1 + 10 + 1, '2026-10-19': 100 + 1})

    def test_codex_token_count_only_logs(self):
        lines = [l for l in codex_lines(2) if l['type'] != 'token_usage_record']
        lines += [tc_event(2, (10, 2), (10, 2)),
                  tc_event(2, (10, 2), (10, 2)),     # repeated cumulative total: same call
                  tc_event(2, (12, 3), (22, 5))]
        p = os.path.join(self.tmp, 'tc.jsonl')
        write_jsonl(p, lines)
        sess, st = c.parse_codex(p)
        calls = [e for e in sess['events'] if e[0] == 'C']
        self.assertEqual([(e[3], e[7]) for e in calls], [(10, 2), (12, 3)])
        self.assertEqual(st['token_count_calls'], 2)
        rc, data = run_e2e(self.tmp, codex={'2026/r.jsonl': lines})
        self.assertEqual(rc, 0)
        rec = {r['model']: r for r in data['meta']['reconciliation']}
        self.assertEqual(rec['gpt-6-luna']['logged_total'], 12 + 15)
        self.assertTrue(data['meta']['oracle']['matched'])

    def test_codex_token_count_ignored_when_usage_records_exist(self):
        lines = codex_lines(2) + [tc_event(2, (999, 9), (999, 9))]
        p = os.path.join(self.tmp, 'r.jsonl')
        write_jsonl(p, lines)
        self.assertEqual(len([e for e in c.parse_codex(p)[0]['events'] if e[0] == 'C']), 2)

    def test_oracle_rows_cover_roles_and_match(self):
        rc, data = run_e2e(self.tmp, claude={
            'p/s.jsonl': [claude_user(ROLE_TAG % 'envoy', 2), claude_asst('m1', 2, 5, 0, 10, 3)]},
            codex={'2026/r.jsonl': codex_lines(3, role='drone')})
        self.assertEqual(rc, 0)
        o = data['meta']['oracle']
        self.assertTrue(o['matched'])
        self.assertEqual(o['mismatches'], [])
        rows = {(r['model'], r['role']): r['oracle'] for r in o['rows']}
        self.assertEqual(rows, {('opus-5.5', 'envoy'): 18, ('gpt-6-luna', 'drone'): 560 + 720})

    def test_oracle_mismatch_fails_unless_allowed(self):
        real = c.run_oracle

        def skewed(jobs, windows, *a):
            out, stats = real(jobs, windows, *a)
            out[(len(windows) - 1, 'opus-5.5', 'top')] += 1
            return out, stats
        c.run_oracle = skewed
        try:
            fx = {'p/s.jsonl': [claude_user('hi', 1), claude_asst('m1', 1, 5, 0, 0, 3)]}
            rc, data = run_e2e(self.tmp, claude=fx)
            self.assertEqual(rc, 2)
            self.assertIsNone(data)
            rc, data = run_e2e(self.tmp, claude=fx, extra=('--allow-mismatch',))
            self.assertEqual(rc, 0)
            self.assertFalse(data['meta']['oracle']['matched'])
            self.assertEqual([(m['day'], m['diff']) for m in data['meta']['oracle']['mismatches']],
                             [(data['days']['last'], -1)])
        finally:
            c.run_oracle = real

    def test_unpriced_models_are_counted_and_shown(self):
        rc, data = run_e2e(self.tmp, claude={'p/s.jsonl': [
            claude_user('hi', 1), claude_asst('m1', 1, 5, 0, 0, 3, model='claude-unknown-0'),
            claude_asst('m2', 1, 5, 0, 0, 3, model='claude-unknown-0'), claude_asst('m3', 1, 5, 0, 0, 3)]})
        self.assertEqual(rc, 0)
        m = data['meta']
        self.assertEqual(m['unpriced_calls'], {'claude-unknown-0': 2})
        self.assertEqual(m['coverageNote'], 'Costs use official API rates (platform.claude.com/docs/en/about-claude/'
                         'pricing, developers.openai.com/api/docs/pricing; fetched 2026-10-20). 2 calls on unpriced '
                         'models (counted as $0): claude-unknown-0 (2)')
        with open(os.path.join(self.tmp, 'out.html')) as fh:
            html = fh.read()
        self.assertIn('id="coverage"', html)
        self.assertIn('2 calls on unpriced models', html)


def stamped(mid, when, out=1, model='claude-opus-5-5'):
    a = claude_asst(mid, 1, 1, 0, 0, out, model=model)
    a['timestamp'] = when
    return a


class Timeline(unittest.TestCase):
    def test_small_daily_values_add_up_over_any_window(self):
        basis, days = defaultdict(set), 30
        acc = {}
        for d in range(days):
            acc[(BASH_RES + ('cat',), d, 'm', 'top')] = [3, 0.4, 0.00004]
            acc[(BASH_RES + ('ls',), d, 'm', 'top')] = [2, 0.3, 0.00003]
            acc[(c.READ_RESULT, d, 'm', 'top')] = [1, 0.4, 0.00004]
        for k in acc:
            basis[k[0]].add(c.B_EST)
        tree = c.build_tree(acc, ['m'], ['top'], basis)

        def rows(n):
            return [r for k in n.get('children', []) for r in rows(k)] if 'children' in n else n['v']
        bash = find(tree, c.RESULTS, 'Built-in tools', 'Bash')
        for node, per_day, units in ((bash, (0.7, 0.00007), 1), (find(tree, *c.READ_RESULT), (0.4, 0.00004), 1),
                                     (find(bash['children'], 'cat'), (0.4, 0.00004), 2), (find(bash['children'], 'ls'), (0.3, 0.00003), 2)):
            got = rows(node)
            for lo in range(days):
                for hi in range(lo, days):
                    inside = [r for r in got if lo <= r[0] <= hi]
                    n = hi - lo + 1
                    self.assertLess(abs(sum(r[4] for r in inside) - per_day[0] * n), units + 1e-9, (node['label'], lo, hi))
                    self.assertLess(abs(sum(r[5] for r in inside) - per_day[1] * n), 0.0001 * units + 1e-9, (lo, hi))
                hi_sum = [r for r in got if r[0] <= lo]
                self.assertLess(abs(sum(r[4] for r in hi_sum) - per_day[0] * (lo + 1)), 1, (node['label'], lo))
                self.assertLess(abs(sum(r[5] for r in hi_sum) - per_day[1] * (lo + 1)), 0.0001, (node['label'], lo))
            self.assertEqual(sum(r[4] for r in rows(bash)), round(0.7 * days))
            self.assertAlmostEqual(sum(r[5] for r in rows(bash)), round(0.00007 * days, 4), places=9)
            self.assertTrue(all(r[3] or r[4] or r[5] for r in got))

    def test_carried_rounding_reaches_the_leaves_of_an_internal_command(self):
        basis = defaultdict(set)
        acc = {(BASH_RES + ('cat', c.EXPLICIT_INSTR, 'AGENTS.md'), 0, 'm', 'top'): [1, 0.4, 0.00004],
               (BASH_RES + ('ls',), 0, 'm', 'top'): [1, 0.6, 0.00006],
               (BASH_RES + ('ls',), 1, 'm', 'top'): [1, 0.1, 0.00001],
               (BASH_RES + ('ls',), 2, 'm', 'top'): [1, 0.5, 0.00005]}
        for k in acc:
            basis[k[0]].add(c.B_EST)
        bash = find(c.build_tree(acc, ['m'], ['top'], basis), c.RESULTS, 'Built-in tools', 'Bash')
        def rows(n):
            return [r for k in n['children'] for r in rows(k)] if 'children' in n else n['v']
        got = rows(bash)
        self.assertEqual(sum(r[4] for r in got), 2)
        self.assertAlmostEqual(sum(r[5] for r in got), 0.0002, places=9)

    def test_split_rows_stay_nonnegative_and_match_their_histogram(self):
        basis, acc, hacc = defaultdict(set), {}, {}
        for d in range(20):
            for cmd, u, usd in (('cat', 0.6, 0.00006), ('ls', 0.3, 0.00003), ('rg', 0.45, 0.00004)):
                if cmd != 'cat' or d % 3 == 0:
                    acc[(BASH_RES + (cmd,), d, 'm', 'top')] = [2, u, usd]
                    hacc[(BASH_RES + (cmd,), d, 'm', 'top')] = {0: [1, 1, u / 3, usd / 3], 2: [1, 1, u * 2 / 3, usd * 2 / 3]}
        for k in acc:
            basis[k[0]].add(c.B_EST)
        bash = find(c.build_tree(acc, ['m'], ['top'], basis, hacc=hacc), c.RESULTS, 'Built-in tools', 'Bash')
        carried = 0
        for leaf in bash['children']:
            for row, bins in zip(leaf['v'], leaf['h']):
                self.assertGreaterEqual(min(row[4], row[5]), 0, (leaf['label'], row))
                self.assertEqual(sum(b[3] for b in bins), row[4], (leaf['label'], row))
                self.assertAlmostEqual(sum(b[4] for b in bins), row[5], places=9)
                self.assertEqual(sum(b[2] for b in bins), row[3])
                carried += not row[3]
        self.assertTrue(carried)

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_dst_days_in_prague(self):
        with LocalTZ('Europe/Prague'):
            rc, data = run_e2e(self.tmp, end='2026-10-26T12:00:00Z', claude={'p/s.jsonl': [
                claude_user('hi', 1),
                stamped('a', '2026-10-25T00:30:00+02:00', 1),     # 25 Oct, before the clocks go back
                stamped('b', '2026-10-25T23:30:00+01:00', 10),    # 25 Oct, the 25th hour of that day
                stamped('c', '2026-10-24T23:59:00+02:00', 100),
                stamped('d', '2026-10-26T00:10:00+01:00', 1000),
                stamped('e', '2026-03-29T01:59:00+01:00', 10000),  # just before the spring change
                stamped('f', '2026-03-29T03:00:00+02:00', 100000),
                stamped('g', '2026-03-28T23:59:00+01:00', 1000000)]})
        self.assertEqual(rc, 0)
        self.assertEqual(day_billed(data), {
            '2026-10-25': 2 + 11, '2026-10-24': 101, '2026-10-26': 1001,
            '2026-03-29': 10001 + 100001, '2026-03-28': 1000001})
        self.assertEqual(data['days'], {'first': '2026-03-28', 'last': '2026-10-26', 'count': 213})
        self.assertEqual(last_days(data, 1), 1001)
        self.assertEqual(last_days(data, 2), 1001 + 13)

    def test_short_history_still_spans_more_than_thirty_days(self):
        with LocalTZ('UTC'):
            rc, data = run_e2e(self.tmp, claude={'p/s.jsonl': [claude_user('hi', 1), claude_asst('m1', 1, 1, 0, 0, 1)]})
        self.assertEqual(rc, 0)
        self.assertEqual(data['days'], {'first': '2026-09-20', 'last': '2026-10-20', 'count': c.MIN_DAYS})
        self.assertGreater(data['days']['count'], 30)

    def test_lookback_starts_at_local_midnight(self):
        with LocalTZ('UTC'):
            rc, data = run_e2e(self.tmp, extra=('--lookback', '3'), claude={'p/s.jsonl': [
                claude_user('hi', 1),
                stamped('a', '2026-10-17T23:59:59Z', 1),
                stamped('b', '2026-10-18T00:00:00Z', 10),
                stamped('c', '2025-01-01T00:00:00Z', 100)]})
        self.assertEqual(rc, 0)
        self.assertEqual(day_billed(data), {'2026-10-18': 11})
        self.assertEqual(data['meta']['collected_from'], '2026-10-18T00:00:00+00:00')
        self.assertEqual(data['days']['first'], '2026-09-20')

    def test_all_history_by_default(self):
        with LocalTZ('UTC'):
            rc, data = run_e2e(self.tmp, claude={'p/s.jsonl': [
                claude_user('hi', 1), stamped('a', '2025-01-01T08:00:00Z', 100), stamped('b', '2026-10-20T08:00:00Z')]})
        self.assertEqual(rc, 0)
        self.assertEqual(day_billed(data), {'2025-01-01': 101, '2026-10-20': 2})
        self.assertEqual(data['days']['first'], '2025-01-01')
        self.assertEqual(data['days']['count'], 658)
        self.assertEqual(last_days(data, 30), 2)

    def test_leaf_rows_are_sparse_and_sorted(self):
        e2e = EndToEnd()
        e2e.setUp()
        try:
            data = e2e.data
            nd, nm, nr = data['days']['count'], len(data['models']), len(data['roles'])
            leaves = []

            def walk(n):
                if 'children' in n:
                    for ch in n['children']:
                        walk(ch)
                else:
                    leaves.append(n)
            for n in data['tree']:
                walk(n)
            for leaf in leaves:
                keys = [tuple(r[:3]) for r in leaf['v']]
                self.assertEqual(keys, sorted(set(keys)), leaf['label'])
                for d, m, k, b, u, usd in leaf['v']:
                    self.assertTrue(0 <= d < nd and 0 <= m < nm and 0 <= k < nr)
                    self.assertTrue(b or u or usd)
                if 'h' in leaf:
                    self.assertEqual(len(leaf['h']), len(leaf['v']))
            self.assertTrue(any(leaf['v'] for leaf in leaves))
        finally:
            shutil.rmtree(e2e.tmp, ignore_errors=True)


class NoSubagentConfig(unittest.TestCase):
    """A home with no ~/.config/subagents and no role-dispatch environment still collects and renders."""

    def test_collects_from_a_bare_home(self):
        import subprocess
        home = tempfile.mkdtemp()
        try:
            write_jsonl(os.path.join(home, '.claude', 'projects', 'proj', 's.jsonl'),
                        [claude_user('plain prompt ' + SECRET, 2), claude_asst('m1', 2, 5, 0, 100, 7)])
            write_jsonl(os.path.join(home, '.claude', 'projects', 'proj', 's', 'subagents', 'agent-1.jsonl'),
                        [claude_user('sub task', 3), claude_asst('m2', 3, 5, 0, 10, 2)])
            write_jsonl(os.path.join(home, '.codex', 'sessions', '2026', 'r.jsonl'), codex_lines(4))
            out, html = os.path.join(home, 'out.json'), os.path.join(home, 'out.html')
            env = {'HOME': home, 'PATH': '/usr/bin:/bin', 'TZ': 'UTC', 'LANG': 'C.UTF-8'}
            r = subprocess.run([sys.executable, '-I', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'collect.py'),
                                '--out', out, '--html', html, '--end', END, '--jobs', '1'],
                               env=env, cwd=home, capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertFalse(os.path.exists(os.path.join(home, '.config')))
            with open(out) as fh:
                raw = fh.read()
            data = json.loads(raw)
            self.assertEqual([x['id'] for x in data['roles']], ['top', 'native'])
            self.assertEqual(data['meta']['sessions_by_role'], {'claude:native': 1, 'claude:top': 1, 'codex:top': 1})
            self.assertTrue(data['meta']['reconciled'])
            self.assertTrue(data['meta']['oracle']['matched'])
            self.assertEqual(data['days']['last'], '2026-10-20')
            with open(html) as fh:
                page = fh.read()
            for blob in (raw, page):
                self.assertNotIn(SECRET, blob)
                self.assertNotIn(home, blob)
        finally:
            shutil.rmtree(home)


RULES = (c.INSTR, 'Instruction files', 'Rules files')


def instr_attachment(h, files):
    return {'type': 'attachment', 'timestamp': ts(h), 'attachment': {
        'type': 'instructions', 'files': [{'path': p, 'content': SECRET + 'x' * n} for p, n in files]}}


def find(nodes, *labels):
    for lab in labels:
        nodes = {n['label']: n for n in nodes}[lab]
        if lab != labels[-1]:
            nodes = nodes['children']
    return nodes


def node_cells(n):
    """{(day, model, role): [billed, unique, usd]} summed over a node's leaves."""
    out = defaultdict(lambda: [0, 0, 0.0])
    if 'children' in n:
        for ch in n['children']:
            for k, v in node_cells(ch).items():
                for j in range(3):
                    out[k][j] += v[j]
        return out
    for d, m, k, *cell in n['v']:
        for j in range(3):
            out[(d, m, k)][j] += cell[j]
    return out


def leaf_cell(n, d=0, m=0, k=0):
    """A leaf's [billed, unique, usd] on day d for model m and role k; zeros when it has no row."""
    return next(([*r[3:]] for r in n['v'] if r[:3] == [d, m, k]), [0, 0, 0.0])


class FileLabels(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp()
        c._repo_root.cache_clear()
        c._worktree_project_repo.cache_clear()

    def tearDown(self):
        shutil.rmtree(self.home)
        c._repo_root.cache_clear()
        c._worktree_project_repo.cache_clear()

    def mk(self, rel, content=None, is_dir=False):
        p = os.path.join(self.home, rel)
        os.makedirs(p if is_dir else os.path.dirname(p), exist_ok=True)
        if content is not None:
            with open(p, 'w') as fh:
                fh.write(content)
        return p

    def label(self, p):
        return c.file_label(p, self.home)

    def test_home_files_get_tilde_prefix(self):
        self.assertEqual(self.label(os.path.join(self.home, '.claude/rules/foo.md')), '~/.claude/rules/foo.md')
        self.assertEqual(self.label('~/.claude/CLAUDE.md'), '~/.claude/CLAUDE.md')

    def test_other_users_and_outside_paths_are_sanitized(self):
        self.assertEqual(self.label('/Users/someone/.claude/CLAUDE.md'), '~/.claude/CLAUDE.md')
        self.assertEqual(self.label('/home/someone/x/AGENTS.md'), '~/x/AGENTS.md')
        self.assertEqual(self.label('/opt/a/b/c/AGENTS.md'), '…/c/AGENTS.md')
        self.assertEqual(self.label(''), 'Unknown file')

    def test_repo_files_show_repo_name_and_relative_path(self):
        self.mk('src/myrepo/.git', is_dir=True)
        self.assertEqual(self.label(os.path.join(self.home, 'src/myrepo/.claude/rules/a.md')), 'myrepo/.claude/rules/a.md')
        # a deleted in-repo worktree folds into the repository's own file
        self.assertEqual(self.label(os.path.join(self.home, 'src/myrepo/.worktrees/gone/AGENTS.md')), 'myrepo/AGENTS.md')

    def test_worktrees_take_the_main_repo_name(self):
        self.mk('wt/a/.git', 'gitdir: %s/src/bare-repo/worktrees/a\n' % self.home)
        self.mk('wt/b/.git', 'gitdir: %s/src/normal/.git/worktrees/b\n' % self.home)
        self.assertEqual(self.label(os.path.join(self.home, 'wt/a/rules/x.md')), 'bare-repo/rules/x.md')
        self.assertEqual(self.label(os.path.join(self.home, 'wt/b/rules/x.md')), 'normal/rules/x.md')

    def test_deleted_t3_worktree_uses_sibling_repo_name(self):
        self.mk('.t3/worktrees/proj/live/.git', 'gitdir: %s/src/real-repo/worktrees/live\n' % self.home)
        self.assertEqual(self.label(os.path.join(self.home, '.t3/worktrees/proj/deleted/AGENTS.md')), 'real-repo/AGENTS.md')
        self.assertEqual(self.label(os.path.join(self.home, '.t3/worktrees/other/deleted/AGENTS.md')), 'other/AGENTS.md')
        self.mk('.t3/worktrees/proj/live/.worktrees/nested/.git', 'gitdir: %s/src/real-repo/worktrees/nested\n' % self.home)
        self.assertEqual(self.label(os.path.join(self.home, '.t3/worktrees/proj/live/.worktrees/nested/r.md')), 'real-repo/r.md')
        self.assertEqual(self.label(os.path.join(self.home, '.t3/worktrees/proj/live/.worktrees/gone/r.md')), 'real-repo/r.md')
        self.assertEqual(self.label(os.path.join(self.home, '.t3/worktrees/proj/deleted/.worktrees/gone/r.md')), 'real-repo/r.md')

    def test_git_tracked_config_dir_keeps_tilde_label(self):
        self.mk('.codex/memories/.git', is_dir=True)
        self.assertEqual(self.label(os.path.join(self.home, '.codex/memories/memory_summary.md')),
                         '~/.codex/memories/memory_summary.md')

    def test_windows_paths_are_sanitized(self):
        self.assertEqual(self.label('C:\\Users\\someone\\.claude\\CLAUDE.md'), '~/.claude/CLAUDE.md')
        self.assertEqual(self.label('c:/home/someone/x/AGENTS.md'), '~/x/AGENTS.md')
        self.assertEqual(self.label('//server/share/Users/someone/r.md'), '~/r.md')
        self.assertEqual(self.label('D:/work/secret-org/repo/AGENTS.md'), '…/repo/AGENTS.md')
        self.assertEqual(self.label('//server/share/a/b.md'), '…/a/b.md')

    def test_gitdir_with_spaces_and_relative_path(self):
        self.mk('wt/s/.git', 'gitdir: %s/src/my repo/.git/worktrees/s\n' % self.home)
        self.mk('src/rel-repo/.git', is_dir=True)
        self.mk('src/rel-repo/.worktrees/r/.git', 'gitdir: ../../.git/worktrees/r\n')
        self.assertEqual(self.label(os.path.join(self.home, 'wt/s/x.md')), 'my repo/x.md')
        self.assertEqual(self.label(os.path.join(self.home, 'src/rel-repo/.worktrees/r/x.md')), 'rel-repo/x.md')

    def test_hidden_dir_worktree_keeps_repo_label(self):
        self.mk('.worktrees/feat/.git', 'gitdir: %s/src/main-repo/.git/worktrees/feat\n' % self.home)
        self.assertEqual(self.label(os.path.join(self.home, '.worktrees/feat/AGENTS.md')), 'main-repo/AGENTS.md')

    def test_project_slug_hides_home_path(self):
        slug = self.home.replace('/', '-').replace('.', '-')
        lab = self.label(os.path.join(self.home, '.claude/projects', slug + '-repo', 'memory/MEMORY.md'))
        self.assertEqual(lab, '~/.claude/projects/~-repo/memory/MEMORY.md')
        self.assertEqual(self.label('/x/.claude/projects/-Users-someone-repo/memory/MEMORY.md'), '…/memory/MEMORY.md')
        self.assertEqual(c.HOME_SLUG_RE.sub('~', '-Users-someone-repo'), '~-repo')
        self.assertNotIn(self.home, lab)


class PerFileSplit(unittest.TestCase):
    def events(self, labelled):
        f = lambda p, n, lab: ('B', p, n, 0, c.B_EST) + ((lab,) if labelled else ())
        return [
            f(RULES, 333, 'repo/a.md'), f(RULES, 101, 'repo/b.md'), ('B', TYPED, 50, 0, c.B_EST),
            call(500, out=7, outs=[(c.P_TEXT, 5, 0)]),
            f(RULES, 77, 'repo/c.md'), f(RULES, 333, 'repo/a.md'),  # a.md injected again (edited version)
            call(1400, cr=480, out=3, outs=[(c.P_TEXT, 5, 0)]),
            call(1405, cr=1400),
            ('R',),
            f(RULES, 10, 'repo/b.md'),
            call(30),
        ]

    def test_files_sum_exactly_to_the_group(self):
        whole, _, _ = replay(self.events(False))
        split, _, _ = replay(self.events(True))
        files = {k: v for k, v in split.items() if k[0][:3] == RULES and len(k[0]) == 4}
        self.assertEqual({k[0][3] for k in files}, {'repo/a.md', 'repo/b.md', 'repo/c.md'})
        g = whole[(RULES, 0, 'opus-5.5', 'top')]
        self.assertEqual(sum(v[0] for v in files.values()), g[0])
        self.assertAlmostEqual(sum(v[1] for v in files.values()), g[1])
        self.assertNotIn((RULES, 0, 'opus-5.5', 'top'), split)
        for k, v in split.items():  # everything outside the group is unchanged
            if len(k[0]) != 4:
                self.assertEqual(v[:2], whole[k][:2])
        # each version counts at its own size: a.md was sent twice, so it outweighs b.md and c.md
        get = lambda f: split[(RULES + (f,), 0, 'opus-5.5', 'top')]
        self.assertGreater(get('repo/a.md')[1], get('repo/b.md')[1] + get('repo/c.md')[1])

    def test_priced_files_sum_to_group_usd(self):
        ev = lambda lab: [e[:4] + ('', ) if False else e for e in self.events(lab)]
        model_events = lambda lab: [e if e[0] != 'C' else e[:2] + ('claude-opus-5-5',) + e[3:] for e in ev(lab)]
        whole, _, _ = replay(model_events(False))
        split, _, _ = replay(model_events(True))
        usd = sum(v[2] for k, v in split.items() if k[0][:3] == RULES)
        self.assertAlmostEqual(usd, whole[(RULES, 0, 'claude-opus-5-5', 'top')][2], places=9)

    def test_top_n_fold_and_single_file_collapse(self):
        acc, basis = {}, defaultdict(set)
        for i in range(5):
            acc[(RULES + ('f%d.md' % i,), 0, 'm', 'top')] = [100 - i, 1, 0.01]
            basis[RULES + ('f%d.md' % i,)].add(c.B_EST)
        mem = (c.INSTR, 'Instruction files', 'User CLAUDE.md')
        acc[(mem + ('~/.claude/CLAUDE.md',), 0, 'm', 'top')] = [9, 1, 0.0]
        tree = c.build_tree(acc, ['m'], ['top'], basis, top_files=3)
        rules = find(tree, c.INSTR, 'Instruction files', 'Rules files')
        self.assertEqual([n['label'] for n in rules['children']], ['f0.md', 'f1.md', 'f2.md', 'Other (2 files)'])
        self.assertEqual(leaf_cell(rules['children'][-1]), [97 + 96, 2, 0.02])
        mem_node = find(tree, c.INSTR, 'Instruction files', 'User CLAUDE.md')
        self.assertEqual([n['label'] for n in mem_node['children']], ['~/.claude/CLAUDE.md'])
        self.assertEqual(leaf_cell(mem_node['children'][0]), [9, 1, 0.0])

    def test_single_file_group_keeps_filename(self):
        acc = {(RULES + ('demo/rules/one.md',), 0, 'm', 'top'): [100, 50, 0.01]}
        basis = defaultdict(set, {RULES + ('demo/rules/one.md',): {c.B_EST}})
        tree = c.build_tree(acc, ['m'], ['top'], basis)
        rules = find(tree, c.INSTR, 'Instruction files', 'Rules files')
        self.assertEqual([n['label'] for n in rules['children']], ['demo/rules/one.md'])
        self.assertEqual(leaf_cell(rules['children'][0]), [100, 50, 0.01])


class PerFileEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_rules_files_are_drillable_and_private(self):
        repo = os.path.join(self.tmp, 'repo')
        os.makedirs(os.path.join(repo, '.git'))
        files = [(os.path.join(repo, '.claude/rules/r%02d.md' % i), 400 + 40 * i) for i in range(14)]
        files.append((os.path.join(repo, 'CLAUDE.md'), 300))
        lines = [claude_user('hi', 2), instr_attachment(2, files), claude_asst('m1', 2, 10, 0, 9000, 20),
                 claude_user('more', 1), claude_asst('m2', 1, 10, 9030, 30, 20)]
        rc, data = run_e2e(self.tmp, claude={'p/s.jsonl': lines})
        self.assertEqual(rc, 0)
        self.assertTrue(data['meta']['reconciled'])
        rules = find(data['tree'], c.INSTR, 'Instruction files', 'Rules files')
        labels = [n['label'] for n in rules['children']]
        self.assertEqual(len(labels), c.FILE_TOP_N + 1)
        self.assertEqual(labels[0], 'repo/.claude/rules/r13.md')
        self.assertEqual(labels[-1], 'Other (2 files)')
        self.assertEqual(find(data['tree'], c.INSTR, 'Instruction files', 'Project CLAUDE.md')['label'], 'Project CLAUDE.md')
        for node in rules['children'][:-1]:
            self.assertEqual(node['fl'], node['label'])
            self.assertEqual([m['label'] for m in node['children']], [c.HOW_LAUNCH])
            self.assertEqual(node['children'][0]['basis'], c.B_EST)
        self.assertEqual(rules['children'][-1]['basis'], c.B_EST)
        raw = ''
        for name in ('out.json', 'out.html'):
            with open(os.path.join(self.tmp, name)) as fh:
                raw += fh.read()
        for bad in (SECRET, self.tmp, c.HOME, '/Users/', '/home/', c.HOME.replace('/', '-').replace('.', '-')):
            self.assertNotIn(bad, raw)

    def test_codex_agents_md_split_at_project_doc(self):
        text = '# AGENTS.md instructions for /Users/someone/proj\n\n' + 'u' * 300 + c.PROJECT_DOC_SEP + 'p' * 500
        ev = c._codex_agents_md(text)
        self.assertEqual([e[2] for e in ev], [('~/.codex/AGENTS.md', c.HOW_LAUNCH), ('~/proj/AGENTS.md', c.HOW_LAUNCH)])
        self.assertEqual(sum(e[1] for e in ev), len(text))
        spaced = '# AGENTS.md instructions for /Users/someone/My Projects/app\n\n<INSTRUCTIONS>x'
        self.assertEqual(c._codex_agents_md(spaced)[0][2][0], '~/My Projects/app/AGENTS.md')

    def test_codex_user_role_agents_md_is_attributed_per_file(self):
        doc = ('# AGENTS.md instructions for /Users/someone/proj\n\n<INSTRUCTIONS>\n' + 'u' * 300 + '\n\n'
               + c.PROJECT_DOC_SEP + '\n\n' + 'p' * 500 + '\n</INSTRUCTIONS>')
        env = '<environment_context>\n  <cwd>/Users/someone/proj</cwd>\n</environment_context>'
        item = {'type': 'message', 'role': 'user',
                'content': [{'type': 'input_text', 'text': doc}, {'type': 'input_text', 'text': env}]}
        events = []
        c._codex_item(item, events, {}, [])
        agents = [e for e in events if e[1] == (c.INSTR, 'AGENTS.md instructions')]
        self.assertEqual([e[5][0] for e in agents], ['~/.codex/AGENTS.md', '~/proj/AGENTS.md'])
        self.assertEqual(sum(e[2] for e in agents), len(doc))
        self.assertEqual([(e[1], e[2]) for e in events if e not in agents], [((c.INSTR, 'Environment context'), len(env))])
        # no project doc: one file, the project's
        events = []
        c._codex_item(dict(item, content=[{'type': 'input_text', 'text': '# AGENTS.md instructions for /Users/someone/proj\n\n<INSTRUCTIONS>\nx\n</INSTRUCTIONS>'}]), events, {}, [])
        self.assertEqual([(e[1][1], e[5]) for e in events], [('AGENTS.md instructions', ('~/proj/AGENTS.md', c.HOW_LAUNCH))])
        # typed prompts that mention the header, or an unterminated doc, stay typed prompts
        for text in ('please read the AGENTS.md instructions for /Users/someone/proj\n</INSTRUCTIONS>',
                     '# AGENTS.md instructions for /Users/someone/proj\n\n<INSTRUCTIONS>\nno end'):
            events = []
            c._codex_item(dict(item, content=[{'type': 'input_text', 'text': text}]), events, {}, [])
            self.assertEqual([(e[1], e[2]) for e in events], [((c.USER, 'Typed prompts'), len(text))])


    def test_codex_developer_mention_of_agents_md_is_not_a_file(self):
        def run(text):
            events = []
            c._codex_item({'type': 'message', 'role': 'developer', 'content': [{'type': 'input_text', 'text': text}]}, events, {}, [])
            return [(e[1], e[2], *e[5:]) for e in events]
        for text in ('Follow any AGENTS.md files in the repository.\n' + 'd' * 200,
                     'Read the AGENTS.md instructions for /Users/someone/proj before editing.\n' + 'd' * 200,
                     'AGENTS.md scope rules\n' + c.PROJECT_DOC_SEP + '\n' + 'd' * 50):
            self.assertEqual(run(text), [(c.P_DEV, len(text))])
        # the anchored wrapper is still a file, the trailing text stays a developer message
        doc = '# AGENTS.md instructions for /Users/someone/proj\n\n<INSTRUCTIONS>\n' + 'p' * 80 + '\n</INSTRUCTIONS>'
        tail = '\n' + 'd' * 40
        self.assertEqual(run(doc + tail), [((c.INSTR, 'AGENTS.md instructions'), len(doc), ('~/proj/AGENTS.md', c.HOW_LAUNCH)),
                                           (c.P_DEV, len(tail))])


class EntryMethods(unittest.TestCase):
    """How an instruction file entered context, and explicit reads of instruction files."""

    def test_instruction_path_predicate(self):
        for p in ('/r/CLAUDE.md', '/r/sub/CLAUDE.local.md', '/r/AGENTS.md', '/h/.claude/rules/a/b.md',
                  '/r/.agents/rules/x.md', '/h/.claude/projects/-r/memory/MEMORY.md', '/h/.codex/memories/m.md'):
            self.assertTrue(c.is_instruction_path(p), p)
        for p in ('/r/README.md', '/h/.claude/rules/x.txt', '/r/claude.md', '/h/.claude/projects/-r/s.jsonl', None):
            self.assertFalse(c.is_instruction_path(p), p)

    def test_explicit_read_commands(self):
        home, cwd = '/nonexistent-home', '/nonexistent-home/repo'
        er = lambda cmd: c.explicit_read(cmd, cwd, home)
        self.assertEqual(er('cat CLAUDE.md'), '~/repo/CLAUDE.md')
        self.assertEqual(er('cat ~/.claude/rules/git.md | head -20'), '~/.claude/rules/git.md')
        self.assertEqual(er("sed -n '1,40p' .claude/rules/a.md"), '~/repo/.claude/rules/a.md')
        self.assertEqual(er('head -n 5 AGENTS.md 2>/dev/null'), '~/repo/AGENTS.md')
        self.assertEqual(er('cd sub && cat CLAUDE.md'), '~/repo/sub/CLAUDE.md')
        self.assertEqual(er('cat CLAUDE.md AGENTS.md'), c.SEVERAL_FILES)
        # a glob or brace operand may name several files, so its text never becomes a file label
        self.assertEqual(er('cat .claude/rules/*.md'), c.SEVERAL_FILES)
        self.assertEqual(er('cat .claude/rules/{a,b}.md'), c.SEVERAL_FILES)
        for cmd in ('cat README.md', 'cat CLAUDE.md README.md', 'rg -n x CLAUDE.md', 'cat CLAUDE.md && rm x',
                    'cat CLAUDE.md; ls', 'sed CLAUDE.md', 'cd a b && cat CLAUDE.md', 'echo CLAUDE.md', '',
                    # writes, follows and redirects to files are not plain reads
                    'sed -i s/a/b/ CLAUDE.md', 'tail -f CLAUDE.md', 'cat AGENTS.md > CLAUDE.md',
                    'cat CLAUDE.md >> out.txt', "sed -n 'w out' CLAUDE.md", "sed 's#a#b#w out' CLAUDE.md",
                    'sed s/a/b/ CLAUDE.md', 'cat AGENTS.md | cat <&3',
                    # every later pipe stage must be a read program without file operands
                    'cat AGENTS.md | cat README.md', 'cat CLAUDE.md | tee CLAUDE.md', 'cat CLAUDE.md |& head',
                    # stdin counts only as one plain '<' file and the only input; later or numbered
                    # redirections can replace descriptor 0, and which one wins is not tracked
                    'cat < AGENTS.md <&3', 'cat < AGENTS.md 0<&3', 'cat AGENTS.md <&-', 'cat 0<&3 AGENTS.md',
                    'cat 0<AGENTS.md', 'cat 3<AGENTS.md', 'cat 3< AGENTS.md', 'cat <>AGENTS.md',
                    'cat < AGENTS.md < CLAUDE.md', 'cat AGENTS.md < CLAUDE.md', 'cat <&3 < AGENTS.md'):
            self.assertIsNone(er(cmd), cmd)
        for cmd in ('cat < AGENTS.md', 'head -50 AGENTS.md', 'cat AGENTS.md 2>&1', 'cat AGENTS.md 2>/dev/null',
                    'sed -n 1,80p AGENTS.md', "sed -n '1,5p;10,20p' AGENTS.md", 'sed 5q AGENTS.md',
                    'cat AGENTS.md | head -n 5', 'head -n 5 < AGENTS.md', 'cat AGENTS.md >&2'):
            self.assertEqual(er(cmd), '~/repo/AGENTS.md', cmd)

    def test_result_labels(self):
        cwd = '/nonexistent-home/repo'
        self.assertEqual(c.result_label_for(None, 'Read', {'file_path': cwd + '/.claude/rules/a.md'}, cwd)[0][0], c.EXPLICIT_INSTR)
        self.assertEqual(c.result_label_for(None, 'Read', {'file_path': 'CLAUDE.md'}, cwd)[0][0], c.EXPLICIT_INSTR)
        self.assertEqual(c.result_label_for(None, 'Read', {'file_path': cwd + '/x.py'}, cwd), ((c.EXPLICIT_OTHER,),))
        self.assertEqual(c.result_label_for(None, 'Bash', {'command': 'cat CLAUDE.md'}, cwd)[0][:2], ('cat', c.EXPLICIT_INSTR))
        self.assertEqual(c.result_label_for(None, 'Bash', {'command': 'git status'}, cwd), ('git status',))
        self.assertEqual(c.result_label_for('srv', 'Read', {'file_path': 'CLAUDE.md'}, cwd), ())
        # a Codex exec script counts only with exactly one literal exec_command call
        one = 'await tools.exec_command({cmd: "cat AGENTS.md"})'
        self.assertEqual(c.result_label_for(None, 'exec', one, cwd)[0][1:], (c.EXPLICIT_INSTR, '…/repo/AGENTS.md'))
        self.assertEqual(c.result_label_for(None, 'exec', one + ';' + one, cwd), ('cat',))
        # any second exec_command call, literal or not, declines the attribution
        self.assertEqual(c.result_label_for(None, 'exec', one + ';await tools.exec_command({cmd: x})', cwd), ('cat',))
        # a literal workdir is the directory the command runs in; a name there may throw before the call (r18c)
        wd = 'await tools.exec_command({cmd: "cat AGENTS.md", workdir: "/nonexistent-home/other"})'
        self.assertEqual(c.result_label_for(None, 'exec', wd, cwd)[0][1:], (c.EXPLICIT_INSTR, '…/other/AGENTS.md'))
        self.assertEqual(c.result_label_for(None, 'exec', 'await tools.exec_command({cmd: "cat AGENTS.md", workdir: d})', cwd), (c.CMD_UNPARSED,))
        # a literal prefix of a computed value is not the value
        self.assertEqual(c.result_label_for(None, 'exec', 'await tools.exec_command({cmd: "cat AGENTS.md" + s})', cwd),
                         (c.CMD_UNPARSED,))
        self.assertEqual(c.result_label_for(
            None, 'exec', 'await tools.exec_command({cmd: "cat AGENTS.md", workdir: "/repo/" + n})', cwd),
            (c.CMD_UNPARSED,))
        # a repeated, spread, computed or shorthand key can override the literal cmd or workdir
        for obj in ('{cmd: "cat AGENTS.md", cmd: dynamic}', '{cmd: "cat AGENTS.md", ...opts}',
                    '{...opts, cmd: "cat AGENTS.md"}', '{cmd: "cat AGENTS.md", "cmd": "cat x"}',
                    "{cmd: \"cat AGENTS.md\", ['cmd']: x}", '{cmd: "cat AGENTS.md", [k]: x}',
                    '{cmd: "cat AGENTS.md", cmd}', '{cmd: "cat AGENTS.md", workdir: "/a", workdir: w}',
                    '{cmd: "cat AGENTS.md", get workdir() { return w }}',
                    "{cmd: \"cat AGENTS.md\", '\\x63md': dynamic}", '{cmd: "cat AGENTS.md", "\\u0063md": dynamic}',
                    '{cmd: "cat AGENTS.md", yield_time_ms: /[(]/.test(s), cmd: dynamic}',
                    '{cmd: "cat AGENTS.md", yield_time_ms: a / b}',
                    '{cmd: "cat AGENTS.md", yield_time_ms: `${f(`)`)}`, cmd: dynamic}',
                    # a brace or quote inside a string value does not end the object early
                    '{cmd:"cat AGENTS.md", note:"}", cmd:dynamic}', '{cmd:"cat AGENTS.md", note:"}", ...opts}',
                    "{cmd:\"cat AGENTS.md\", note:'\\'}', cmd:dynamic}", '{cmd:"cat AGENTS.md", note:`}`, cmd:dynamic}',
                    '{cmd:"cat AGENTS.md", note:`${"}"}`, cmd:dynamic}'):
            js = 'await tools.exec_command(' + obj + ')'
            # the label follows the value that wins at runtime; the attribution is declined either way
            self.assertEqual(c.result_label_for(None, 'exec', js, cwd), (c.tool_command('exec', js),), js)
            self.assertIn(c.tool_command('exec', js), ('cat', c.CMD_UNPARSED), js)
        self.assertEqual(c.tool_command('exec', 'await tools.exec_command({cmd:"cat AGENTS.md", note:"}", cmd:dynamic})'),
                         c.CMD_UNPARSED)
        # a script that ends inside a string, template or comment is not parsed
        for js in ('await tools.exec_command({cmd:"cat AGENTS.md"}); const s = "}',
                   'await tools.exec_command({cmd:"cat AGENTS.md"}); const s = `${"x"',
                   'await tools.exec_command({cmd:"cat AGENTS.md", note:"a\n"})',
                   'await tools.exec_command({cmd:"cat AGENTS.md"}); /*',
                   'const x = `${1 /* }`; await tools.exec_command({cmd:"cat AGENTS.md"})'):
            self.assertNotIn(c.EXPLICIT_INSTR, c.result_label_for(None, 'exec', js, cwd)[0], js)
        # a closed comment inside an interpolation hides its brace and backtick
        for js in ('const x = `${a /* } */ }`; await tools.exec_command({cmd:"cat AGENTS.md"})',
                   'const x = `${a /* ` */ }`; await tools.exec_command({cmd:"cat AGENTS.md"})'):
            assert_label_or_unparsed(self, exec_attribution(js, cwd), (c.EXPLICIT_INSTR, '…/repo/AGENTS.md'), js)
        # only one enclosing quote pair is removed from a key: a key with embedded quotes is neither cmd nor workdir
        js = 'await tools.exec_command({cmd:"cat AGENTS.md", "\'workdir\'":"/wrong"})'
        self.assertEqual(c.result_label_for(None, 'exec', js, cwd)[0][1:], (c.EXPLICIT_INSTR, '…/repo/AGENTS.md'))
        js = 'await tools.exec_command({cmd:"cat AGENTS.md", \'"workdir"\':"/wrong"})'
        self.assertEqual(c.result_label_for(None, 'exec', js, cwd)[0][1:], (c.EXPLICIT_INSTR, '…/repo/AGENTS.md'))
        for js in ('await tools.exec_command({"\'cmd\'":"cat AGENTS.md"})', "await tools.exec_command({'\"cmd\"':\"cat AGENTS.md\"})",
                   'await tools.exec_command({"\\"cmd\\"":"cat AGENTS.md"})'):
            self.assertEqual(c.result_label_for(None, 'exec', js, cwd), (c.CMD_UNPARSED,), js)
        assign = 'await tools.exec_command(Object.assign({cmd: "cat AGENTS.md"}, o))'
        self.assertNotIn(c.EXPLICIT_INSTR, c.result_label_for(None, 'exec', assign, cwd)[0])
        # quoted keys, other non-literal properties and a trailing comma do not hide cmd
        js = 'await tools.exec_command({"cmd": "cat AGENTS.md", yield_time_ms: f(1, 2), max_output_tokens: n,})'
        assert_label_or_unparsed(self, exec_attribution(js, cwd), (c.EXPLICIT_INSTR, '…/repo/AGENTS.md'))
        js = 'await tools.exec_command({"cmd":"cat AGENTS.md","workdir":' + json.dumps(cwd) + ',"max_output_tokens":4000})'
        self.assertEqual(c.result_label_for(None, 'exec', js, None)[0][1:], (c.EXPLICIT_INSTR, '…/repo/AGENTS.md'))
        # a malformed working directory leaves relative operands unplaced instead of raising
        self.assertEqual(c.result_label_for(None, 'Read', {'file_path': 'AGENTS.md'}, 42), ((c.EXPLICIT_OTHER,),))
        self.assertEqual(c.result_label_for(None, 'Bash', {'command': 'cat AGENTS.md'}, ['/x']), ('cat',))
        for bad in (42, 0, False, [], {}):
            self.assertEqual(c.result_label_for(None, 'shell', {'command': 'cat AGENTS.md', 'workdir': bad}, cwd), ('cat',), bad)
            self.assertEqual(c.result_label_for(None, 'shell', {'command': 'cat AGENTS.md', 'cwd': bad}, cwd), ('cat',), bad)
        # a shell argv runs its -c script as written
        self.assertEqual(c.result_label_for(None, 'shell', {'command': ['bash', '-lc', 'cat AGENTS.md']}, cwd)[0][1:],
                         (c.EXPLICIT_INSTR, '…/repo/AGENTS.md'))

    def test_method_labels_do_not_change_per_file_totals(self):
        """No double counting: splitting files by entry method only deepens leaf paths."""
        grp = (c.INSTR, 'Instruction files', 'Rules files')
        def evs(how):
            lab = (lambda f, m: (f, m)) if how else (lambda f, m: f)
            return [('B', grp, 700, 0, c.B_EST, lab('a.md', c.HOW_LAUNCH)), ('B', grp, 300, 0, c.B_EST, lab('b.md', c.HOW_LAUNCH)),
                    ('B', TYPED, 200, 0, c.B_EST), call(1500, out=10),
                    ('B', grp, 333, 0, c.B_EST, lab('a.md', c.HOW_ACCESS)), ('B', grp, 111, 0, c.B_EST, lab('b.md', c.HOW_IMPORT)),
                    call(2100, cr=1500, out=10), call(2200, cr=2100, out=5)]
        flat, _, _ = replay(evs(False))
        split, _, _ = replay(evs(True))
        folded = defaultdict(lambda: [0, 0, 0.0])
        for (path, r, m, k), v in split.items():
            key = (path[:4] if path[:3] == grp else path, r, m, k)
            for j in range(3):
                folded[key][j] += v[j]
        self.assertEqual({k: v[0] for k, v in folded.items()}, {k: v[0] for k, v in flat.items()})
        for k, v in flat.items():
            self.assertAlmostEqual(folded[k][1], v[1], places=6)
        self.assertIn((grp + ('a.md', c.HOW_ACCESS), 0, 'opus-5.5', 'top'), split)


class EntryMethodsEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_entry_methods_and_explicit_reads(self):
        repo = os.path.join(self.tmp, 'repo')
        os.makedirs(os.path.join(repo, '.git'))
        rule, claude_md = os.path.join(repo, '.claude/rules/git.md'), os.path.join(repo, 'CLAUDE.md')
        sub_md, imp = os.path.join(repo, 'sub/CLAUDE.md'), os.path.join(repo, 'docs/imp.md')

        def nested(h, path, parent=None):
            content = {'path': path, 'type': 'Project', 'content': SECRET + 'n' * 500}
            if parent:
                content['parent'] = parent
            return {'type': 'attachment', 'timestamp': ts(h), 'attachment': {'type': 'nested_memory', 'displayPath': path, 'content': content}}

        def tool(mid, h, cid, name, inp, cr, cw):
            o = claude_asst(mid, h, 10, cr, cw, 20)
            o['cwd'] = repo
            o['message']['content'] = [{'type': 'tool_use', 'id': cid, 'name': name, 'input': inp}]
            return o

        def result(cid, h, n):
            return {'type': 'user', 'timestamp': ts(h), 'message': {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': cid, 'content': SECRET + 'o' * n}]}}
        start = instr_attachment(3, [(rule, 800), (claude_md, 400)])
        start['attachment']['reason'] = 'session_start'
        compact_new = instr_attachment(2, [(rule, 800)])
        compact_new['attachment']['reason'] = 'compaction'
        lines = [claude_user('hi', 3), start, claude_asst('m1', 3, 10, 0, 1500, 20),
                 nested(2.9, sub_md), nested(2.9, imp, parent=sub_md),
                 tool('m2', 2.9, 't1', 'Read', {'file_path': rule}, 1500, 400),
                 result('t1', 2.9, 900),
                 tool('m3', 2.8, 't2', 'Bash', {'command': 'cat CLAUDE.md'}, 1900, 300),
                 result('t2', 2.8, 450),
                 tool('m4', 2.7, 't3', 'Read', {'file_path': os.path.join(repo, 'x.py')}, 2200, 200),
                 result('t3', 2.7, 300),
                 claude_asst('m5', 2.6, 10, 2400, 100, 20)]
        old = [{'type': 'system', 'subtype': 'compact_boundary', 'timestamp': ts(2.5)},
               instr_attachment(2.5, [(rule, 800)]), claude_asst('m6', 2.5, 10, 0, 300, 20)]
        new = [claude_user('hi', 2), compact_new, claude_asst('m7', 2, 10, 0, 300, 20)]
        codex = codex_lines(1)
        for o in codex:
            p = o['payload']
            if o['type'] == 'turn_context':
                p['cwd'] = repo
            if p.get('type') == 'function_call':
                p['arguments'] = json.dumps({'cmd': 'cat AGENTS.md'})
        rc, data = run_e2e(self.tmp, claude={'p/s.jsonl': lines + old, 'p/t.jsonl': new}, codex={'2026/r.jsonl': codex})
        self.assertEqual(rc, 0)
        self.assertTrue(data['meta']['reconciled'])
        self.assertTrue(data['meta']['oracle']['matched'])
        tree = data['tree']
        hows = lambda *labels: {n['label'] for n in find(tree, *labels)['children']}
        files = find(tree, c.INSTR, 'Instruction files')['children']
        by_fl = {}
        def walk(n):
            if n.get('fl'):
                by_fl[n['fl']] = n
            for ch in n.get('children', []):
                walk(ch)
        for n in files:
            walk(n)
        self.assertEqual({m['label'] for m in by_fl['repo/.claude/rules/git.md']['children']}, {c.HOW_LAUNCH, c.HOW_COMPACT})
        self.assertEqual({m['label'] for m in by_fl['repo/CLAUDE.md']['children']}, {c.HOW_LAUNCH})
        self.assertEqual({m['label'] for m in by_fl['repo/sub/CLAUDE.md']['children']}, {c.HOW_ACCESS})
        self.assertEqual({m['label'] for m in by_fl['repo/docs/imp.md']['children']}, {c.HOW_IMPORT})
        # explicit reads stay under Tool results and point back at the file
        read = find(tree, c.RESULTS, 'Built-in tools', 'Read')
        self.assertEqual({n['label'] for n in read['children']}, {c.EXPLICIT_INSTR, c.EXPLICIT_OTHER})
        self.assertEqual(find(read['children'], c.EXPLICIT_INSTR, 'repo/.claude/rules/git.md')['xf'], 'repo/.claude/rules/git.md')
        self.assertEqual(hows(c.RESULTS, 'Built-in tools', 'Bash', 'cat', c.EXPLICIT_INSTR), {'repo/CLAUDE.md'})
        self.assertEqual(hows(c.RESULTS, 'Built-in tools', 'exec_command', 'cat', c.EXPLICIT_INSTR), {'repo/AGENTS.md'})
        # no double counting: the read's tokens are not in the file node, and every leaf adds up to the total
        rule_read = find(read['children'], c.EXPLICIT_INSTR, 'repo/.claude/rules/git.md')
        self.assertGreater(sum(v[0] for v in node_cells(rule_read).values()), 0)
        total = sum(v[0] for n in tree for v in node_cells(n).values())
        self.assertEqual(total, sum(v[0] for v in node_cells({'children': tree}).values()))
        raw = ''
        for name in ('out.json', 'out.html'):
            with open(os.path.join(self.tmp, name)) as fh:
                raw += fh.read()
        for bad in (SECRET, self.tmp, '/Users/', '/home/'):
            self.assertNotIn(bad, raw)


    def test_malformed_cwd_keeps_usage_and_declines_attribution(self):
        """A non-string cwd must not drop the session; relative reads stay unattributed tool output."""
        for i, bad in enumerate((42, 0, [])):
            with self.subTest(cwd=bad):
                self._malformed_cwd_case(os.path.join(self.tmp, str(i)), bad)

    def _malformed_cwd_case(self, tmp, bad):
        repo = os.path.join(tmp, 'repo')
        os.makedirs(repo)
        asst = claude_asst('m1', 1, 100, 20, 0, 10)
        asst['cwd'] = bad
        asst['message']['content'] = [{'type': 'tool_use', 'id': 't1', 'name': 'Read', 'input': {'file_path': 'AGENTS.md'}}]
        bash = claude_asst('m2', 0.9, 100, 20, 0, 10)
        bash['cwd'] = {'path': repo}
        bash['message']['content'] = [{'type': 'tool_use', 'id': 't2', 'name': 'Bash', 'input': {'command': 'cat AGENTS.md'}}]
        result = lambda cid, h: {'type': 'user', 'timestamp': ts(h), 'message': {'role': 'user', 'content': [
            {'type': 'tool_result', 'tool_use_id': cid, 'content': 'o' * 300}]}}
        codex = codex_lines(1)
        for o in codex:
            p = o['payload']
            if o['type'] == 'turn_context':
                p['cwd'] = bad
            if p.get('type') == 'function_call':
                p['arguments'] = json.dumps({'cmd': 'cat AGENTS.md'})
        # a later malformed cwd forgets the earlier valid one rather than keep resolving against it
        codex.insert(1, {'timestamp': ts(1), 'type': 'turn_context', 'payload': {'model': 'gpt-6-luna', 'cwd': repo}})
        rc, data = run_e2e(tmp, claude={'p/s.jsonl': [asst, result('t1', 0.95), bash, result('t2', 0.85), claude_asst('m3', 0.8, 100, 500, 0, 10)]}, codex={'2026/r.jsonl': codex})
        self.assertEqual(rc, 0)
        self.assertTrue(data['meta']['oracle']['matched'])
        counts = data['meta']['counts']
        self.assertNotIn('claude_unparseable_files', counts)
        self.assertNotIn('codex_unparseable_files', counts)
        self.assertEqual(data['meta']['sessions'], 2)
        tree = data['tree']
        self.assertEqual({n['label'] for n in find(tree, c.RESULTS, 'Built-in tools', 'Read')['children']}, {c.EXPLICIT_OTHER})
        for tool in ('Bash', 'exec_command'):
            cat = find(tree, c.RESULTS, 'Built-in tools', tool, 'cat')
            self.assertNotIn(c.EXPLICIT_INSTR, {n['label'] for n in cat.get('children', [])}, tool)


BASH_RES = (c.RESULTS, 'Built-in tools', 'Bash')
BASH_CALL = (c.CALLS, 'Built-in tools', 'Bash')


class CommandLabels(unittest.TestCase):
    def check(self, cases):
        for cmd, want in cases:
            with self.subTest(cmd=cmd):
                self.assertEqual(c.command_label(cmd), want)

    def test_program_and_subcommand(self):
        self.check([
            ('git commit -m "secret message"', 'git commit'), ('gh pr view 123', 'gh pr'),
            ('docker compose up -d', 'docker compose'), ('kubectl get pods', 'kubectl get'),
            ('git -C /x/y status --short', 'git status'), ('pnpm --filter web test', 'pnpm test'),
            ('python3 -m pytest -q', 'python3 -m pytest'), ('python3 -c "print(1)"', 'python3 -c'),
            ('rg -n foo src/', 'rg'), ('curl https://example.com/token?x=1', 'curl'),
        ])

    def test_program_specific_option_arity(self):
        self.check([
            # Package values are arguments, even when the value is itself an allowlisted program.
            ('npx -p git eslint --version', 'npx eslint'),
            ('npx --package git eslint --version', 'npx eslint'),
            ('npx --package=git eslint --version', 'npx eslint'),
            ('pnpx -p git eslint --version', 'pnpx eslint'),
            ('pnpm -p git dlx eslint --version', 'pnpm dlx'),
            ('pnpm --package git dlx eslint --version', 'pnpm dlx'),
            ('pnpm --package=git dlx eslint --version', 'pnpm dlx'),
            ('pnpm dlx -p git eslint --version', 'pnpm dlx'),
            ('pnpm dlx --package=git eslint --version', 'pnpm dlx'),
            # Other commands keep their own -p meaning: git -p is a flag, cargo -p takes a value.
            ('git -p status', 'git status'), ('cargo -p private-package test', 'cargo test'),
        ])

    def test_wrappers_env_and_setup_segments(self):
        self.check([
            ('sudo -E env FOO=1 git status', 'git status'), ('FOO=bar BAZ=1 npm run build', 'npm run'),
            ('timeout 30 python3 -m pytest', 'python3 -m pytest'), ('nohup node server.js &', 'node'),
            ('time make all', 'make all'), ('xargs -0 rm', 'rm'), ('command -v jq', 'jq'),
            ('cd /repo && pnpm test --filter x', 'pnpm test'), ('export X=1; ls -la', 'ls'),
            ('true&&git push', 'git push'), ('cd /repo', 'cd'),
        ])

    def test_wrapper_options_with_values_and_equals(self):
        self.check([
            ('sudo -u root git status', 'git status'), ('sudo --user root git status', 'git status'),
            ('sudo --user=root git status', 'git status'),
            ('sudo -g wheel git status', 'git status'), ('sudo --group wheel git status', 'git status'),
            ('sudo --group=wheel git status', 'git status'),
            ('sudo -C 7 git status', 'git status'), ('sudo --close-from 7 git status', 'git status'),
            ('sudo --close-from=7 git status', 'git status'),
            ('sudo -D /Users/private/repo git status', 'git status'),
            ('sudo --chdir /Users/private/repo git status', 'git status'),
            ('sudo --chdir=/Users/private/repo git status', 'git status'),
            ('sudo -h host git status', 'git status'), ('sudo --host host git status', 'git status'),
            ('sudo --host=host git status', 'git status'),
            ('sudo -p prompt git status', 'git status'), ('sudo --prompt prompt git status', 'git status'),
            ('sudo --prompt=prompt git status', 'git status'),
            ('sudo -r role git status', 'git status'), ('sudo --role role git status', 'git status'),
            ('sudo --role=role git status', 'git status'),
            ('sudo -t type git status', 'git status'), ('sudo --type type git status', 'git status'),
            ('sudo --type=type git status', 'git status'),
            ('time -o /tmp/timing git status', 'git status'),
            ('time --output /tmp/timing git status', 'git status'),
            ('/usr/bin/time --output=/tmp/timing git status', 'git status'),
            ("time -f '%E' git status", 'git status'), ("time --format '%E' git status", 'git status'),
            ('time --format=%E git status', 'git status'),
            ('time -a -p -v git status', 'git status'),
            ('env -i FOO=bar git status', 'git status'), ('env --ignore-environment FOO=bar git status', 'git status'),
            ('env -u TOKEN git status', 'git status'), ('env --unset=TOKEN git status', 'git status'),
            ('env -C /Users/private/repo git status', 'git status'),
            ('env --chdir=/Users/private/repo git status', 'git status'),
            ("env -S 'git status'", 'git status'),
            ("env --split-string 'git status'", 'git status'),
            ("env --split-string='git status'", 'git status'),
            ("env -S 'TOKEN=value git status'", 'git status'),
        ])

    def test_wrapper_option_values_are_not_the_program(self):
        self.check([
            # The value of exec -a / xargs -a is an argument, even when it names an allowlisted program.
            ('exec -a git python -V', 'python'), ('exec -a /usr/bin/git python -V', 'python'),
            ('xargs -a git echo hello', 'echo'), ('xargs -a /usr/bin/git echo hello', 'echo'),
            ('xargs -agit echo hello', 'echo'), ('xargs --arg-file git echo hello', 'echo'),
            ('xargs --arg-file=git echo hello', 'echo'), ('xargs --arg-file=/x/git echo hello', 'echo'),
            ('xargs -0 -n1 -I{} grep x', 'grep'), ('nice -10 git status', 'git status'),
            ('nice -n 5 git status', 'git status'), ('stdbuf -oL git log', 'git log'),
            ('caffeinate -t 60 git pull', 'git pull'), ('timeout -- 5 git fetch', 'git fetch'),
            ('exec git status', 'git status'),
            # GNU xargs -e takes only an attached value, so the next word is the program.
            ('xargs -e git echo', 'git'),
            # So do the GNU long forms: --replace git status runs git, --replace=git echo runs echo.
            ('xargs --replace git status', 'git status'), ('xargs --replace={} git status', 'git status'),
            ('xargs --replace=git echo hi', 'echo'), ('xargs --eof git status', 'git status'),
            ('xargs --eof=git echo hi', 'echo'), ('xargs --max-lines git status', 'git status'),
            ('xargs --max-lines=1 git status', 'git status'),
        ])

    def test_unknown_wrapper_options_are_unparsed(self):
        self.check([
            ('xargs -q git status', c.CMD_UNPARSED), ('sudo --bogus git status', c.CMD_UNPARSED),
            ('timeout --nope 5 git fetch', c.CMD_UNPARSED), ('env --zzz git status', c.CMD_UNPARSED),
        ])

    def test_audited_program_option_tables(self):
        self.check([
            ('pnpm -w run build', 'pnpm run'), ('pnpm --workspace-root run build', 'pnpm run'),
            ('pnpm --workspace run build', 'pnpm run'), ('pnpm --reporter silent test', 'pnpm test'),
            ('npm --workspaces run test', 'npm run'),
            ('git --no-optional-locks status', 'git status'), ('git -P log', 'git log'),
            ('git --config-env=a=B status', 'git status'),
            ('codex --sandbox read-only exec x', 'codex exec'), ('codex -m gpt-6 exec x', 'codex exec'),
            ('codex -c k=v resume', 'codex resume'),
            ('docker -c ctx ps', 'docker ps'), ('docker -l debug ps', 'docker ps'),
            ('claude -p hi', 'claude'), ('claude --permission-mode plan -p hi', 'claude'),
        ])

    def test_pipes_paths_and_shell_structure(self):
        self.check([
            ('cat /etc/passwd | grep root', 'cat'), ('ls 2>&1|grep x', 'ls'), ('ls 2>/dev/null', 'ls'),
            ('/usr/local/bin/rg -n foo', 'rg'), ('./scripts/run.sh --flag', c.CMD_REJECTED),
            ('bash -lc "git log --oneline"', 'git log'), ("cat <<'EOF'\nREADME.md stuff\nEOF", 'cat'),
            ('f() { git status; }; f', 'git status'), ('f(){ ls; };f', 'ls'),
            ('function g { pnpm test; }; g', 'pnpm test'), ('ARR=(a b c); echo hi', 'echo'),
            ('case $x in a) echo 1;; esac', 'echo'), ('(cd x && ls)|wc', 'ls'),
        ])

    def test_argv_lists(self):
        self.assertEqual(c.command_label(['bash', '-lc', 'git diff HEAD']), 'git diff')
        self.assertEqual(c.command_label(['/bin/ls', '-la', '/tmp/private']), 'ls')
        self.assertEqual(c.command_label([]), c.CMD_UNPARSED)

    def test_privacy_rejects(self):
        self.check([
            ('README.md', c.CMD_REJECTED), ('exampleConfig.ts', c.CMD_REJECTED), ('4242', c.CMD_REJECTED),
            ('deadbeef01', c.CMD_REJECTED), ('ghp_abcdefghijklmnopqrstuvwxyz0123456789', c.CMD_REJECTED),
            ('sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAA', c.CMD_REJECTED), ('https://evil.example/x', c.CMD_REJECTED),
            ('/Users/someone/secret.txt', c.CMD_REJECTED), ('"quoted string"', c.CMD_REJECTED), ('-f', c.CMD_REJECTED),
            ('$CMD --go', c.CMD_UNPARSED), ('', c.CMD_UNPARSED), ('echo "unterminated', c.CMD_UNPARSED),
            ('git 1234567', 'git'), ('git deadbeefcafe', 'git'), ('FIXTURE_SECRET_TEXT.sh', c.CMD_REJECTED),
            ('./MyPrivateTool', c.CMD_REJECTED), ('python3 -m /tmp/x.py', 'python3 -m'),
            # user-named scripts, targets, modules and packages are not on the allowlists
            ('./private-client.sh', c.CMD_REJECTED), ('private-client --x', c.CMD_REJECTED),
            ('"private-client"', c.CMD_REJECTED), ('make private-client', 'make'),
            ('python3 -m private_client', 'python3 -m'), ('npx private-client', 'npx'),
            ('pnpm private-client', 'pnpm'), ('git private-client', 'git'),
            # an option argument is never taken for the subcommand
            ('docker --config private-client ps', 'docker ps'), ('docker --config private-client', 'docker'),
            ('docker --config login ps', 'docker ps'), ('docker --tlsdir private-client ps', 'docker ps'),
            # after an unknown option, two allowlisted words are ambiguous, so neither is shown
            ('docker --tlsdir login ps', 'docker'), ('git --no-pager log x', 'git log'),
            # a single word after an unknown option may be its argument ("login" as a certificate file name)
            ('docker --tlscacert login --help', 'docker'), ('docker --tlscacert login', 'docker'),
            ('git --no-pager status', 'git status'), ('pnpm --silent test', 'pnpm test'),
            ('npx vitest run', 'npx vitest'), ('python3.14 -m pytest', 'python3.14 -m pytest'),
        ])

    def test_labels_never_carry_arguments(self):
        cmds = ['git commit -m "FIXTURE_SECRET_TEXT"', 'curl -H "Authorization: Bearer tok" https://h/p',
                'cat ~/.ssh/id_rsa | base64', 'gh api repos/o/r/pulls/42', 'psql postgres://u:pw@h/db -c "select 1"',
                "node -e 'console.log(process.env.TOKEN)'", 'make deploy-prod-2026', 'git checkout feature/ABC-123-x']
        for cmd in cmds:
            lab = c.command_label(cmd)
            with self.subTest(cmd=cmd, label=lab):
                self.assertLessEqual(len(lab.split()), 3)
                words = lab.split()
                self.assertTrue(words[0] in c.PROGRAMS or c.PY_RE.match(words[0]) or lab == c.CMD_REJECTED, lab)
                for part in words:
                    self.assertNotRegex(part, r'[/:@~"\'=]')
                self.assertNotIn('SECRET', lab)
                self.assertNotIn('2026', lab)

    def test_tool_command_shapes(self):
        self.assertEqual(c.tool_command('exec', 'const r = await tools.exec_command({cmd:"rg -n foo"}); text(r)'), 'rg')
        self.assertEqual(c.tool_command('exec', "await tools.exec_command({cmd: 'git status'})"), 'git status')
        self.assertEqual(c.tool_command('exec', 'text("hello")'), c.CMD_NO_SHELL)
        self.assertEqual(c.tool_command('exec', '// tools.exec_command({cmd:"git status"})\n'
                                                '/* exec_command({cmd:"cat"}) */ await tools.exec_command({cmd:"ls"})'),
                         'ls')
        self.assertEqual(c.tool_command('exec', 'await tools.exec_command({cmd: "echo // not a comment"})'), 'echo')
        self.assertEqual(c.tool_command('exec', 'const x = "ls"; await tools.exec_command({cmd: x})'), 'ls')
        self.assertEqual(c.tool_command('exec', 'const x = f(); await tools.exec_command({cmd: x})'), c.CMD_UNPARSED)
        tc = lambda js: c.tool_command('exec', js)
        run = 'await tools.exec_command({cmd})'
        # rule 1: a cmd bound to a static literal resolves through its binding
        for js in ('const cmd = "git status"; ' + run, 'let cmd = `git status`; ' + run,
                   'var cmd = "git " + "status"; ' + run,
                   'for (const cmd of ["git status", "ls"]) ' + run,
                   '["git status", "ls"].map(cmd => tools.exec_command({cmd}))',
                   'const xs = ["git status"]; for (const x of xs) await tools.exec_command({cmd: x})',
                   'for (const [k, cmd] of Object.entries({a: "git status"})) ' + run,
                   'const o = {cmd: "git status", yield_time_ms: 1}; await tools.exec_command(o)',
                   'const steps = [{cmd: "git status"}]; for (const s of steps) await tools.exec_command({cmd: s.cmd})',
                   'await tools.exec_command({cmd: "git " + "status"})',
                   'await tools.exec_command({cmd: ["git", "status"]})'):
            assert_label_or_unparsed(self, tc(js), 'git status', js)
        # anything runtime-dependent stays Unparsed
        for js in ('const cmd = `git ${x}`; ' + run, 'const cmd = f(); ' + run, 'let cmd = "ls"; cmd = x; ' + run,
                   'function go(cmd) { return tools.exec_command({cmd}) }', 'const go = cmd => tools.exec_command({cmd})',
                   'for (const cmd of xs) ' + run, 'await tools.exec_command({cmd: x + "ls"})',
                   'await tools.exec_command({cmd: `${x} status`})', 'await tools.exec_command({cmd: "git" + x})',
                   'await tools.exec_command(opts)', 'const cmd = "ls"; { const cmd = f(); ' + run + ' }',
                   # an entry after cmd that may override it
                   'await tools.exec_command({cmd: "ls", ...rt})', 'await tools.exec_command({cmd: "ls", [k]: rt})',
                   # a binding mutated after its declaration
                   'const o = {cmd: "ls"}; o.cmd = rt; await tools.exec_command(o)',
                   'const o = {cmd: "ls"}; Object.assign(o, rt); await tools.exec_command(o)',
                   'const xs = ["ls"]; xs[0] = rt; for (const cmd of xs) ' + run,
                   'const xs = ["ls"]; xs.unshift(rt); for (const cmd of xs) ' + run,
                   'for (let cmd of ["ls"]) { cmd = rt; ' + run + ' }',
                   # a destructuring or for-of/in target writes the binding
                   'let cmd = "ls"; ({cmd} = rt); ' + run, 'let cmd = "ls"; [cmd] = rt; ' + run,
                   'let cmd = "ls"; for (cmd of xs) {} ' + run,
                   # a block-scoped declaration is not visible outside its block or for loop
                   'const cmd = rt; { const cmd = "ls"; } ' + run, 'for (let cmd = "ls"; f();) {} ' + run,
                   # enumeration of an object whose keys are not all static, or that reorders integer keys
                   'for (const cmd of Object.values({...rt, a: "ls"})) ' + run,
                   'for (const [k, cmd] of Object.entries({[k]: rt, a: "ls"})) ' + run,
                   'for (const cmd of Object.values({a: "ls", 1: rt})) ' + run,
                   # whitespace around the target or the for keyword does not hide a write or a loop scope
                   'let cmd = "ls"; [cmd]        = rt; ' + run, 'let cmd = "ls"; ({cmd}\n\n        = rt); ' + run,
                   'let cmd = "ls"; for          (cmd of xs) {} ' + run,
                   'for                 (let cmd = "ls"; f();) {}\n' + run,
                   # an unbraced compound loop body is not delimited, so its binding is not used
                   'for (let cmd = "ls"; f();) if (ok) {}\n' + run + ';',
                   'for (const cmd of ["ls"]) if (ok) {}\n' + run + ';',
                   'for (let cmd = "ls"; f();) tag:while (ok) {} ' + run,
                   # a computed key holding brackets, a catch parameter shadowing the binding
                   'const o = {cmd: "ls"}; o[["cmd"][0]] = rt; await tools.exec_command(o)',
                   'const cmd = "ls"; try { f(); } catch (cmd) { ' + run + ' }',
                   'const cmd = "ls"; try { f(); } catch ({cmd}) { ' + run + ' }',
                   # a parameter list mentioning the name shadows it, whatever the key form
                   'const cmd = "ls"; try { f(); } catch ({"value": cmd}) { ' + run + ' }',
                   'const cmd = "ls"; try { f(); } catch ({["value"]: cmd}) { ' + run + ' }',
                   'const cmd = "ls"; function g({"value": cmd}) { ' + run + ' }',
                   'const cmd = "ls"; const g = ({"value": cmd}) => { ' + run + ' }',
                   # writes through parentheses or another reference
                   'const o = {cmd: "ls"}; (o).cmd = "rg"; await tools.exec_command(o)',
                   'let cmd = "ls"; (cmd) = "rg"; ' + run,
                   'const o = {cmd: "ls"}; const alias = o; alias.cmd = "rg"; await tools.exec_command(o)',
                   'const xs = ["ls"]; const ys = xs; ys[0] = "rg"; for (const cmd of xs) ' + run,
                   'const o = {cmd: "ls"}; f(o); await tools.exec_command(o)',
                   'const o = {a: {cmd: "ls"}}; const i = o.a; i.cmd = "rg"; await tools.exec_command(o.a)',
                   # a callback that can leak the element, or one not recognised as read-only
                   'const xs = [{cmd: "ls"}]; const ys = xs.map((o) => o); ys[0].cmd = "rg"; '
                   'for (const o of xs) await tools.exec_command(o)',
                   'const xs = [{cmd: "ls"}]; xs.map((o, i, all) => f(all)); for (const o of xs) await tools.exec_command(o)',
                   'const xs = [{cmd: "ls"}]; xs.forEach(g); for (const o of xs) await tools.exec_command(o)',
                   'const xs = [["a", {cmd: "ls"}]]; xs.map(([k, o]) => f(o)); '
                   'for (const [k, o] of xs) await tools.exec_command(o)',
                   'const xs = [["a", {cmd: "ls"}]]; xs.map(([k, ...r]) => r); '
                   'for (const [k, o] of xs) await tools.exec_command(o)',
                   # a read that returns an inner reference, or a write through one
                   'const xs = [["a", {cmd: "ls"}]]; f(xs[0][1]); for (const [k, o] of xs) await tools.exec_command(o)',
                   'const xs = [["ls"]]; xs[0][0] = "rg"; for (const [cmd] of xs) ' + run,
                   'const xs = ["ls"]; xs.length = 0; for (const cmd of xs) ' + run,
                   'const o = {k: "a", a: {cmd: "ls"}}; f(o.a); await tools.exec_command(o.a)',
                   'const o = {k: "a", a: {cmd: "ls"}}; o.k.x = 1; f(o.k[0]); await tools.exec_command(o.a)',
                   # a spread or for-of element keeps references to nested objects
                   'const xs = [{cmd: "ls"}]; f([...xs]); await tools.exec_command({cmd: xs[0].cmd})',
                   'const xs = [{cmd: "ls"}]; for (const x of xs) f(x); await tools.exec_command({cmd: xs[0].cmd})',
                   'const xs = [{cmd: "ls"}]; for (const x of xs) if (a) tools.exec_command(x); else f(x);',
                   'const xs = [{cmd: "ls"}]; xs.map((o) => ({...o})); await tools.exec_command(xs[0])',
                   # a primitive __proto__ initializer is not an own property
                   'for (const cmd of Object.values({__proto__: "ls", a: "pwd"})) await tools.exec_command({cmd})'):
            self.assertEqual(tc(js), c.CMD_UNPARSED, js)
        # exec_command on another receiver is not the shell tool; a flat spread and a read-only loop are kept
        assert_label_or_unparsed(self, tc('other.exec_command({cmd: "ls"}); await tools.apply_patch("x")'), 'apply_patch')
        assert_label_or_unparsed(self, tc('other . exec_command({cmd: "ls"})'), c.CMD_NO_SHELL)
        self.assertEqual(tc('await tools . exec_command({cmd: "ls"})'), 'ls')
        for js in ('const xs = ["ls"]; const ys = [...xs]; await tools.exec_command({cmd: xs[0]})',
                   'const xs = [{cmd: "ls"}]; for (const x of xs) { await tools.exec_command(x); log(x.cmd) }'):
            assert_label_or_unparsed(self, tc(js), 'ls', js)
        # read-only callbacks, string positions and .length keep the binding
        for js in ('const xs = [{cmd: "ls"}]; await Promise.all(xs.map((o) => tools.exec_command(o)))',
                   'const xs = [{cmd: "ls"}]; xs.forEach((x, i) => log(x.cmd, i)); '
                   'for (const o of xs) await tools.exec_command(o)',
                   'const xs = [["a", "ls"]]; await Promise.all(xs.map(async ([k, cmd]) => '
                   '[k, await tools.exec_command({cmd})]))',
                   'const xs = [["a", {cmd: "ls"}]]; log(xs[0][0], xs.length); '
                   'for (const [k, o] of xs) await tools.exec_command(o)',
                   'const o = {k: "a", cmd: "ls", x: ["y"]}; log(o.k); await tools.exec_command(o)',
                   'const xs = [{k: "a", cmd: "ls"}]; xs.map(({k, cmd}) => f(k, cmd)); '
                   'for (const o of xs) await tools.exec_command(o)'):
            assert_label_or_unparsed(self, tc(js), 'ls', js)
        assert_label_or_unparsed(self, tc('await tools.exec_command({...rt, cmd: "ls"})'), 'ls')
        assert_label_or_unparsed(self, tc('const cmd = "ls"; { ' + run + ' }'), 'ls')
        assert_label_or_unparsed(self, tc('for (let cmd = "ls"; f();) { ' + run + ' }'), 'ls')
        assert_label_or_unparsed(self, tc('for (let cmd = "ls"; f();) ' + run), 'ls')
        assert_label_or_unparsed(self, tc('for await (const cmd of ["ls"]) ' + run), 'ls')
        assert_label_or_unparsed(self, tc('const cmd = "ls"; try { f(); } catch (e) { ' + run + ' }'), 'ls')
        assert_label_or_unparsed(self, tc('const o = {cmd: "ls"}; log(o.cmd); await tools.exec_command(o)'), 'ls')
        assert_label_or_unparsed(self, tc('for (const o of [{cmd: "ls"}]) await tools.exec_command(o)'), 'ls')
        assert_label_or_unparsed(self, tc('const o = {cmd: "ls"}; seen[o[["cmd"][0]]] = 1; await tools.exec_command(o)'), 'ls')
        assert_label_or_unparsed(self, tc('const [cmd] = ["ls"]; seen          [cmd] = 1; ' + run), 'ls')
        assert_label_or_unparsed(self, tc('const [cmd] = ["ls"]; seen[cmd] = 1; ' + run), 'ls')
        assert_label_or_unparsed(self, tc('for (const cmd of Object.values({a: "ls", b: rt})) ' + run), 'ls')
        assert_label_or_unparsed(self, tc('const x = `${a /* ` */ }`; // ' + run + '\ntext(1)'), c.CMD_NO_SHELL)
        # only the approved "git status " + value keeps a label; any other runtime value may hold any shell text
        self.assertEqual(tc('await tools.exec_command({cmd: "git status " + x})'), 'git status')
        self.assertEqual(tc('await tools.exec_command({cmd: `ls -l ${x}`})'), c.CMD_UNPARSED)
        # r18c: a runtime value can hold any shell text, so a label that words past it decide is Unparsed
        self.assertEqual(tc('await tools.exec_command({cmd: `cd ${d} && git show`})'), c.CMD_UNPARSED)
        self.assertEqual(tc('await tools.exec_command({cmd: "cd " + d + " && git status"})'), c.CMD_UNPARSED)
        self.assertEqual(tc('await tools.exec_command({cmd: `A=${x} git log`})'), c.CMD_UNPARSED)
        self.assertEqual(tc('await tools.exec_command({cmd: `cd x && ${t} run`})'), c.CMD_UNPARSED)
        assert_label_or_unparsed(self, tc('await tools.exec_command({cmd: `ls ${"}"} x`})'), 'ls')
        assert_label_or_unparsed(self, tc('await Promise.all(xs.map(async w => tools.exec_command('
                            '{cmd: `sed -n 1p ${w}`, workdir: "/tmp", yield_time_ms: 1})))'), 'sed')
        # an escaped \${ is literal text, so the template is static
        self.assertEqual(tc('await tools.exec_command({cmd: `echo \\${x}`})'), 'echo')
        # rule 2: a mention inside a string or comment is not a call
        for js in ('text("tools.exec_command({cmd: \'ls\'})")', '// await tools.exec_command({cmd: "ls"})\ntext(1)',
                   'const s = `use tools.exec_command({cmd})`; text(s)', '/* tools.mcp__t3_code__x() */ text(1)'):
            assert_label_or_unparsed(self, tc(js), c.CMD_NO_SHELL, js)
        assert_label_or_unparsed(self, tc('text("tools.exec_command"); await tools.exec_command({cmd: "ls"})'), 'ls')
        # rule 3: tool discovery without any call
        self.assertEqual(tc('text(ALL_TOOLS.map(t => t.name).join())'), c.TOOL_DISCOVERY)
        assert_label_or_unparsed(self, tc('text(ALL_TOOLS.length); await tools.exec_command({cmd: "ls"})'), 'ls')
        self.assertEqual(tc('text("ALL_TOOLS")'), c.CMD_NO_SHELL)
        # rule 4: MCP servers and apps come from a fixed list; the function name is never shown
        self.assertEqual(tc('await tools.mcp__t3_code__list_items({})'), 'MCP: t3_code')
        self.assertEqual(tc('await tools.mcp__codex_apps__slack_read_item({})'), 'MCP: slack')
        for js in ('await tools.mcp__codex_apps__unlisted_app_call({})', 'await tools.mcp__unlisted_server__call({})',
                   'await tools.mcp__({})'):
            self.assertEqual(tc(js), c.MCP_OTHER, js)
        # rule 5: known built-ins, prefixes, and unknown names
        self.assertEqual(tc('await tools.write_stdin({session_id: 1, chars: ""})'), 'write_stdin')
        self.assertEqual(tc('await tools.apply_patch("x")'), 'apply_patch')
        self.assertEqual(tc('await tools.web__search({q: "x"})'), 'web')
        self.assertEqual(tc('await tools.unknown_tool({})'), c.CMD_NO_SHELL)
        # precedence: a resolvable exec_command wins; otherwise the first labelled tools call
        assert_label_or_unparsed(self, tc('await tools.mcp__t3_code__x({}); await tools.exec_command({cmd: "ls"})'), 'ls')
        assert_label_or_unparsed(self, tc('await tools.mcp__t3_code__x({}); await tools.exec_command({cmd: f()})'), 'MCP: t3_code')
        self.assertEqual(tc('await tools.exec_command({cmd: f()}); await tools.mcp__t3_code__x({})'), c.CMD_UNPARSED)
        assert_label_or_unparsed(self, tc('await tools.unknown_tool({}); await tools.web__open({})'), 'web')
        # rule 6: added subcommands and programs, arithmetic loops, and a variable program
        self.assertEqual(c.command_label('pnpm vitest run x'), 'pnpm vitest')
        self.assertEqual(c.command_label('mise run build'), 'mise run')
        self.assertEqual(c.command_label('for ((i=0;i<3;i++)); do ls; done'), 'ls')
        self.assertEqual(c.command_label('(( n++ )) && git status'), 'git status')
        self.assertEqual(c.command_label('n=$((n+1)); git status'), 'git status')
        # "( (" opens nested subshells, not arithmetic
        self.assertEqual(c.command_label('x=$( (git diff; ls) | wc -l)'), 'git diff')
        self.assertEqual(c.command_label('( (git diff) )'), 'git diff')
        self.assertEqual(c.command_label('for f in a b; do git status; done'), 'git status')
        self.assertEqual(c.command_label('$TOOL run x'), c.CMD_UNPARSED)
        self.assertEqual(c.tool_command('Bash', {'command': 'git status', 'description': 'x'}), 'git status')
        self.assertEqual(c.tool_command('exec_command', '{"cmd": "cargo build --release"}'), 'cargo build')
        self.assertEqual(c.tool_command('shell', {'command': ['bash', '-lc', 'pnpm lint']}), 'pnpm lint')
        self.assertEqual(c.tool_command('Bash', {}), c.CMD_UNPARSED)
        self.assertEqual(c.cmd_label_for(None, 'Bash', {'command': 'ls'}), ('ls',))
        self.assertEqual(c.cmd_label_for('srv', 'Bash', {'command': 'ls'}), ())
        self.assertEqual(c.cmd_label_for(None, 'Read', {'file_path': '/x'}), ())

    def test_model_provider(self):
        for m, want in [('opus-5.5', 'anthropic'), ('haiku-4.5', 'anthropic'), ('fable-5.1', 'anthropic'),
                        ('claude-unknown-0', 'anthropic'), ('gpt-6-luna', 'openai'), ('gpt-6.1-sol', 'openai'),
                        ('o3', 'openai'), ('mystery', 'other'), ('', 'other')]:
            self.assertEqual(c.model_provider(m), want, m)


class CommandSplit(unittest.TestCase):
    def events(self, labelled):
        lab = lambda x: (x,) if labelled else ()
        b = lambda n, x: ('B', BASH_RES, n, 0, c.B_EST) + lab(x)
        return [
            ('B', TYPED, 50, 0, c.B_EST),
            call(500, out=30, outs=[(BASH_CALL, 10, 0) + lab('git status'), (BASH_CALL, 20, 0) + lab('cat')]),
            b(400, 'git status'), b(900, 'cat'),
            call(1900, cr=480, out=12, outs=[(BASH_CALL, 12, 0) + lab('git status')], t=1001.0),
            b(250, 'git status'),
            call(2200, cr=1900, t=1002.0, model='gpt-6-luna'),
            ('R',),
            b(40, 'rg'),
            call(80, t=1003.0),
        ]

    def test_commands_sum_exactly_to_the_tool(self):
        whole, _, _ = replay(self.events(False))
        split, _, _ = replay(self.events(True))
        for group, kids in ((BASH_RES, {'git status', 'cat', 'rg'}), (BASH_CALL, {'git status', 'cat'})):
            parts = {k: v for k, v in split.items() if k[0][:3] == group and len(k[0]) == 4}
            self.assertEqual({k[0][3] for k in parts}, kids)
            for model in ('opus-5.5', 'gpt-6-luna'):
                g = whole.get((group + (c.CMD_UNPARSED,), 0, model, 'top'))
                if g is None:  # the gpt call bills only results
                    self.assertTrue(group == BASH_CALL and model == 'gpt-6-luna', (group, model))
                    continue
                mine = [v for k, v in parts.items() if k[2] == model]
                self.assertEqual(sum(v[0] for v in mine), g[0], (group, model))
                self.assertAlmostEqual(sum(v[1] for v in mine), g[1], places=6)
                self.assertAlmostEqual(sum(v[2] for v in mine), g[2], places=9)
        for k, v in split.items():  # everything outside the shell tools is unchanged
            if len(k[0]) != 4:
                self.assertEqual(v[:2], whole[k][:2])

    def test_unlabelled_shell_events_land_in_unparsed(self):
        whole, _, _ = replay(self.events(False))
        res = {k[0] for k in whole if k[0][:3] == BASH_RES}
        self.assertEqual(res, {BASH_RES + (c.CMD_UNPARSED,)})

    def test_fold_keeps_unparsed_and_folds_rejected(self):
        acc, basis = {}, defaultdict(set)
        rows = [('git status', 500), ('cat', 400), ('rg', 300), ('ls', 200), ('sed', 100),
                (c.CMD_UNPARSED, 5), (c.CMD_NO_SHELL, 3), (c.CMD_REJECTED, 50)]
        for lab, n in rows:
            for mi, m in enumerate(('opus-5.5', 'gpt-6-luna')):
                acc[(BASH_RES + (lab,), 0, m, 'top')] = [n + mi, 1, 0.5]
                basis[BASH_RES + (lab,)].add(c.B_EST)
        tree = c.build_tree(acc, ['opus-5.5', 'gpt-6-luna'], ['top'], basis, top_cmds=3)
        bash = find(tree, c.RESULTS, 'Built-in tools', 'Bash')
        labels = [n['label'] for n in bash['children']]
        # the folded bucket sorts by its total like any other child
        self.assertEqual(labels, ['git status', 'cat', 'Other (3 commands)', 'rg', c.CMD_UNPARSED, c.CMD_NO_SHELL])
        other = find(bash['children'], 'Other (3 commands)')
        self.assertEqual(leaf_cell(other), [200 + 100 + 50, 3, 1.5])
        self.assertEqual(leaf_cell(other, m=1), [201 + 101 + 51, 3, 1.5])
        # per model, the children add up to the node and to the raw accumulator
        for m in (0, 1):
            self.assertEqual(sum(cell[(0, m, 0)][0] for cell in [node_cells(bash)]),
                             sum(v[0] for k, v in acc.items() if k[2] == ('opus-5.5', 'gpt-6-luna')[m]))
        only_rejected = c.build_tree({(BASH_RES + (c.CMD_REJECTED,), 0, 'm', 'top'): [7, 1, 0.0],
                                      (BASH_RES + ('ls',), 0, 'm', 'top'): [9, 1, 0.0]},
                                     ['m'], ['top'], basis, top_cmds=3)
        self.assertEqual([n['label'] for n in find(only_rejected, c.RESULTS, 'Built-in tools', 'Bash')['children']],
                         ['ls', c.CMD_REJECTED])


    def test_serialized_children_add_up_to_the_unsplit_leaf(self):
        # unique 0.6 + 0.6 would round to 2 per leaf; the unsplit leaf serializes 1.2 -> 1 (likewise $0.00003 x2)
        basis = defaultdict(set)
        split = {(BASH_RES + ('cat',), 0, 'm', 'top'): [5, 0.6, 0.00003],
                 (BASH_RES + ('ls',), 0, 'm', 'top'): [5, 0.6, 0.00003],
                 (BASH_RES + ('rg',), 0, 'm', 'top'): [5, 0.6, 0.00003]}
        for k in split:
            basis[k[0]].add(c.B_EST)
        tree = c.build_tree(split, ['m'], ['top'], basis)
        cells = [leaf_cell(n) for n in find(tree, c.RESULTS, 'Built-in tools', 'Bash')['children']]
        self.assertEqual(sum(x[0] for x in cells), 15)
        self.assertEqual(sum(x[1] for x in cells), 2)  # round(1.8)
        self.assertAlmostEqual(sum(x[2] for x in cells), 0.0001, places=9)  # round(0.00009, 4)

    def test_half_cent_boundaries_round_like_the_unsplit_leaf(self):
        def bash_cells(cells):
            basis = defaultdict(set)
            split = {(BASH_RES + (lab,), 0, 'm', 'top'): cell for lab, cell in cells}
            for k in split:
                basis[k[0]].add(c.B_EST)
            tree = c.build_tree(split, ['m'], ['top'], basis)
            kids = find(tree, c.RESULTS, 'Built-in tools', 'Bash')['children']
            return [leaf_cell(n) for n in kids if any(leaf_cell(n))]
        # compact() serializes an unsplit $0.00005 as 0.0001; a single split child must match
        for cell, usd in (([1, 1, 0.00005], 0.0001), ([1, 1, -0.00005], -0.0001), ([1, 2.5, 0.00015], 0.0001)):
            out = bash_cells([('cat', cell)])
            self.assertAlmostEqual(sum(x[2] for x in out), usd, places=9)
            self.assertAlmostEqual(sum(x[2] for x in out), round(cell[2], 4), places=9)
            self.assertEqual(sum(x[1] for x in out), round(cell[1]))
        out = bash_cells([('cat', [1, 0.5, 0.00002]), ('ls', [1, 0.5, 0.00003])])
        self.assertAlmostEqual(sum(x[2] for x in out), round(0.00005, 4), places=9)
        self.assertEqual(sum(x[1] for x in out), round(1.0))


class CommandEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_bash_and_exec_have_private_command_children(self):
        def tool_use(mid, h, cid, cmd, model='claude-opus-5-5', cr=0, cw=3000):
            o = claude_asst(mid, h, 10, cr, cw, 20, model=model)
            o['message']['content'] = [{'type': 'tool_use', 'id': cid, 'name': 'Bash',
                                        'input': {'command': cmd, 'description': SECRET}}]
            return o

        def result(cid, h, n):
            return {'type': 'user', 'timestamp': ts(h), 'message': {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': cid, 'content': SECRET + 'o' * n}]}}
        lines = [claude_user('hi', 3),
                 tool_use('m1', 3, 't1', 'cd /Users/someone/repo && git status --short'), result('t1', 3, 4000),
                 tool_use('m2', 2.9, 't2', 'cat ' + SECRET + '.txt | head', cr=3000, cw=1500), result('t2', 2.9, 2000),
                 tool_use('m3', 2.8, 't3', '/Users/someone/' + SECRET + '.sh', cr=4500, cw=900), result('t3', 2.8, 900),
                 claude_asst('m4', 2.7, 10, 5400, 100, 30)]
        codex = codex_lines(2)
        for o in codex:
            p = o['payload']
            if p.get('type') == 'function_call':
                p['arguments'] = json.dumps({'cmd': 'rg -n ' + SECRET + ' /Users/someone'})
        rc, data = run_e2e(self.tmp, claude={'p/s.jsonl': lines}, codex={'2026/r.jsonl': codex})
        self.assertEqual(rc, 0)
        self.assertTrue(data['meta']['reconciled'])
        self.assertEqual({m['id']: m['provider'] for m in data['models']}['gpt-6-luna'], 'openai')
        self.assertEqual({m['id']: m['provider'] for m in data['models']}['opus-5.5'], 'anthropic')
        bash = find(data['tree'], c.RESULTS, 'Built-in tools', 'Bash')
        self.assertEqual({n['label'] for n in bash['children']}, {'git status', 'cat', c.CMD_REJECTED})
        ex = find(data['tree'], c.RESULTS, 'Built-in tools', 'exec_command')
        self.assertEqual([n['label'] for n in ex['children']], ['rg'])
        raw = ''
        for name in ('out.json', 'out.html'):
            with open(os.path.join(self.tmp, name)) as fh:
                raw += fh.read()
        for bad in (SECRET, '/Users/someone', '--short'):
            self.assertNotIn(bad, raw)


class ReviewR13(unittest.TestCase):
    """r12 F1 (receiver across whitespace/comments), r12 F2 (regex literals), roles from data."""
    cwd = '/nonexistent-home/repo'

    def test_receiver_across_whitespace_and_comments(self):
        tc = lambda js: c.tool_command('exec', js)
        call = 'exec_command({cmd:"cat AGENTS.md"})'
        for recv in ('other .   ', 'other /*x*/ . \n ', 'other\n\t.\n', 'a?.', 'a ?. ', 'a\n?.\n',
                     'x . tools.', 'x /*c*/ .tools . ', 'other // c\n.'):
            js = recv + call
            assert_label_or_unparsed(self, tc(js), c.CMD_NO_SHELL, js)
            assert_label_or_unparsed(self, c.result_label_for(None, 'exec', js, self.cwd), (c.CMD_NO_SHELL,), js)
        # tools.<other tool> on a property tools is not a tool call either
        self.assertIn(c._exec_script_label('x . tools.apply_patch("p")'), (None, c.CMD_UNPARSED))
        # bare and tools. calls, across whitespace, newlines and comments, still count
        for recv in ('', 'await ', 'tools.', 'tools /*x*/ . \n ', 'tools\n.', '...', 'f(); '):
            js = recv + call
            assert_label_or_unparsed(self, tc(js), 'cat', js)
            assert_label_or_unparsed(self, exec_attribution(js, self.cwd), (c.EXPLICIT_INSTR, '…/repo/AGENTS.md'), js)

    def test_regex_literals_are_masked(self):
        tc = lambda js: c.tool_command('exec', js)
        for js in ('text(/tools.exec_command({cmd:"ls"})/.source)',
                   'const r = /tools.exec_command({cmd:"ls"})/g; text(r)',
                   'x = cond ? /exec_command({cmd:"ls"})/ : /b/',
                   'if (/[/]tools.exec_command({cmd:"ls"})/.test(s)) text(1)',
                   'return /\\/tools.exec_command({cmd:"ls"})/',
                   'if (ok) /exec_command({cmd:"ls"})/.test(s)',
                   'while (f(a)) /tools.exec_command({cmd:"ls"})/.test(s)'):
            self.assertNotEqual(tc(js), 'ls', js)
            self.assertEqual(len(c.result_label_for(None, 'exec', js.replace('ls', 'cat AGENTS.md'), self.cwd)), 1, js)
        # a quote or comment opener inside a regex does not shift the masking
        assert_label_or_unparsed(self, tc('const r = /"/; await tools.exec_command({cmd:"ls"})'), 'ls')
        assert_label_or_unparsed(self, tc("const r = /'[`]/; await tools.exec_command({cmd:'pwd'})"), 'pwd')
        assert_label_or_unparsed(self, tc('const r = /a[//]b/; await tools.exec_command({cmd:"ls"})'), 'ls')
        # division is not a regex
        for js in ('a / b; tools.exec_command({cmd:"ls"})', 'const q = a / b / c; tools.exec_command({cmd:"ls"})',
                   'f(x) / 2; tools.exec_command({cmd:"ls"}) / 3', 'xs[0] / 2 + i++ / 4; tools.exec_command({cmd:"ls"})',
                   '"s".length / 2; tools.exec_command({cmd:"ls"})'):
            assert_label_or_unparsed(self, tc(js), 'ls', js)
        # after a one-line comment the '/' still divides; after a newline, a control-flow head or any '}'
        # the scan stops
        js = 'x = a /* c */ / 2; tools.exec_command({cmd:"ls /tmp"})'
        assert_label_or_unparsed(self, tc(js), 'ls', js)
        res = c.result_label_for(None, 'exec', js.replace('ls /tmp', 'cat AGENTS.md'), self.cwd)
        assert_label_or_unparsed(self, res, (('cat', c.EXPLICIT_INSTR, '…/repo/AGENTS.md'),), js)
        for js in ('if (ok) /"/.test(s); tools.exec_command({cmd:"ls /tmp"})',
                   'for (;;) /`/.test(s); tools.exec_command({cmd:"ls /tmp"})',
                   '{} /"/.test(s); tools.exec_command({cmd:"ls /tmp"})',
                   'const q = {} / 2; tools.exec_command({cmd:"ls /tmp"})',
                   'x = {a: {}} / 2; tools.exec_command({cmd:"ls /tmp"})',
                   'return {} / 2; tools.exec_command({cmd:"ls /tmp"})',
                   'const o = {a: "' + 'x' * 600 + '"} / 2; tools.exec_command({cmd:"ls /tmp"})',
                   'x = {a: ")"} / 2; tools.exec_command({cmd:"ls /tmp"})',
                   'const q = function() {} / 2; tools.exec_command({cmd:"ls /tmp"})',
                   'const q = async function* g() {} / 2; tools.exec_command({cmd:"ls /tmp"})',
                   'function f() {} /"/.test(s); tools.exec_command({cmd:"ls /tmp"})',
                   'const C = class A extends B.C { m() {} } / 2; tools.exec_command({cmd:"ls /tmp"})',
                   'function f() { return {} / 2; } tools.exec_command({cmd:"ls /tmp"})',
                   'x = a ? b : {} / 2; tools.exec_command({cmd:"ls /tmp"})'):
            assert_label_or_unparsed(self, tc(js), c.CMD_NO_SHELL, js)
            res = c.result_label_for(None, 'exec', js.replace('ls /tmp', 'cat AGENTS.md'), self.cwd)
            assert_label_or_unparsed(self, res, (c.CMD_NO_SHELL,), js)

    def test_slash_after_control_heads_and_blocks_stops(self):
        tc = lambda js: c.tool_command('exec', js)
        pwd = 'tools.exec_command({cmd:"pwd"})'
        for head in ('if (s === "(")', 'if (' + ' ' * 600 + 'ok)', 'if' + ' ' * 600 + '(ok)', 'while (f(")"))',
                     'function f() {}', 'class A { m() {} }', 'label: {}', 'switch (x) { case 1: {}',
                     'x = a?.b ? {} : 1; {}', 'const f = () => {}', 'function f() { return\n{}',
                     'for await (const x of xs)', 'while (ok) { break\n'):
            js = head + ' /tools.exec_command({cmd:"ls"})/.test(s);\n' + pwd
            assert_label_or_unparsed(self, tc(js), c.CMD_NO_SHELL, head[:20])
            res = c.result_label_for(None, 'exec', js.replace('"pwd"', '"cat AGENTS.md"'), self.cwd)
            assert_label_or_unparsed(self, res, (c.CMD_NO_SHELL,), head[:20])

    def test_regex_in_interpolation_and_comment_separators(self):
        tc = lambda js: c.tool_command('exec', js)
        for head in ('text(`value: ${s.replace(/"/g, "")}`);\nawait ', 'text(`v ${s.replace(/`/g, "")}`);\nawait ',
                     'text(`${a / 2} ${ {a: 1}.a }`); ', 'await/* comment */', 'await/*\n*/'):
            js = head + 'tools.exec_command({cmd:"pwd"})'
            assert_label_or_unparsed(self, tc(js), 'pwd', head[:20])
            res = c.result_label_for(None, 'exec', js.replace('"pwd"', '"cat AGENTS.md"'), self.cwd)
            assert_label_or_unparsed(self, res, (('cat', c.EXPLICIT_INSTR, '…/repo/AGENTS.md'),), head[:20])

    def test_unknown_role_profiles_split_and_label(self):
        for role, label in (('envoy-peer', 'Envoy peer'), ('zz-made-up9', 'Zz made up9'), ('review', 'Review')):
            tag = ROLE_TAG % role
            parts = dict(c.split_text('A' * 7 + tag, 'Typed prompts'))
            self.assertEqual(parts, {(c.USER, 'Typed prompts'): 7, (c.INSTR, 'Subagent role profile', label): len(tag)})
        # not a role slug: stays prompt text and never becomes a label
        for bad in ('Envoy', 'x' * 33, 'a_b', 'a b', '-a', '1a', 'a/b', '', 'a"b'):
            tag = ROLE_TAG % bad
            self.assertEqual(dict(c.split_text(tag, 'Typed prompts')), {(c.USER, 'Typed prompts'): len(tag)}, bad)
        self.assertEqual(c.role_order({'top', 'zeta', 'envoy-peer'}),
                         ['top', 'envoy-peer', 'zeta'])

    def test_unknown_role_sessions(self):
        tmp = tempfile.mkdtemp()
        try:
            p = os.path.join(tmp, 'a.jsonl')
            write_jsonl(p, [claude_user('task ' + ROLE_TAG % 'envoy-peer', 1), claude_asst('m1', 1, 10, 0, 0, 5)])
            self.assertEqual(c.parse_claude(p, True)[0]['role'], 'envoy-peer')
            write_jsonl(p, [claude_user('task ' + ROLE_TAG % 'Bad Role', 1), claude_asst('m1', 1, 10, 0, 0, 5)])
            self.assertEqual(c.parse_claude(p, True)[0]['role'], 'native')
            write_jsonl(p, codex_lines(1, role='envoy-peer'))
            self.assertEqual(c.parse_codex(p)[0]['role'], 'envoy-peer')
            write_jsonl(p, codex_lines(1, role='x' * 40))
            self.assertEqual(c.parse_codex(p)[0]['role'], 'top')
            rc, data = run_e2e(os.path.join(tmp, 'e2e'), claude={'proj/s.jsonl': [
                claude_user('task ' + ROLE_TAG % 'zz-made-up', 1), claude_asst('m1', 1, 10, 0, 0, 5)]},
                codex={'2026/r.jsonl': codex_lines(1, role='envoy-peer')})
            self.assertEqual(rc, 0)
            self.assertTrue(data['meta']['oracle']['matched'])
            self.assertEqual(data['roles'], [{'id': 'envoy-peer', 'label': 'Envoy peer'},
                                             {'id': 'zz-made-up', 'label': 'Zz made up'}])
            self.assertEqual(data['meta']['sessions_by_role'], {'claude:zz-made-up': 1, 'codex:envoy-peer': 1})
            roles = {k[2] for n in data['tree'] for k, v in node_cells(n).items() if v[0]}
            self.assertEqual(roles, {0, 1})
        finally:
            shutil.rmtree(tmp)



class ReviewR14(unittest.TestCase):
    """r13 F1 (prefix ++/--, labelled break/continue), r13 F2 (numeric literals), undecided slashes."""
    cwd = '/nonexistent-home/repo'
    call = 'tools.exec_command({cmd:"cat AGENTS.md"})'
    hit = (('cat', c.EXPLICIT_INSTR, '…/repo/AGENTS.md'),)

    def check(self, js, label, attributed):
        js = js.replace('CALL', self.call)
        got = c.tool_command('exec', js)
        if got == c.CMD_UNPARSED:  # r18 may miss a label r17 gave, never give another one
            label, attributed = got, False
        self.assertEqual(got, label, js)
        res = c.result_label_for(None, 'exec', js, self.cwd)
        self.assertEqual(res, self.hit if attributed else (label,), js)

    def test_regex_after_prefix_update_and_statement_ends(self):
        for js in ('++/CALL/.lastIndex;', '--/CALL/.lastIndex;', 'x = 1; ++/CALL/.lastIndex;',
                   'x\n++/CALL/.lastIndex;', 'a + ++/CALL/.lastIndex;', 'f(--/CALL/.lastIndex)',
                   'done: while (true) { break done\n /CALL/.lastIndex; }',
                   'done: while (true) { continue done\n /CALL/.lastIndex; }', 'debugger\n/CALL/.lastIndex',
                   'if (a) /CALL/.test(s)', 'const q = () => {}\n/CALL/.test(s)', '{} /CALL/.test(s)',
                   'for (const x of /CALL/.exec(s)) f(x)', 'typeof /CALL/', 'void /CALL/', 'delete /CALL/.x',
                   'x in /CALL/', 'x instanceof /CALL/', 'for (const of of /CALL/.source) {}',
                   'for (const x of of /CALL/) {}', 'x\r++/CALL/.lastIndex;', 'x\u2028++/CALL/.lastIndex;',
                   'x\u2029--/CALL/.lastIndex;', 'x /* a\rb */ ++/CALL/.lastIndex;',
                   'done: while (true) { break done\r/CALL/.lastIndex; }'):
            self.check(js, c.CMD_NO_SHELL, False)

    def test_division_after_operands_still_labels(self):
        for js in ('1. / CALL / 2;', '.5 / CALL / 2;', '1.5 / CALL / 2;', '1e3 / CALL / 2;', '1e-3 / CALL / 2;',
                   '1.e+3 / CALL / 2;', '.5e-3 / CALL / 2;', '0x1F / CALL / 2;', '1_000 / CALL / 2;',
                   '10n / CALL / 2;', 'i++ / CALL / 2;', 'i-- / CALL / 2;', 'a[0]++ / CALL / 2;',
                   'a+++b / CALL / 2;', 'x = 1..toString().length / CALL / 2;', 'this / CALL / 2', 'x-->0; CALL',
                   'const q = (() => {}) / CALL / 2;'):
            self.check(js, 'cat', True)
        # 'of', '}' and a newline before a dividing '/' stop the scan: a missed label
        for js in ('a++\r/ CALL / 2', 'a\u2028/ CALL / 2', 'const of = 4; of / CALL / 2', 'for (x in of / CALL / 2) f(x)',
                   'const q = {if() {}, for() {}} / CALL / 2;', 'x = {} / CALL / 2'):
            self.check(js, c.CMD_NO_SHELL, False)

    def test_undecided_stops_the_scan(self):
        # the scan stops where it cannot tell regex from division, or code from a comment
        for js in ('yield /CALL/', 'await /CALL/', 'await\n/CALL/', 'yield ++/CALL/.x', 'x = /CALL',
                   'x = /tools.exec_command({cmd:"ls"}) ; CALL', '\\u0061 /CALL/', 'x = ) / CALL / 2',
                   'x = a <!-- b\nCALL', 'x\n--> CALL', '--> CALL', 'text(`${await /CALL/}`)'):
            self.check(js, c.CMD_NO_SHELL, False)
        # a call before that point still names the command, without attribution
        self.check('CALL; await /x/', 'cat', False)
        self.check('CALL; x = /y', 'cat', False)


class ReviewR15(unittest.TestCase):
    """r14 F1 (class expression bodies) and the whitelist of contexts where a '/' is classified."""
    cwd = '/nonexistent-home/repo'
    call = 'tools.exec_command({cmd:"cat AGENTS.md"})'
    hit = (('cat', c.EXPLICIT_INSTR, '…/repo/AGENTS.md'),)
    check = ReviewR14.check

    REGEX = ('', 'f(', '[', '{', 'a,', ';', 'a ? b :', 'a ?', 'x =', 'x +=', 'x **=', 'x ??=', 'x >>>=', 'x /=',
             'a +', 'a -', 'a *', 'a %', 'a **', 'a &', 'a |', 'a ^', '!', '~', 'a <', 'a >', 'a <=', 'a >=',
             'a ==', 'a !=', 'a ===', 'a !==', 'a &&', 'a ||', 'a ??', '() =>', 'a + +', 'a - -', 'a\n+',
             'return', 'return\n', 'typeof', 'void', 'delete', 'new', 'throw', 'case', 'x in', 'x instanceof',
             '[...typeof')
    DIVISION = ('a', '$', '_x', 'x.return', 'x?.typeof', 'x.\nin', 'this.#in', 'a?.#new', '1.', '.5',
                '0x1F', '10n', '"s"', "'s'", '`t`', '/re/g', 'a[0]', 'f()', '(() => {})', 'x.if(a)',
                'a++', 'a--', 'a ++', 'this', 'null', 'true', 'false', 'async', 'get', 'x.yield', 'x.of')
    STOP = ('if (a)', 'while (a)', 'for (;;)', 'with (a)', 'for await (x of y)', 'if (a)\n', '{}', 'x = {}',
            'f() {}', 'x = class extends A {}', '() => {}', '++', 'x\n++', '--', 'a + ++', 'yield', 'await',
            'x of', 'break done', 'continue done', 'break', 'continue', 'debugger', 'do', 'else', 'let', 'static',
            'super', 'extends', 'default', 'a.', '...', '#x', '@x', 'x = )', 'x = ]', '@', 'x }', 'a\n', 'f(x)\n')

    def test_class_expression_bodies(self):
        for js in ('const f = class extends mixin(A) {} / /x; CALL/.source;',
                   'const f = class extends (A) {} / /x; CALL/.source;',
                   'const f = class extends a.b.c.d.e.f.g.h.i.j.k.l {} / /x; CALL/.source;',
                   'const f = class A extends mixin(A) { m() {} } / /x; CALL/.source;',
                   'x = class {} / /x; CALL/.source;'):
            self.check(js, c.CMD_NO_SHELL, False)
        self.check('CALL; const f = class extends mixin(A) {} / /x/.source;', 'cat', False)

    def test_whitelist(self):
        for p in self.REGEX:
            self.check(p + ' /CALL/', c.CMD_NO_SHELL, False)
            self.check(p + ' /x/; CALL', 'cat', True)
        for p in self.DIVISION:
            self.check(p + ' / CALL / 2', 'cat', True)
        for p in self.STOP:
            self.check(p + ' /CALL/', c.CMD_NO_SHELL, False)
            self.check(p + ' / CALL / 2', c.CMD_NO_SHELL, False)
            self.check('CALL; ' + p + ' /x/', 'cat', False)

    def test_division_positives(self):
        for p in ('a', '1.', 'a[0]', 'f()', 'a++', '"s".length'):
            self.check(p + ' / CALL / 2', 'cat', True)
            self.check('x = ' + p + ' / b; CALL', 'cat', True)



class ReviewR16(unittest.TestCase):
    """r15 F1: a newline before a '/' that would divide. JavaScript may insert a semicolon there, and the
    '/' then opens a regex."""
    cwd = '/nonexistent-home/repo'
    call = 'tools.exec_command({cmd:"cat AGENTS.md"})'
    hit = (('cat', c.EXPLICIT_INSTR, '…/repo/AGENTS.md'),)
    check = ReviewR14.check

    BEFORE_NEWLINE = ('var a', 'let a', 'const a = b', 'var a, b', 'import "m"', "import x from 'm'",
                      'import {x} from "m"', 'export * from "m"', 'export {x} from \'m\'', 'function f',
                      'async function f', 'function* g', 'class A', 'x = class A', '1', '1.5', '0x1F', '10n',
                      '`t`', 'a[0]', '[a]', 'f()', 'f(x)', 'a++', 'a--', 'this', 'null', 'x.y', '"s"', '/re/')

    def test_repro(self):
        self.check('var a\n/x; CALL/.source;', c.CMD_NO_SHELL, False)

    def test_newline_before_division_stops(self):
        for p in self.BEFORE_NEWLINE:
            for gap in ('\n', ' \n ', '\r', '\u2028', '\u2029', ' // c\n', ' /* a\nb */ ', '/*\n*/'):
                self.check(p + gap + '/x; CALL/.source;', c.CMD_NO_SHELL, False)
                self.check(p + gap + '/ CALL / 2', c.CMD_NO_SHELL, False)
                self.check('CALL; ' + p + gap + '/x/', 'cat', False)

    def test_newline_in_interpolation_stops(self):
        self.check('x = `${a\n/ CALL / 2}`', c.CMD_NO_SHELL, False)
        self.check('x = `${a / b}`; CALL', 'cat', True)

    def test_same_line_division_still_labels(self):
        for p in self.BEFORE_NEWLINE:
            if p.split()[0] in ('var', 'let', 'const', 'import', 'export', 'function', 'async') or 'class' in p:
                continue
            for gap in (' ', '', ' /* a b */ '):
                self.check(p + gap + '/ CALL / 2', 'cat', True)
        self.check('x = a\n+ b / c; CALL', 'cat', True)
        self.check('x = a /* one line */ / b; CALL', 'cat', True)

    def test_regex_after_newline_unchanged(self):
        for p in ('x =', 'return', 'f(', 'a,', ';', 'a +', 'typeof'):
            self.check(p + '\n/x; tools/.source; CALL', 'cat', True)
            self.check(p + '\n/CALL/', c.CMD_NO_SHELL, False)


class ReviewR17(unittest.TestCase):
    """r16 F1: a call read before a stop whose label the unread rest can change or overtake; r16 F2: a
    hashbang line."""
    cwd = '/nonexistent-home/repo'
    call = 'tools.exec_command({cmd:"cat AGENTS.md"})'
    hit = (('cat', c.EXPLICIT_INSTR, '…/repo/AGENTS.md'),)
    check = ReviewR14.check
    STOP = 'let x = 1, y = 1, z = 1;\nx\n/ y / z;\n'

    def test_repro(self):
        self.check('const o={cmd:"ls"};\nconst f=()=>tools.exec_command(o);\n' + self.STOP
                   + 'o.cmd="cat AGENTS.md";\nf();', c.CMD_UNPARSED, False)

    def test_label_the_rest_can_change_is_unparsed(self):
        for js in ('let s = "ls";\nconst f = () => tools.exec_command({cmd: s});\nSTOP s = "cat AGENTS.md"; f();',
                   'const o = {cmd: "ls"};\nconst f = () => tools.exec_command(o);\nSTOP o.cmd = "cat"; f();',
                   'const f = () => tools.exec_command({cmd: "ls"});\nSTOP CALL; f();',
                   'function f() { tools.exec_command({cmd: "ls"}); }\nSTOP CALL; f();',
                   'const o = {run() { tools.exec_command({cmd: "ls"}); }};\nSTOP CALL; o.run();',
                   'class A { static run() { tools.exec_command({cmd: "ls"}); } }\nSTOP CALL; A.run();',
                   'g();\ntools.exec_command({cmd: "ls"});\nSTOP function g() { CALL; }',
                   'tools.exec_command({cmd: "ls", x: g()});\nSTOP function g() { CALL; }',
                   'tools.exec_command({cmd: "ls " + g()});\nSTOP function g() { CALL; }',
                   'tools.exec_command({cmd: `ls ${g()}`});\nSTOP function g() { CALL; }',
                   'tools.exec_command({cmd: "ls", x: new G});\nSTOP function G() { CALL; }',
                   'exec_command({cmd: "ls"});\nSTOP function exec_command() { CALL; }',
                   'if (a) { tools.exec_command({cmd: "ls"}); }\nSTOP',
                   'await tools.mcp__t3_code__x({});\nSTOP CALL',
                   'tools.exec_command(o);\nSTOP CALL',
                   # a name read before a stop: the rest can hoist it, put it in its dead zone or rebind it
                   'tools.exec_command({cmd: s});\nSTOP let s = "cat AGENTS.md";',
                   'tools.exec_command({cmd});\nSTOP',
                   'tools.exec_command({[k]: "ls"});\nSTOP',
                   'tools.exec_command({cmd: "ls", w: a ? 1 : 2});\nSTOP',
                   'tools.exec_command({cmd: "ls", get w() { return CALL; }});\nSTOP'):
            self.check(js.replace('STOP', self.STOP), c.CMD_UNPARSED, False)
        self.check('ALL_TOOLS;\n' + self.STOP + 'CALL', c.CMD_NO_SHELL, False)
        # r18: a rest that never names the bindings the call reads cannot change them
        for js, label in (('let s = "ls";\ntools.exec_command({cmd: s});\nSTOP', 'ls'),
                          ('const o = {cmd: "ls"};\ntools.exec_command(o);\nSTOP', 'ls'),
                          ('tools.exec_command({cmd: "git status " + value});\nSTOP', 'git status')):
            self.assertEqual(c.tool_command('exec', js.replace('STOP', self.STOP)), label, js)

    def test_first_statement_with_inert_argument_still_labels(self):
        for js in ('CALL;\nSTOP', 'const r = await CALL;\nSTOP', '\n let r = CALL\nSTOP',
                   'await tools.exec_command({cmd: "cat AGENTS.md", workdir: "/x", yield_time_ms: 1000});\nSTOP',
                   'tools.exec_command({cmd: ["cat", "AGENTS.md"], "workdir": `/x`, a: [1, -2, .5, true, null]});\nSTOP'):
            self.check(js.replace('STOP', self.STOP), 'cat', False)
        # the same name reads label once the whole script is read
        self.check('tools.exec_command({cmd: "git status " + value});', 'git status', False)
        self.check('tools.exec_command({cmd: "cat AGENTS.md", workdir: dir});', 'cat', False)  # no static workdir

    def test_mutation_in_read_text_without_a_stop(self):
        for js in ('const o = {cmd: "ls"};\nconst f = () => tools.exec_command(o);\no.cmd = "cat AGENTS.md";\nf();',
                   'let s = "ls";\nconst f = () => tools.exec_command({cmd: s});\ns = "cat AGENTS.md";\nf();',
                   'const o = {run() { return tools.exec_command(p); }};\nconst p = {cmd: "ls"};\np.cmd = "cat";',
                   # a rest element is a write
                   'let s = "ls";\n[...s] = "cat";\ntools.exec_command({cmd: s});',
                   'let s = "ls";\n[a, ...s] = "cat";\ntools.exec_command({cmd: s});',
                   'let s = "ls";\n({...s} = {a: 1});\ntools.exec_command({cmd: s});',
                   # eval reaches any binding
                   'let s = "ls";\nconst f = () => tools.exec_command({cmd: s});\neval(\'s = "cat"\');\nf();',
                   'const o = {cmd: "ls"};\neval("o.cmd = 1");\ntools.exec_command(o);',
                   # a top-level var is a property of the global object in a sloppy script
                   'var s = "ls";\nglobalThis.s = "cat";\ntools.exec_command({cmd: s});',
                   'var s = "ls";\nthis.s = "cat";\ntools.exec_command({cmd: s});',
                   'var s = "ls";\nFunction("s = 1")();\ntools.exec_command({cmd: s});'):
            self.check(js, c.CMD_UNPARSED, False)
        for js in ('const s = "cat AGENTS.md";\nconst t = [...s];\ntools.exec_command({cmd: s});',
                   'const s = "cat AGENTS.md";\nglobalThis.s = 1;\ntools.exec_command({cmd: s});',
                   'CALL;\neval("x");'):
            self.check(js, 'cat', js.startswith('CALL'))  # only a literal cmd is attributed

    def test_script_that_rebinds_tools_or_exec_command(self):
        # a hoisted declaration, a parameter, a write, an identifier escape or a with makes the call reach no tool
        for js in ('CALL;\nfunction tools() {}', 'CALL;\nvar tools;', 'CALL;\n{ function tools() {} }',
                   'const f = (tools) => CALL;\nf(x);', 'tools.exec_command = g;\nCALL;', 'CALL;\nclass tools {}',
                   'exec_command({cmd: "cat AGENTS.md"});\nfunction exec_command() {}',
                   'CALL;\nfunction \\u0074ools() {}', 'with (o) { CALL; }',
                   'await tools.mcp__t3_code__x({});\nfunction tools() {}'):
            self.check(js, c.CMD_UNPARSED, False)
        # and so does any such mention in the unread rest of a stopped scan
        for rest in ('function tools() {}', 'var tools;', 'let tools = 1;', '{ function tools() {} }',
                     'function \\u0074ools() {}', 'f(tools);'):
            self.check('CALL;\n' + self.STOP + rest, c.CMD_UNPARSED, False)
        for js in ('CALL;\nObject.keys(tools);', 'const exec_command = 1;\nCALL;', 'CALL;\nwith (o) {}'):
            self.check(js, 'cat', True)
        self.check('CALL;\n' + self.STOP + 'tools.x();', 'cat', False)

    def test_hashbang_is_a_comment(self):
        for head in ('#! CALL', '#!/usr/bin/env node CALL', '#!/usr/bin/node'):
            for end in ('\n', '\r', '\r\n', ' ', ' '):
                js = head.replace('CALL', 'tools.exec_command({cmd:"ls"})') + end + 'CALL;'
                self.check(js, 'cat', True)
        self.check('#! CALL', c.CMD_NO_SHELL, False)
        self.check('#!/usr/bin/env node', c.CMD_NO_SHELL, False)
        # only at offset 0. Node's .mjs loader accepts '#!' after a BOM, but a VM script or module and the .cjs
        # loader reject one, so no label
        self.check('﻿#! x\nCALL;', c.CMD_NO_SHELL, False)
        self.check(' #! x\nCALL;', c.CMD_NO_SHELL, False)
        # a '#!' past offset 0 is a syntax error, read or unread
        self.check('CALL;\n#! x', c.CMD_UNPARSED, False)
        self.check('CALL;\n' + self.STOP + '#! x', c.CMD_UNPARSED, False)

    def test_call_in_a_template_interpolation(self):
        # the interpolation runs before the code after the template, but the scan masks it
        for js in ('const t = `${tools.exec_command({cmd: "ls"})}`;\nCALL;',
                   'const t = `a ${exec_command({cmd: "ls"})}`;\nCALL;',
                   'const t = `${await tools.view_image({path: "x"})}`;\nCALL;'):
            self.check(js, c.CMD_UNPARSED, False)
        # and so does a write, call or tool reference there, which the scan cannot see
        for js in ('let s = "ls";\n`${s = "cat AGENTS.md"}`;\ntools.exec_command({cmd: s});',
                   'const saved = tools.exec_command;\n`${tools.exec_command = a => saved({cmd: "ls"})}`;\nCALL;',
                   'let n = 1;\n`${n++}`;\nCALL;',
                   'const o = {cmd: "ls"};\n`${"}" + (delete o.cmd)}`;\ntools.exec_command(o);'):
            self.check(js, c.CMD_UNPARSED, False)
        self.check('const t = `tools.exec_command({cmd: "ls"})`;\nCALL;', 'cat', True)  # no interpolation
        self.check('const t = `(a) ${a.b} = ${c}`;\nCALL;', 'cat', True)
        self.check('tools.exec_command({cmd: `cat (x) ${dir}`});', 'cat', False)
        # a call there is fine: the body it runs is read where it is declared
        self.check('let s = "ls";\nfunction f() { s = "cat AGENTS.md"; }\n`${f()}`;\ntools.exec_command({cmd: s});',
                   c.CMD_UNPARSED, False)
        self.check('let s = "ls";\n`${f()}`;\ntools.exec_command({cmd: s});', 'ls', False)
        # a '}' in a comment or regex does not end the interpolation
        for js in ('let s = "ls";\n`${ /* } */ s = "cat AGENTS.md" }`;\ntools.exec_command({cmd: s});',
                   'let s = "ls";\n`${ /}/.test("x") ? (s = "cat AGENTS.md") : 0 }`;\ntools.exec_command({cmd: s});',
                   'const saved = tools.exec_command;\n`${ /* } */ tools.exec_command = x => saved({cmd: "ls"}) }`;\n'
                   'tools.exec_command({cmd: "cat AGENTS.md"});'):
            self.check(js, c.CMD_UNPARSED, False)
        # a comparison or an arrow writes nothing; any assignment operator does
        self.check('const t = `${a === b ? 1 : 2} ${a <= b} ${xs.map(x => x.k).join()}`;\nCALL;', 'cat', True)
        for op in ('+=', '??=', '>>>='):
            self.check('let s = "ls";\n`${s ' + op + ' "x"}`;\ntools.exec_command({cmd: s});', c.CMD_UNPARSED, False)

    def test_proto_initializer_is_not_a_property(self):
        self.check('const o = {__proto__: "cat AGENTS.md"};\ntools.exec_command({cmd: o.__proto__});',
                   c.CMD_UNPARSED, False)
        self.check('const o = {__proto__: {cmd: "cat AGENTS.md"}};\ntools.exec_command(o);', c.CMD_UNPARSED, False)
        # the other keys stay own properties
        self.check('const o = {"__proto__": "x", cmd: "cat AGENTS.md"};\ntools.exec_command(o);', 'cat', False)
        self.check('tools.exec_command({__proto__: null, cmd: "cat AGENTS.md"});', 'cat', False)

    def test_stray_hash_in_a_fully_read_script(self):
        # Node rejects a '#' outside a class body, so nothing runs
        for js in ('CALL; #notPrivate;', 'const x = 1; #y; CALL;', 'CALL; # comment'):
            self.check(js, c.CMD_UNPARSED, False)
        for js in ('class A { #x = 1; m() { return #x in this; } }\nCALL;', 'const s = "#x";\nCALL;',
                   'CALL; // #x'):
            self.check(js, 'cat', True)


class ReviewR18(unittest.TestCase):
    """r17 F1: a template tag in the argument runs code first; r17 F2: tools replaced through eval, an alias or
    globalThis. r18 labels an exec script only through an allowlist grammar (README, "Exec scripts")."""
    cwd = '/nonexistent-home/repo'
    hit = (('cat', c.EXPLICIT_INSTR, '…/repo/AGENTS.md'),)
    CALL = 'tools.exec_command({cmd: "cat AGENTS.md"})'

    def label(self, js):
        return c.tool_command('exec', js.replace('CALL', self.CALL))

    def res(self, js):
        return c.result_label_for(None, 'exec', js.replace('CALL', self.CALL), self.cwd)

    def unparsed(self, js):
        self.assertEqual(self.label(js), c.CMD_UNPARSED, js)
        self.assertEqual(self.res(js), (c.CMD_UNPARSED,), js)

    def test_template_tag_in_the_argument(self):
        # a tag is a call: it runs before exec_command, and its body may run another command
        for js in ('tools.exec_command({cmd: "cat AGENTS.md", x: 1..constructor.constructor`return '
                   'tools.exec_command({cmd: "rm x"})`()});',
                   'tools.exec_command({cmd: "cat AGENTS.md", x: f`a`});',
                   'tools.exec_command({cmd: f`cat AGENTS.md`});',
                   'tools.exec_command({cmd: "cat AGENTS.md", x: [f`a`]});',
                   'tools.exec_command({cmd: "cat AGENTS.md", x: (f)`a`});',
                   'const c = f`cat AGENTS.md`;\nCALL;',
                   'f`a`;\nCALL;'):
            self.unparsed(js)

    def test_tools_replaced_before_the_call(self):
        for js in ('eval("tools = {exec_command: x => x}");\nCALL;',
                   'const t = tools;\nt.exec_command = () => 0;\nCALL;',
                   'const t=tools;\nt.exec_command({cmd: "ls"});\nCALL;',
                   'globalThis.tools = {exec_command: x => x};\nCALL;',
                   'globalThis["to" + "ols"].exec_command = () => 0;\nCALL;',
                   'const g = 1..constructor.constructor("return this")();\ng.tools = 0;\nCALL;',
                   'tools.exec_command = () => 0;\nCALL;',
                   'x = tools;\nx.exec_command = 0;\nCALL;', 'let t = 1;\nt = tools;\nCALL;',
                   'tools["exec_command"] = () => 0;\nCALL;'):
            self.unparsed(js)

    def test_tools_named_or_reached_after_the_call(self):
        # hoisting (var, function) or a binding read can reach tools from code after the call
        for js in ('CALL;\nvar tools = 1;', 'CALL;\nfunction tools() {}', 'CALL;\nconst t = tools;',
                   'CALL;\nglobalThis.x = 1;', 'CALL;\neval("1");', 'CALL;\nconst s = "\\u0074ools";',
                   'CALL;\n#x;', 'CALL;\nFunction("return 1")();'):
            self.unparsed(js)

    def test_bom_then_hashbang(self):
        # A BOM is JS whitespace. Node's .mjs loader accepts a '#!' after it, but a VM script or module and the
        # .cjs loader reject one, so the collector does not treat it as a hashbang: no label either way.
        self.unparsed('﻿#! x\nCALL;')
        self.unparsed('﻿#!/usr/bin/env node\nCALL;')
        self.assertEqual(self.res('﻿CALL;'), self.hit)
        self.assertEqual(self.res('#!/usr/bin/env node\nCALL;'), self.hit)

    def test_unsafe_integer_is_outside_the_grammar(self):
        # a JS number rounds an integer above 2**53 - 1, so the value read would differ from the one run
        for n in ('9007199254740992', '9007199254740993', '0x20000000000000', '1e400', '-9007199254740993'):
            self.unparsed('tools.exec_command({cmd: "cat AGENTS.md", n: %s});' % n)
        for n in ('9007199254740991', '0x1FFFFFFFFFFFFF', '-9007199254740991', '1e3', '0.5'):
            self.assertEqual(self.res('tools.exec_command({cmd: "cat AGENTS.md", n: %s});' % n), self.hit, n)

    def test_prefix_declarations(self):
        for js in ('const c = "cat AGENTS.md";\ntools.exec_command({cmd: c});',
                   'let c = "cat AGENTS.md"\ntools.exec_command({cmd: c})',
                   'const d = "AGENTS.md", c = `cat ${d}`;\ntools.exec_command({cmd: c});',
                   'const b = "cat";\nconst c = b + " AGENTS.md";\ntools.exec_command({cmd: c});',
                   'const o = {cmd: "cat AGENTS.md", yield_time_ms: 1000};\ntools.exec_command(o);',
                   'const cmd = "cat AGENTS.md";\ntools.exec_command({cmd});',
                   ';;\nconst n = -1, ok = true, z = null, xs = ["a", 2];\nCALL;'):
            self.assertEqual(self.res(js), self.hit, js)
        # var, an unbound or reserved initializer, a later write and a '(' on the next line stay outside
        for js in ('var c = "cat AGENTS.md";\ntools.exec_command({cmd: c});',
                   'const c = path.join("a", "b");\nCALL;', 'const c = x;\nCALL;', 'const c = String.raw`x`;\nCALL;',
                   'const text = "x";\nCALL;', 'const Promise = 1;\nCALL;', 'const c = "a", c = "b";\nCALL;',
                   'const o = {cmd: "ls"};\ntools.exec_command(o);\no.cmd = "cat AGENTS.md";',
                   'const c = "ls"\n(tools.exec_command({cmd: "cat AGENTS.md"}))',
                   'const c = "ls"\n[tools.exec_command({cmd: "cat AGENTS.md"})]',
                   'const c = "ls"\nf();\nCALL;', 'const c = {get cmd() { return "ls"; }};\nCALL;',
                   'const c = {["cmd"]: "ls"};\nCALL;', 'const c = {...o};\nCALL;'):
            self.unparsed(js)

    def test_call_wrappers_and_result_bindings(self):
        for js in ('await CALL;', '(CALL);', 'text(await CALL);', 'const r = await CALL;\nconsole.log(r.output);',
                   'let r = CALL;', 'var r = await CALL;', 'const {output} = await CALL;',
                   'const {output: out, exit_code} = await CALL;', 'const [r] = await Promise.all([CALL]);',
                   'const [a, , b] = await Promise.allSettled([CALL, tools.exec_command({cmd: "ls"})]);',
                   'await Promise.race([CALL]);', 'text((await CALL).output);'):
            self.assertEqual(self.label(js), 'cat', js)
        self.assertEqual(self.res('const r = await CALL;\nconsole.log(r.output);'), self.hit)
        # a call before it, or a wrapper that is not on the list, is not the first call
        for js in ('console.log(await CALL);', 'f(CALL);', 'Promise.resolve(CALL);', 'Promise.all(xs.map(CALL));',
                   'const [a = f()] = await CALL;', 'const {a: {b}} = await CALL;', 'const [...r] = await CALL;',
                   'const {[k]: v} = await CALL;', 'await Promise.all([f(), CALL]);', 'x.text(CALL);'):
            self.assertIn(self.label(js), (c.CMD_UNPARSED, c.CMD_NO_SHELL), js)
            self.assertNotEqual(self.res(js), self.hit, js)

    def test_runtime_values_label_without_attribution(self):
        for js in ('tools.exec_command({cmd: "git status " + value});', 'tools.exec_command({cmd: `git status ${value}`});'):
            self.assertEqual(self.label(js), 'git status', js)
            self.assertEqual(self.res(js), ('git status',), js)
        for js in ('tools.exec_command({cmd: "cat -n " + file});', 'tools.exec_command({cmd: `cat -n ${file}`});',
                   'tools.exec_command({cmd: `LC_ALL=C cat -n ${f}`});'):
            self.unparsed(js)
        # a runtime value where the program name is, or first in a '+' join, may be anything
        for js in ('tools.exec_command({cmd: `${bin} AGENTS.md`});', 'tools.exec_command({cmd: bin + " AGENTS.md"});',
                   'tools.exec_command({cmd: "ca" + rest});', 'tools.exec_command({cmd: `ca${t} AGENTS.md`});',
                   'tools.exec_command({cmd: c});', 'tools.exec_command(args);'):
            self.unparsed(js)

    def test_later_declaration_of_a_name_the_call_reads(self):
        # a duplicate declaration is an early error (nothing runs); a later let/const/class of a name read is a
        # TDZ throw before the call; a var or function of Promise hoists to undefined
        for js in ('tools.exec_command({cmd: "cat -n " + file});\nconst file = "x";',
                   'tools.exec_command({cmd: "cat -n " + file});\nlet [file] = [1];',
                   'tools.exec_command({cmd: "cat -n " + file});\nlet {a: file} = o;',
                   'tools.exec_command({cmd: "cat -n " + file});\nclass file {}',
                   'tools.exec_command({cmd: "cat -n " + file});\nlet a = 1, file = 2;',
                   'text(await CALL);\nconst text = 1;',
                   'await Promise.all([CALL]);\nvar Promise = 1;',
                   'await Promise.all([CALL]);\nfunction Promise() {}',
                   'const c = "cat AGENTS.md";\ntools.exec_command({cmd: c});\nlet c = 2;',
                   'const r = await CALL;\nfunction r() {}',
                   'const r = await CALL;\nvar x = [r];',
                   'tools.exec_command({cmd: "cat -n " + file});\nx = /a/;\nfile;'):  # a slash the scan cannot follow
            self.unparsed(js)
        for js in ('tools.exec_command({cmd: "git status " + file});\nconsole.log(file);',
                   'const r = await CALL;\nconst x = r.output, y = {r};', 'text(await CALL);\ntext("done");'):
            self.assertIn(self.label(js), ('cat', 'git status'), js)

    def test_literal_entries_before_the_call(self):
        for js in ('const n = 1;\nawait Promise.all(["a", n, 2, CALL]);', 'text({a: "x", b: await CALL});',
                   'const n = 1;\ntext({n, b: await CALL});'):
            self.assertEqual(self.res(js), self.hit, js)
        for js in ('await Promise.all([f(), CALL]);', 'await Promise.all([n, CALL]);', 'text({[k]: 1, b: await CALL});',
                   'text({n, b: await CALL});', 'text({__proto__: "x", b: await CALL});',
                   'await Promise.all(["a" + f(), CALL]);', 'text({tools: 1, b: await CALL});'):
            self.unparsed(js)

    def test_map_callback_runs_the_call_for_the_first_element(self):
        X = 'tools.exec_command({cmd: c})'
        P = 'const cs = ["cat AGENTS.md"];\n'
        for js in (P + 'await Promise.all(cs.map(async c => %s));' % X,
                   P + 'await Promise.all(cs.map(c => %s));' % X,
                   P + 'await Promise.all(cs.map(async (c, i) => {\n  const r = await %s;\n  return r;\n}));' % X,
                   P + 'await Promise.all(cs.map(async c => {\n  return await %s;\n}));' % X,
                   'const cs = [["cat AGENTS.md", 1]];\nawait Promise.all(cs.map(async ([c, n]) => %s));' % X,
                   'const cs = [{c: "cat AGENTS.md"}];\nawait Promise.all(cs.map(async ({c}) => %s));' % X,
                   P + 'await Promise.all(cs.map(async c => %s));\ncs.push("x");\nconst c = 2;' % X):
            self.assertEqual(self.res(js), self.hit, js)
        # more elements run the call once each: the first label stands, the attribution does not
        js = 'const cs = ["cat AGENTS.md", "ls"];\nawait Promise.all(cs.map(async c => %s));' % X
        self.assertEqual(self.label(js), 'cat')
        self.assertEqual(self.res(js), ('cat',))
        self.assertIsNone(c._raw_command('exec', js))
        for js in ('const cs = [];\nawait Promise.all(cs.map(async c => %s));' % X,
                   'const cs = [["cat AGENTS.md"]];\nawait Promise.all(cs.map(async ([a, c]) => %s));' % X,
                   'const cs = [{a: 1}];\nawait Promise.all(cs.map(async ({c}) => %s));' % X,
                   P + 'await Promise.all(cs.map(async (c, c) => %s));' % X,
                   'const cs = [["ls", "cat AGENTS.md"]];\nawait Promise.all(cs.map(async ([c, c]) => %s));' % X,
                   P + 'await Promise.all(cs.map(c => await %s));' % X,
                   P + 'await Promise.all(cs.map(async\nc => %s));' % X,
                   P + 'await Promise.all(cs.map(async c => {\n  f();\n  await %s;\n}));' % X,
                   P + 'await Promise.all(cs.map(async c => cs.map(async d => tools.exec_command({cmd: d}))));',
                   P + 'await Promise.all(cs.map(async (c = f()) => %s));' % X,
                   'const cs = "ab";\nawait Promise.all(cs.map(async c => %s));' % X):
            self.unparsed(js)

    def test_map_callback_names_and_later_writes(self):
        X = 'tools.exec_command({cmd: c})'
        P = 'const cs = ["cat AGENTS.md"];\n'
        # a declaration in the callback body of a parameter or result name: an early error or a TDZ throw
        for js in (P + 'await Promise.all(cs.map(async c => {\n  await %s;\n  function c() {}\n}));' % X,
                   P + 'await Promise.all(cs.map(async c => {\n  await %s;\n  let c = 1;\n}));' % X,
                   P + 'await Promise.all(cs.map(async c => {\n  const r = await %s;\n  { var r = 1; }\n}));' % X,
                   P + 'await Promise.all(cs.map(async c => {\n  const r = await %s;\n}));\nlet cs = 1;' % X,
                   'const r = 1;\n' + P + 'await Promise.all(cs.map(async c => {\n  const r = await %s;\n}));'
                   '\nlet r = 2;' % X,
                   P + 'await Promise.all(cs.map(async c => {\n  const r = await tools.exec_command({cmd: "cat -n " + r});\n}));',
                   # the approved runtime form, so only the TDZ of the result name makes it Unparsed
                   P + 'await Promise.all(cs.map(async c => {\n  const r = await tools.exec_command({cmd: "git status " + r});\n}));'):
            self.unparsed(js)
        # a later write through an object the argument shares
        for js in ('const cs = [{cmd: "cat AGENTS.md"}];\nawait Promise.all(cs.map(async c => tools.exec_command(c)));'
                   '\ncs[0].cmd = "rm x";',
                   'const o = {cmd: "cat AGENTS.md"};\nconst cs = [o];\n'
                   'await Promise.all(cs.map(async c => tools.exec_command(c)));\no.cmd = "rm x";'):
            self.unparsed(js)


def replay_h(events, windows=None, role='top', src='claude'):
    acc, recon, counters, basis, hacc = {}, {}, defaultdict(int), defaultdict(set), {}
    windows = windows or [(0, 1e12)]
    c.attribute({'src': src, 'role': role, 'events': events}, ratios(), windows, acc, recon, counters, basis,
                hacc=hacc)
    return acc, hacc, counters, basis


def hist_leaves(nodes):
    for n in nodes:
        if 'children' in n:
            yield from hist_leaves(n['children'])
        else:
            yield n


class ReviewR18c(unittest.TestCase):
    """r18 F1: a name the argument reads outside cmd throws a ReferenceError before the call runs, and a repeated
    key still evaluates the value it drops; r18 F2: the two guards its mutants removed are not equivalent."""
    cwd, hit, CALL = ReviewR18.cwd, ReviewR18.hit, ReviewR18.CALL
    label, res, unparsed = ReviewR18.label, ReviewR18.res, ReviewR18.unparsed

    def test_unbound_name_outside_cmd_is_unparsed(self):
        for js in ('tools.exec_command({cmd: "cat AGENTS.md", yield_time_ms: missing});',
                   'tools.exec_command({cmd: "cat AGENTS.md", meta: {x: missing}});',
                   'tools.exec_command({cmd: "cat AGENTS.md", meta: [missing]});',
                   'tools.exec_command({cmd: "cat AGENTS.md", meta: missing + " x"});',
                   'tools.exec_command({cmd: "cat AGENTS.md", meta: `${missing}`});',
                   'tools.exec_command({cmd: "cat AGENTS.md", yield_time_ms});',
                   'tools.exec_command({cmd: "cat AGENTS.md", workdir: dir});',
                   'tools.exec_command({cmd: "cat AGENTS.md", workdir: "/repo/" + n});',
                   'tools.exec_command({cmd: "git status " + tag`x`});',
                   'tools.exec_command({cmd: ["cat", missing]});',
                   'tools.exec_command({cmd: {x: missing}});',
                   'const jobs = [{cmd: "cat AGENTS.md", yield_time_ms: missing}];\n'
                   'await Promise.all(jobs.map(async job => tools.exec_command(job)));',
                   'tools.exec_command({cmd: "cat AGENTS.md", meta: {cmd: missing}});',
                   'tools.web__search({cmd: missing});',
                   'tools.web__search({q: missing});', 'tools.update_plan({plan: missing});',
                   'tools.apply_patch(missing);'):
            self.unparsed(js)
            self.assertIsNone(c._raw_command('exec', js), js)
        # the control: literals only still label and attribute
        self.assertEqual(self.label('tools.exec_command({cmd: "echo synthetic-marker"});'), 'echo')
        self.assertEqual(self.res('CALL;'), self.hit)
        self.assertEqual(self.label('tools.web__search({q: "x"});'), 'web')

    def test_repeated_key_is_unparsed(self):
        for js in ('tools.exec_command({cmd: missing, cmd: "cat AGENTS.md"});',
                   'tools.exec_command({cmd: "ls", cmd: "cat AGENTS.md"});',
                   'tools.exec_command({cmd: "cat AGENTS.md", "cmd": "cat AGENTS.md"});',
                   'tools.exec_command({cmd: "cat AGENTS.md", workdir: "/a", workdir: "/b"});',
                   'tools.exec_command({cmd: "cat AGENTS.md", meta: {a: 1, a: 2}});',
                   'const o = {cmd: "ls", cmd: "cat AGENTS.md"};\ntools.exec_command(o);'):
            self.unparsed(js)
            self.assertIsNone(c._raw_command('exec', js), js)

    def test_runtime_cmd_is_unparsed_outside_the_approved_form(self):
        # a runtime value can hold any shell text, even an unmatched quote ("cat -n " + '"' runs no cat), and an
        # unbound one throws before the call, so only the approved "git status " + value form keeps its label
        for js in ('tools.exec_command({cmd: "cat -n " + missing});', 'tools.exec_command({cmd: `cat -n ${x}`});',
                   'tools.exec_command({cmd: "cat AGENTS.md" + s});', 'tools.exec_command({cmd: `cat -n ${d}/AGENTS.md`});',
                   'tools.exec_command({cmd: `LC_ALL=C /bin/cat -n ${f}`});', 'tools.exec_command({cmd: `npm run ${t}`});',
                   'tools.exec_command({cmd: `foo -x ${f}`});', 'tools.exec_command({cmd: `git ${value} status`});',
                   'tools.exec_command({cmd: "git " + value});', 'tools.exec_command({cmd: `cd ${dir} && git show`});',
                   'tools.exec_command({cmd: `${x} status`});', 'tools.exec_command({cmd: `git "-p ${v} "`});',
                   'tools.exec_command({cmd: "git status" + x});', 'tools.exec_command({cmd: `git status -- ${f}`});',
                   'tools.exec_command({cmd: `git status ${a} ${b}`});', 'tools.exec_command({cmd: " git status " + x});',
                   'tools.exec_command({cmd: "git  status " + x});'):
            self.assertEqual(c._exec_script_verdict(js), (c.CMD_UNPARSED, 'cmd holds a runtime value'), js)
            self.unparsed(js)
            self.assertIsNone(c._raw_command('exec', js), js)
        for js in ('tools.exec_command({cmd: "git status " + value});', 'tools.exec_command({cmd: `git status ${value}`});',
                   'tools.exec_command({cmd: "git status " + value + ""});'):  # the same string
            self.assertEqual(c._exec_script_verdict(js), ('git status', None), js)
            self.assertEqual(self.res(js), ('git status',), js)
            self.assertIsNone(c._raw_command('exec', js), js)

    def test_proto_key_in_a_wrapper_object(self):
        # kills the entries-proto-key mutant: a "__proto__" key is prototype syntax, never a plain property
        self.unparsed('text({"__proto__": "x", b: await CALL});')
        self.unparsed('text({__proto__: "x", b: await CALL});')

    def test_runtime_value_first_in_a_join(self):
        # kills the runtime-first-join mutant: the join itself is outside the grammar, a number may come first
        self.assertEqual(c._exec_script_verdict('tools.exec_command({cmd: x + " status"});'),
                         (c.CMD_UNPARSED, 'concatenation outside the grammar'))
        self.unparsed('tools.exec_command({cmd: "cat AGENTS.md", meta: missing + " x"});')


class Histogram(unittest.TestCase):
    def test_daily_zero_metric_cells_preserve_occurrences_in_selected_dates(self):
        first = datetime.date(2026, 10, 20)
        with LocalTZ('UTC'):
            start = datetime.datetime.combine(first, datetime.time()).timestamp()
            days = c.Days(first, first + datetime.timedelta(days=1), start, start + 2 * 86400)
            events = []
            for day in range(2):
                for i, size in enumerate((11, 13, 17, 19)):
                    events.append(('B', (c.INSTR,), size, 0, c.B_EST, 'file-%d.md' % i))
                for i, size in enumerate((23, 29, 31, 37)):
                    events.append(('B', BASH_RES, size, 0, c.B_EST, 'command-%d' % i))
                for i, size in enumerate((7, 9)):
                    events.append(('B', (c.TOOLCTX, 'Tool schemas', 'schema-%d' % i), size, 0, c.B_EST))
                for i, size in enumerate((6, 8)):
                    events.append(('B', TYPED + ('prompt-%d' % i,), size, 0, c.B_EST))
                events.append(call(1, t=start + day * 86400 + 12 * 3600))
                events.append(('R',))
            acc, hacc, _, basis = replay_h(events, windows=days)
        self.assertEqual(sum(b[0] for bins in hacc.values() for b in bins.values()), 24)
        tree = c.build_tree(acc, ['opus-5.5'], ['top'], basis, hacc=hacc)
        self.check_tree(tree)
        rows = [(row, bins) for leaf in hist_leaves(tree) for row, bins in zip(leaf['v'], leaf['h'])]
        for lo, hi, expected in ((0, 0, 12), (1, 1, 12), (0, 1, 24)):
            selected = [(row, bins) for row, bins in rows if lo <= row[0] <= hi]
            self.assertEqual(sum(b[1] for _, bins in selected for b in bins), expected)
            self.assertEqual(sum(row[3] for row, _ in selected), hi - lo + 1)
        self.assertTrue(any(not any(row[3:]) and any(b[1] for b in bins) for row, bins in rows))

    def test_size_bin_edges(self):
        self.assertEqual([c.size_bin(t) for t in (0, 99.9, 100, 199, 200, 999, 1000, 999999, 1000000, 5e6)],
                         [0, 0, 1, 1, 2, 3, 4, 12, 13, 13])
        self.assertEqual(c.size_bin(c.HIST_EDGES[-1]), len(c.HIST_EDGES))

    def check_sums(self, acc, hacc):
        self.assertEqual(set(acc), set(hacc))
        for k, cell in acc.items():
            bins = hacc[k]
            self.assertEqual(sum(x[1] for x in bins.values()), cell[0], k)
            self.assertAlmostEqual(sum(x[2] for x in bins.values()), cell[1], places=6, msg=k)
            self.assertAlmostEqual(sum(x[3] for x in bins.values()), cell[2], places=9, msg=k)

    def test_bins_add_up_to_every_cell(self):
        events = [('P', [((c.INSTR, 'X'), 300), ((c.INSTR, 'Y'), 4000)])] + CommandSplit().events(True)[:-2] + [
            ('T',), call(2600, cr=2200, t=1003.0), ('R',), ('B', TYPED, 40, 0, c.B_EST), call(80, t=1004.0)]
        acc, hacc, counters, _ = replay_h(events)
        self.check_sums(acc, hacc)
        self.assertEqual(counters['hist_fallback'], 0)
        whole, _, _ = replay(events)  # the histogram does not change the cells
        self.assertEqual(acc, whole)

    def test_rereads_add_tokens_not_occurrences(self):
        b = BASH_RES + ('cat',)
        events = [('B', b, 400, 0, c.B_EST), call(400), call(400, cr=400, t=1001.0),
                  ('B', b, 3000, 0, c.B_EST), call(3400, cr=400, t=1002.0)]
        acc, hacc, _, _ = replay_h(events)
        bins = hacc[(b, 0, 'opus-5.5', 'top')]
        self.assertEqual(sorted(bins), [c.size_bin(400), c.size_bin(3000)])
        small, large = bins[c.size_bin(400)], bins[c.size_bin(3000)]
        self.assertEqual((small[0], small[1]), (1, 1200))  # read on three calls, counted once
        self.assertEqual((large[0], large[1]), (1, 3000))
        self.assertAlmostEqual(small[2], 400, places=6)
        self.assertAlmostEqual(large[2], 3000, places=6)
        self.check_sums(acc, hacc)

    def test_output_prefix_and_unattributed_occurrences(self):
        events = [('P', [((c.INSTR, 'X'), 300), ((c.INSTR, 'X'), 5000)]),
                  ('B', TYPED, 100, 0, c.B_EST), call(6000, out=40, outs=[(BASH_CALL, 40, 0, 'ls')]),
                  call(6100, cr=6000, t=1001.0), ('P', [((c.INSTR, 'X'), 300)]), call(7000, t=1002.0)]
        acc, hacc, _, _ = replay_h(events)
        x = hacc[((c.INSTR, 'X'), 0, 'opus-5.5', 'top')]
        # two sections from the first snapshot, one from the second; re-reads in between add none
        self.assertEqual(sum(v[0] for v in x.values()), 3)
        self.assertEqual(x[c.size_bin(300)][0], 2)
        self.assertEqual(x[c.size_bin(5000)][0], 1)
        ls = hacc[(BASH_CALL + ('ls',), 0, 'opus-5.5', 'top')]
        self.assertEqual(ls[c.size_bin(40)][0], 1)
        un = [v for k, v in hacc.items() if k[0] == c.UNATTRIB]
        self.assertTrue(un)
        self.assertEqual(sum(b[0] for v in un for b in v.values()), 3)  # one per call with a residual
        self.check_sums(acc, hacc)

    def test_reset_before_a_call_drops_the_occurrence(self):
        events = [('B', TYPED, 300, 0, c.B_EST), ('R',), ('B', TYPED, 50, 0, c.B_EST), call(50)]
        acc, hacc, _, _ = replay_h(events)
        bins = hacc[(TYPED, 0, 'opus-5.5', 'top')]
        self.assertEqual({b: v[0] for b, v in bins.items()}, {c.size_bin(50): 1})
        self.check_sums(acc, hacc)

    def test_serialized_histograms_match_cells(self):
        acc, hacc, _, basis = replay_h(CommandSplit().events(True))
        tree = c.build_tree(acc, ['opus-5.5', 'gpt-6-luna'], ['top'], basis, top_cmds=1, hacc=hacc)
        self.check_tree(tree)
        other = find(tree, c.RESULTS, 'Built-in tools', 'Bash', 'Other (2 commands)')
        self.assertTrue(other['h'][0])
        plain = c.build_tree(acc, ['opus-5.5', 'gpt-6-luna'], ['top'], basis, top_cmds=1)
        self.assertNotIn('h', next(hist_leaves(plain)))

    def check_tree(self, tree):
        seen = 0
        for leaf in hist_leaves(tree):
            self.assertEqual(len(leaf['v']), len(leaf['h']), leaf['label'])
            self.assertEqual([r[:3] for r in leaf['v']], sorted(r[:3] for r in leaf['v']))
            for row, bins in zip(leaf['v'], leaf['h']):
                cell = row[3:]
                self.assertTrue(any(cell) or any(b[1] for b in bins), leaf['label'])
                seen += 1
                self.assertEqual([b[0] for b in bins], sorted({b[0] for b in bins}))
                self.assertTrue(all(0 <= b[0] <= len(c.HIST_EDGES) for b in bins))
                self.assertEqual(sum(b[2] for b in bins), cell[0], leaf['label'])
                self.assertEqual(sum(b[3] for b in bins), cell[1], leaf['label'])
                self.assertAlmostEqual(sum(b[4] for b in bins), cell[2], places=9, msg=leaf['label'])
                self.assertTrue(all(b[1] >= 0 for b in bins))
        self.assertGreater(seen, 0)

    def test_end_to_end_leaves_carry_consistent_histograms(self):
        e2e = EndToEnd()
        e2e.setUp()
        try:
            self.check_tree(e2e.data['tree'])
            self.assertEqual(e2e.data['meta']['size_bin_edges'], list(c.HIST_EDGES))
            self.assertEqual(e2e.data['meta']['size_bin_fallbacks'], 0)
        finally:
            shutil.rmtree(e2e.tmp, ignore_errors=True)


class AgentMessages(unittest.TestCase):
    BRIEF = '<task-brief>\nGoal:\nfix it\n</task-brief>'

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def claude_blocks(self, lines, is_sub=False):
        p = os.path.join(self.tmp, 'a.jsonl')
        write_jsonl(p, lines)
        sess, _ = c.parse_claude(p, is_sub)
        out = defaultdict(int)
        for e in sess['events']:
            if e[0] == 'B':
                out[e[1]] += e[2]
        return dict(out)

    def notif(self, tid):
        return ('<task-notification><task-id>x</task-id><tool-use-id>%s</tool-use-id><status>completed</status>'
                '<summary>done</summary><result>%s</result></task-notification>' % (tid, 'r' * 30))

    def calls(self, *tools):
        o = claude_asst('m1', 2, 10, 0, 0, 5)
        o['message']['content'] = [{'type': 'tool_use', 'id': cid, 'name': name, 'input': {'prompt': 'p'}}
                                   for cid, name in tools]
        return o

    def test_brief_after_role_profile_is_agent_message(self):
        profile = ROLE_TAG % 'drone'
        for sep in ('', '\n\n'):
            text = profile + sep + self.BRIEF
            self.assertEqual(c.split_text(text, 'Typed prompts'),
                             [((c.INSTR, 'Subagent role profile', 'Drone'), len(profile)),
                              (c.P_BRIEF, len(sep + self.BRIEF))])
        rem = '<system-reminder>' + 'r' * 20 + '</system-reminder>'
        parts = dict(c.split_text(rem + profile + self.BRIEF + '\nlater', 'Typed prompts'))
        self.assertEqual(parts[c.P_BRIEF], len(self.BRIEF))
        self.assertEqual(parts[(c.USER, 'Typed prompts')], len('\nlater'))
        self.assertEqual(parts[(c.INSTR, 'Subagent role profile', 'Drone')], len(profile))

    def test_brief_without_anchor_stays_with_sender_text(self):
        profile = ROLE_TAG % 'envoy'
        for text in (self.BRIEF,
                     'Please write a <task-brief> for the agent, developer; Goal: ship',
                     profile + 'Note first.\n' + self.BRIEF,
                     profile + '<task-brief>\nGoal: never closed',
                     'Tell the subagent to delegate_task the developer message to another agent.'):
            parts = c.split_text(text, 'Typed prompts')
            self.assertNotIn(c.P_BRIEF, dict(parts), text)
            self.assertEqual(sum(n for _, n in parts), len(text))
        # a role profile alone stays an instruction, never a delivered brief
        self.assertEqual(c.split_text(profile, 'Typed prompts'),
                         [((c.INSTR, 'Subagent role profile', 'Envoy'), len(profile))])
        pasted = '<pasted_content>' + profile + self.BRIEF + '</pasted_content>'
        self.assertEqual(c.split_text(pasted, 'Typed prompts'), [((c.USER, 'Pasted content'), len(pasted))])

    def test_codex_brief_and_role(self):
        p = os.path.join(self.tmp, 'r.jsonl')
        lines = codex_lines(1)
        text = ROLE_TAG % 'drone' + '\n' + self.BRIEF
        lines[2]['payload']['content'] = [{'type': 'input_text', 'text': text}]
        write_jsonl(p, lines)
        sess, _ = c.parse_codex(p)
        self.assertEqual(sess['role'], 'drone')
        got = [(e[1], e[2]) for e in sess['events'] if e[0] == 'B' and e[1][0] in (c.AGENTMSG, c.USER)]
        self.assertEqual(got, [(c.P_BRIEF, len('\n' + self.BRIEF))])

    def test_codex_developer_and_agent_items_stay_put(self):
        def run(item):
            events, pending = [], []
            c._codex_item(item, events, {}, pending)
            return [(e[1], e[2]) for e in events], [(x[0], x[1]) for x in pending]
        dev = '<multi_agent_role>You are a worker agent; the developer sends task briefs.</multi_agent_role>'
        self.assertEqual(run({'type': 'message', 'role': 'developer', 'content': [{'type': 'input_text', 'text': dev}]}),
                         ([(c.P_DEV, len(dev))], []))
        mine = ROLE_TAG % 'drone' + self.BRIEF
        self.assertEqual(run({'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': mine}]}),
                         ([], [(c.P_TEXT, len(mine))]))
        # self-addressed agent_message items are not proven inter-agent deliveries
        am = {'type': 'agent_message', 'author': '/root', 'recipient': '/root',
              'content': [{'type': 'input_text', 'text': 'Message Type: MESSAGE\nSender: /root\nPayload: x'}]}
        self.assertEqual(run(am), ([(c.P_NOTIF, len(am['content'][0]['text']))], []))

    def test_claude_notifications_split_by_launching_tool(self):
        a, b, unknown = self.notif('tA'), self.notif('tB'), self.notif('tZ')
        got = self.claude_blocks([claude_user('go', 3), self.calls(('tA', 'Agent'), ('tB', 'Bash')),
                                  claude_user(a + b + unknown, 1), claude_asst('m2', 1, 10, 0, 0, 5)])
        self.assertEqual(got[c.P_AGENT_NOTIF], len(a))
        self.assertEqual(got[c.P_NOTIF], len(b) + len(unknown))
        mcp = self.claude_blocks([claude_user('go', 3), self.calls(('tA', 'mcp__srv__Agent')),
                                  claude_user(a, 1), claude_asst('m2', 1, 10, 0, 0, 5)])
        self.assertNotIn(c.P_AGENT_NOTIF, mcp)

    def test_claude_queued_commands(self):
        def queued(mode, prompt, origin=None):
            att = {'type': 'queued_command', 'commandMode': mode, 'prompt': prompt}
            if origin:
                att['origin'] = origin
            return {'type': 'attachment', 'timestamp': ts(1), 'attachment': att,
                    'rendered': [{'content': 'x' * 50}]}
        got = self.claude_blocks([claude_user('go', 3), self.calls(('tA', 'Agent'), ('tB', 'Bash')),
                                  queued('task-notification', self.notif('tA')),
                                  queued('task-notification', self.notif('tB')),
                                  queued('prompt', 'agent: ' + self.notif('tA')),
                                  queued('prompt', 'hi', {'kind': 'peer', 'from': 'sess-2', 'senderTaskId': 't'}),
                                  claude_asst('m2', 1, 10, 0, 0, 5)])
        self.assertEqual(got[c.P_AGENT_NOTIF], 50)
        self.assertEqual(got[c.P_NOTIF], 50)
        self.assertEqual(got[c.P_PEER], 50)
        self.assertEqual(got[(c.USER, 'Typed prompts')], len('go') + 50)

    def test_claude_peer_origin(self):
        def user(origin, meta=False):
            o = claude_user('<agent-message>hello</agent-message>', 1)
            o['origin'] = origin
            if meta:
                o['isMeta'] = True
            return o
        n = len('<agent-message>hello</agent-message>')
        for origin, meta, want in (({'kind': 'peer', 'from': 'sess-2', 'senderTaskId': 't'}, True, c.P_PEER),
                                   ({'kind': 'peer', 'from': 'sess-2'}, False, c.P_PEER),
                                   ({'kind': 'peer'}, False, (c.USER, 'Typed prompts')),
                                   ({'kind': 'human'}, False, (c.USER, 'Typed prompts'))):
            got = self.claude_blocks([user(origin, meta), claude_asst('m1', 1, 10, 0, 0, 5)])
            self.assertEqual(got.get(want), n, origin)

    def test_native_subagent_prompt_is_brief_and_own_output_is_not(self):
        lines = [claude_user('Find the callers of foo', 2), claude_asst('m1', 2, 10, 0, 0, 5, text='brief: done'),
                 claude_user('[Request interrupted by user]', 1), claude_asst('m2', 1, 10, 0, 0, 5)]
        sub = self.claude_blocks(lines, is_sub=True)
        top = self.claude_blocks(lines)
        interrupted = len('[Request interrupted by user]')
        self.assertEqual(sub[c.P_BRIEF], top[(c.USER, 'Typed prompts')] - interrupted)
        self.assertEqual(sub[(c.USER, 'Typed prompts')], interrupted)
        self.assertNotIn(c.P_BRIEF, top)
        self.assertEqual(sum(sub.values()), sum(top.values()))

    def test_agent_messages_sit_after_user_messages(self):
        self.assertEqual(c.TOP_ORDER.index(c.AGENTMSG), c.TOP_ORDER.index(c.USER) + 1)

if __name__ == '__main__':
    unittest.main()
