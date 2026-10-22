#!/usr/bin/env python3
"""Collect token usage with a progress display, then open the dashboard in the default browser.

Options not listed below are passed to collect.py, e.g. --lookback 7 or --top-files 20.
"""
import argparse
import multiprocessing
import os
import pathlib
import sys
import time
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import collect  # noqa: E402

APP_ID = 'token-analysis'
PHASES = {'parse': 'Reading session logs', 'check': 'Cross-checking usage', 'attribute': 'Attributing tokens'}
BAR = 30
OWN_COLLECT_OPTIONS = ('--out', '--html', '--template')


def resource_dir():
    """Directory holding template.html: the PyInstaller bundle when frozen, else this file's."""
    return getattr(sys, '_MEIPASS', HERE)


def default_out_dir():
    return os.path.join(os.path.expanduser('~'), 'Library', 'Application Support', APP_ID)


class Progress:
    """Renders collect.run progress: a redrawn bar on a terminal, one line per phase otherwise."""

    def __init__(self, stream, clock=time.monotonic):
        self.stream = stream
        self.tty = stream.isatty()
        self.clock = clock
        self.phase = None
        self.last_draw = None
        self.line_open = False

    def __call__(self, phase, done, total):
        label = PHASES.get(phase, phase)
        if phase != self.phase:
            self.finish()
            self.phase = phase
            if not self.tty:
                self.stream.write('%s (%d)...\n' % (label, total))
                self.stream.flush()
        if not self.tty:
            return
        now = self.clock()
        if done < total and self.last_draw is not None and now - self.last_draw < 0.1:
            return
        self.last_draw = now
        filled = BAR * done // total if total else BAR
        self.line_open = done < total  # end a finished bar at once so collect.py's summary starts on its own line
        self.stream.write('\r%-22s [%s%s] %d/%d%s' % (
            label, '#' * filled, '.' * (BAR - filled), done, total, '' if self.line_open else '\n'))
        self.stream.flush()

    def finish(self):
        if self.line_open:
            self.stream.write('\n')
            self.stream.flush()
        self.line_open = False
        self.phase = None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], epilog=__doc__.split('\n', 2)[2].strip(),
                                 allow_abbrev=False)
    ap.add_argument('--out-dir', default=default_out_dir(),
                    help='where token-data.json and token-drilldown.html go (default ~/Library/Application Support/%s)'
                    % APP_ID)
    ap.add_argument('--no-browser', action='store_true', help='do not open the dashboard in a browser')
    args, collect_args = ap.parse_known_args(argv)
    for a in collect_args:
        name = a.split('=', 1)[0]
        if len(name) > 2 and any(o.startswith(name) for o in OWN_COLLECT_OPTIONS):  # collect.py accepts prefixes
            ap.error('%s is set by token-analysis; use --out-dir instead' % name)

    args.out_dir = os.path.abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, 'token-data.json')
    html = os.path.join(args.out_dir, 'token-drilldown.html')
    print('Collecting token usage from local Claude Code and Codex logs...', file=sys.stderr)
    progress = Progress(sys.stderr)
    try:
        rc = collect.main(['--out', out, '--html', html,
                           '--template', os.path.join(resource_dir(), 'template.html')] + collect_args, progress)
    finally:
        progress.finish()
    if rc:
        return rc
    print('Dashboard: %s' % html)
    if not args.no_browser:
        webbrowser.open(pathlib.Path(html).as_uri())
    return 0


if __name__ == '__main__':
    multiprocessing.freeze_support()  # collect.py's ProcessPoolExecutor re-launches the frozen binary
    sys.exit(main())
