#!/usr/bin/env python3
"""Token-usage collector for the drill-down dashboard.

Reads Claude Code transcripts (~/.claude/projects/**/*.jsonl, subagent files included)
and Codex rollouts (~/.codex/sessions/**/*.jsonl), attributes every logged API call's
tokens to context categories, and writes token-data.json plus the dashboard HTML.

Deterministic, standard library only, no network, read-only on the log directories.
See README.md for the methodology.
"""
import argparse
import bisect
import datetime
import functools
import json
import math
import os
import re
import shlex
import sys
import time
import unicodedata
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

# ---------------------------------------------------------------- config (edit here)
# USD per million tokens: official API rates from platform.claude.com/docs/en/about-claude/pricing
# and developers.openai.com/api/docs/pricing, fetched 2026-10-20. 'long' holds the rates for a call
# whose prompt (all input tokens of that call: uncached + cache reads + cache writes) is above
# long['above'] tokens; each call is tiered on its own prompt. Models without an entry are reported
# but priced at $0 and listed under meta.unpriced_models.
PRICING_SOURCES = ('platform.claude.com/docs/en/about-claude/pricing', 'developers.openai.com/api/docs/pricing')
PRICING_FETCHED = '2026-10-20'
PRICING = {
    'fable-5.1': {'input': 10.00, 'cache_read': 0.25, 'output': 50.0},
    'opus-5.5': {'input': 4.00, 'cache_read': 0.20, 'output': 20.0},
    'sonnet-5.5': {'input': 2.00, 'cache_read': 0.10, 'output': 10.0},
    'haiku-5.5': {'input': 0.10, 'cache_read': 0.01, 'output': 0.50,
                  'long': {'above': 100000, 'input': 0.50, 'cache_read': 0.05, 'output': 2.50}},
    'haiku-4.5': {'input': 1.00, 'cache_read': 0.10, 'output': 5.0},
    'gpt-6.1-sol': {'input': 2.00, 'cache_read': 0.10, 'output': 10.0,
                    'long': {'above': 272000, 'input': 4.00, 'cache_read': 0.20, 'output': 15.0}},
    'gpt-6-luna': {'input': 0.10, 'cache_read': 0.01, 'output': 0.50,
                   'long': {'above': 272000, 'input': 0.20, 'cache_read': 0.02, 'output': 0.75}},
    'gpt-6-astra': {'input': 10.00, 'cache_read': 1.00, 'output': 50.0,
                    'long': {'above': 272000, 'input': 20.00, 'cache_read': 2.00, 'output': 75.0}},
}
# Anthropic cache writes are billed as a multiple of the input rate, per TTL (true for every model above).
# Other providers' cache writes are billed at the input rate.
CACHE_WRITE_MULT = {'5m': 1.25, '1h': 2.0}
MODEL_ALIASES = {
    'claude-fable-5-1': 'fable-5.1',
    'claude-opus-5-5': 'opus-5.5',
    'claude-sonnet-5-5': 'sonnet-5.5',
    'claude-haiku-5-5': 'haiku-5.5',
    'claude-haiku-4-5': 'haiku-4.5',
    'gpt-6.1-sol': 'gpt-6.1-sol',
    'gpt-6-luna': 'gpt-6-luna',
    'gpt-6-astra': 'gpt-6-astra',
}
# Raw IDs may carry a context-window tag ("[1m]") or a snapshot date ("-20251001"); both are dropped
# before the alias lookup.
MODEL_SUFFIX_RE = re.compile(r'(?:\[[^\]]*\]|-\d{8})+$')
MODEL_LABELS = {'fable-5.1': 'Fable 5.1', 'opus-5.5': 'Opus 5.5', 'sonnet-5.5': 'Sonnet 5.5',
                'haiku-5.5': 'Haiku 5.5', 'haiku-4.5': 'Haiku 4.5'}
PRIMARY_MODELS = ['opus-5.5', 'haiku-5.5', 'gpt-6-luna', 'gpt-6-astra']
# Display order of the known roles; any other role found in a subagent profile tag comes after them.
ROLES = [('top', 'Top-level'), ('envoy', 'Envoy'), ('drone', 'Drone'), ('review', 'Review'),
         ('native', 'Native subagent')]
IMAGE_TOKENS = 1600          # fixed estimate per image block (no pixel sizes in logs)
FALLBACK_CHARS_PER_TOKEN = 3.5
CALIBRATION_BOUNDS = (1.5, 6.0)
MAX_CHILDREN = 9             # long tails beyond this are merged into "Other …"
MIN_DAYS = 31                # the timeline always spans more than 30 days, so the 30-day preset starts on a dated day
FILE_TOP_N = 12              # per-file instruction nodes keep this many files, the rest become "Other (k files)"
CMD_TOP_N = 12               # shell tool nodes keep this many commands, the rest become "Other (k commands)"
# Upper edges of the per-occurrence size bins (tokens, 1-2-5 steps): bin 0 is <100, bin 13 is >=1M.
HIST_EDGES = (100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000)
CLAUDE_ROOT = '~/.claude/projects'
CODEX_ROOT = '~/.codex/sessions'

HOME = os.path.expanduser('~')
HOME_RE = re.compile(r'^/(?:Users|home)/[^/]+(?=/|$)')
HOME_SLUG_RE = re.compile(r'-(?:Users|home)-[A-Za-z0-9_]+')
AGENTS_HDR_RE = re.compile(r'AGENTS\.md instructions for ([^\n]+)')
# Codex sends AGENTS.md as a user message: the header line, then the docs wrapped in <INSTRUCTIONS>.
AGENTS_DOC_RE = re.compile(r'\A\s*# AGENTS\.md instructions for [^\n]+\n.*?</INSTRUCTIONS>', re.S)
WIN_ABS_RE = re.compile(r'^(?:[A-Za-z]:/|//)')
WIN_HOME_RE = re.compile(r'^(?:[A-Za-z]:)?/+(?:[^/]+/){0,2}(?:Users|home)/[^/]+(?=/|$)', re.I)
PROJECT_DOC_SEP = '--- project-doc ---'
WT_DIR_RE = re.compile(r'^\.?worktrees/[^/]+/')
# A role is a short lowercase slug; anything else is not a profile tag and stays prompt text.
ROLE_TAG = r'<t3-subagent-profile role="([a-z][a-z0-9-]{0,31})">'
ROLE_RE = re.compile(ROLE_TAG)
PROFILE_SPAN_RE = re.compile(ROLE_TAG + r'.*?</t3-subagent-profile>', re.S)
SYSREM_RE = re.compile(r'<system-reminder>.*?</system-reminder>', re.S)
PASTE_RE = re.compile(r'<pasted_content\b.*?</pasted_content>', re.S)
NOTIF_RE = re.compile(r'<task-notification>.*?</task-notification>', re.S)
ENVCTX_RE = re.compile(r'<environment_context>.*?</environment_context>', re.S)
PERM_RE = re.compile(r'<permissions instructions>.*?</permissions instructions>', re.S)
COMPACT_PREFIX = 'This session is being continued from a previous conversation'
T3_MARKERS = [('## T3 Code orchestration', 'Orchestration guidance'),
              ('<pull_request_linking>', 'PR linking'),
              ('### Showing visuals', 'Visuals / HTML guidance'),
              ('<runtime_info>', 'Runtime info')]
SUBAGENT_TOOLS = {'Agent', 'Task', 'TaskOutput', 't3-code:delegate_task', 't3-code:task_status',
                  't3-code:t3_thread_wait', 't3-code:t3_thread_read', 'collaboration.wait_agent'}

# Category paths (top level keeps the dashboard's existing categories).
INSTR, TOOLCTX, CALLS, RESULTS, USER, ASSIST, SUBRES = (
    'Agent instructions', 'Tool context', 'Tool calls', 'Tool results', 'User messages',
    'Assistant output', 'Subagent results')
UNATTRIB = ('Unattributed prefix',)
READ_RESULT = (RESULTS, 'Built-in tools', 'Read')
P_THINK = (ASSIST, 'Thinking')
P_TEXT = (ASSIST, 'Visible text')
TOP_ORDER = [INSTR, TOOLCTX, CALLS, RESULTS, USER, ASSIST, SUBRES, UNATTRIB[0]]

# Basis labels shown in tooltips.
B_EST = 'estimated (chars/token)'
B_IMG = 'estimated (fixed per image)'
B_OUT = 'estimated split of measured output'
B_MEAS = 'measured'
B_RESID = 'residual (measured context − estimates)'


# ---------------------------------------------------------------- helpers
def role_label(role):
    """Display label of a role: the fixed one for a known role, else the slug as words ("envoy-peer" ->
    "Envoy peer")."""
    return dict(ROLES).get(role) or role.replace('-', ' ').capitalize()


def role_order(roles):
    """The roles found, known ones in their fixed order, then the others sorted."""
    known = [r for r, _ in ROLES]
    return [r for r in known if r in roles] + sorted(set(roles) - set(known))


def model_id(raw):
    if raw in MODEL_ALIASES:
        return MODEL_ALIASES[raw]
    base = MODEL_SUFFIX_RE.sub('', raw)
    return MODEL_ALIASES.get(base, raw)


TS_RE = re.compile(r'(\d{4})-(\d\d)-(\d\d)[T ](\d\d):(\d\d)(?::(\d\d)(?:[.,](\d+))?)?\s*(Z|[+-]\d\d(?::?\d\d)?)?', re.I)


def parse_ts(s, naive_tz=datetime.timezone.utc):
    """ISO 8601 -> aware datetime. Honors `Z` and ±HH[:MM] offsets; naive input gets naive_tz."""
    m = TS_RE.fullmatch(s.strip()) if isinstance(s, str) else None
    if not m:
        return None
    y, mo, d, h, mi, sec, frac, off = m.groups()
    try:
        dt = datetime.datetime(int(y), int(mo), int(d), int(h), int(mi), int(sec or 0),
                               int((frac or '0')[:6].ljust(6, '0')))
    except ValueError:
        return None
    if off is None:
        return dt.replace(tzinfo=naive_tz) if naive_tz else dt.astimezone()
    if off.upper() == 'Z':
        return dt.replace(tzinfo=datetime.timezone.utc)
    sign = -1 if off[0] == '-' else 1
    digits = off[1:].replace(':', '')
    delta = datetime.timedelta(hours=int(digits[:2]), minutes=int(digits[2:4] or 0))
    return dt.replace(tzinfo=datetime.timezone(sign * delta))


def ts_epoch(s):
    dt = parse_ts(s)
    return dt.timestamp() if dt else None


def chars_of(x):
    """Total text length of a JSON-ish value; images return 0 (sized separately)."""
    if x is None:
        return 0
    if isinstance(x, str):
        return len(x)
    if isinstance(x, list):
        return sum(chars_of(i) for i in x)
    if isinstance(x, dict):
        t = x.get('type')
        if t in ('image', 'input_image'):
            return 0
        if 'text' in x and isinstance(x['text'], str):
            return len(x['text'])
        if 'content' in x:
            return chars_of(x['content'])
        return 0
    return 0


def count_images(x):
    if isinstance(x, list):
        return sum(count_images(i) for i in x)
    if isinstance(x, dict):
        if x.get('type') in ('image', 'input_image'):
            return 1
        return count_images(x.get('content'))
    return 0


def server_label(s, codex=False):
    s = re.sub(r'^claude_ai_', '', s)
    s = re.sub(r'^plugin_[^_]+_', '', s)
    return s.replace('_', '-') if codex else s.replace('_', ' ')


def tool_key(name, namespace=None):
    """Returns (server or None, tool label). Server None means built-in."""
    if namespace:
        if namespace.startswith('mcp__'):
            return server_label(namespace[5:].strip('_'), codex=True), name
        return None, '%s.%s' % (namespace, name)
    m = re.match(r'^mcp__(.+?)__(.+)$', name or '')
    if m:
        return server_label(m.group(1)), m.group(2)
    return None, name or '?'


def flat_tool(server, tool):
    return tool if server is None else '%s:%s' % (server, tool)


def call_path(server, tool):
    return (CALLS, 'Built-in tools', tool) if server is None else (CALLS, 'MCP servers', server, tool)


def result_path(server, tool):
    if flat_tool(server, tool) in SUBAGENT_TOOLS:
        return (SUBRES, flat_tool(server, tool))
    return (RESULTS, 'Built-in tools', tool) if server is None else (RESULTS, 'MCP servers', server, tool)


def schema_path(server, tool):
    return (TOOLCTX, 'Built-in tools', tool) if server is None else (TOOLCTX, 'MCP tool schemas', server, tool)


# ---------------------------------------------------------------- shell command labels
# Built-in tools whose calls and results are split by the command they ran. Codex 'exec' is a
# code-mode script; its command is the first tools.exec_command({cmd: ...}) inside it.
SHELL_TOOLS = {'Bash', 'exec', 'exec_command', 'shell', 'shell_command', 'local_shell'}
CMD_UNPARSED = 'Unparsed'
CMD_NO_SHELL = 'No shell command'      # exec scripts that call no exec_command
CMD_REJECTED = 'Other commands'        # programs that are not on the allowlist
HEREDOC_RE = re.compile(r'(?<!<)<<(?!<)-?\s*([\'"]?)([A-Za-z_][A-Za-z0-9_]*)\1')
SHELLS = {'bash', 'sh', 'zsh', 'dash', 'ksh', 'fish'}
# Labels come only from these lists, so a script, alias or argument the user named can never become one.
PROGRAMS = set('''
    cat head tail less more sed gsed awk gawk grep egrep fgrep rg ag ack fd find sort uniq wc cut tr tee nl paste
    join comm diff cmp patch column fold fmt rev tac od xxd hexdump strings file stat ls tree du df pwd echo printf
    mkdir rmdir rm cp mv ln touch chmod chown chgrp realpath readlink basename dirname date sleep which whereis type
    whoami id uname hostname printenv kill pkill pgrep ps pstree top htop lsof jq yq base64 shasum sha256sum sha1sum
    md5 md5sum openssl tar zip unzip gzip gunzip zcat bzip2 xz curl wget ssh scp rsync nc ping dig nslookup host
    mktemp mkfifo seq split read exit uptime man history getconf watch tmux screen vim nvim nano code perl ruby lua
    git gh glab git-lfs git-ai node npm npx pnpm pnpx yarn bun bunx deno corepack fnm nvm tsc tsx ts-node eslint
    prettier biome oxlint oxfmt vitest jest mocha playwright playwright-cli cypress vite next webpack esbuild rollup
    turbo nx lerna knip commitlint lint-staged python python3 pip pip3 pipx uv uvx poetry pytest ruff black isort
    mypy pyright ty flake8 pylint tox nox conda pyenv lint-imports gem bundle rake rails go gofmt cargo rustc rustup
    rustfmt java javac mvn gradle kotlin swift swiftc xcrun xcodebuild clang gcc g++ cc make cmake ninja bazel just
    docker docker-compose podman kubectl helm terraform tofu aws gcloud az brew apt apt-get sqlite3 psql mysql
    redis-cli ffmpeg ffprobe magick convert montage sips codex claude opencode ollama pre-commit stow direnv open
    osascript defaults security launchctl systemctl plutil mdfind mdls pbcopy pbpaste say sw_vers sysctl vm_stat
    log diskutil ditto codesign spctl hdiutil pkgutil lsregister lsappinfo pmset screencapture terminal-notifier
    caffeinate acli jira mise
'''.split())
# Programs whose subcommand is shown ("git status", "pnpm test"), with the subcommands that may appear.
# Any other word after the program is dropped, so "make my-private-target" shows as "make".
_PKG = 'run test install i ci add remove rm uninstall exec dlx x create init build lint start dev typecheck format ' \
       'list ls outdated update upgrade why view info publish pack audit link unlink store config prune rebuild ' \
       'dedupe patch version whoami'
# Tools that pnpm, yarn and bun run by name from node_modules/.bin ("pnpm vitest").
_PKG_BIN = ' vitest jest tsc eslint prettier biome oxlint playwright turbo knip'
SUBCOMMANDS = {k: set(v.split()) for k, v in {
    'git': 'status diff log show add commit push pull fetch checkout switch branch merge rebase reset restore stash '
           'tag rev-parse rev-list ls-files ls-tree ls-remote grep blame config remote worktree submodule cherry-pick '
           'revert clean describe shortlog cat-file for-each-ref symbolic-ref merge-base merge-tree name-rev reflog '
           'apply am format-patch bisect init clone mv rm notes range-diff sparse-checkout lfs gc fsck check-ignore '
           'update-index hash-object show-ref diff-tree read-tree archive var help version',
    'gh': 'pr issue repo run workflow api auth release gist search browse label status cache secret variable '
          'codespace extension alias config project ruleset attestation',
    'npm': _PKG, 'pnpm': _PKG + _PKG_BIN, 'yarn': _PKG + _PKG_BIN, 'bun': _PKG + _PKG_BIN,
    'mise': 'run exec x install use ls list trust tasks env where which upgrade doctor',
    'uv': 'run pip sync add remove lock venv tool python init build publish tree export cache self version',
    'cargo': 'build test run check clippy fmt doc add remove update install new init bench tree metadata publish '
             'clean fix',
    'go': 'build test run vet fmt mod get install generate env version list doc tool work clean',
    'docker': 'ps run exec build images pull push logs compose stop start rm rmi inspect cp network volume system '
              'info version tag login buildx kill restart stats top container image context',
    'kubectl': 'get describe apply delete logs exec port-forward config create edit rollout scale top explain diff '
               'patch label annotate auth cluster-info version wait cp',
    'helm': 'install upgrade uninstall list template repo dependency lint status history rollback show package pull '
            'search version get',
    'terraform': 'init plan apply destroy validate fmt output state import workspace show providers refresh graph '
                 'console version',
    'brew': 'install uninstall list info search update upgrade services doctor outdated cleanup tap untap link '
            'unlink reinstall deps uses leaves config',
    'pip': 'install uninstall list show freeze download wheel check config cache index inspect',
    'pipx': 'install uninstall list run upgrade ensurepath inject',
    'poetry': 'install add remove run update lock show build publish shell env check init new version export',
    'make': 'test build lint install clean all check format fmt dev run setup deploy release help docs typecheck ci '
            'serve start watch generate e2e',
    'codex': 'exec resume login logout mcp apply review sandbox app app-server proto completion cloud features '
             'plugin debug',
    'claude': 'mcp config update doctor install plugin setup-token',
    'pre-commit': 'run install autoupdate clean gc uninstall validate-config try-repo',
    'deno': 'run test fmt lint check task compile bench info install upgrade eval doc',
    'rustup': 'update install default show toolchain target component override run which self',
    'conda': 'install create activate deactivate list env remove update info run config search',
    'swift': 'build test run package',
    'xcrun': 'simctl xcodebuild swift clang notarytool stapler devicectl xctrace',
    'defaults': 'read write delete domains find export import',
    'security': 'find-generic-password find-internet-password find-certificate add-generic-password '
                'delete-generic-password dump-keychain list-keychains unlock-keychain cms',
    'systemctl': 'status start stop restart enable disable reload list-units daemon-reload is-active show cat',
    'launchctl': 'list load unload bootstrap bootout kickstart print start stop enable disable',
    'aws': 's3 s3api ec2 iam sts lambda ecr eks cloudformation logs ssm secretsmanager dynamodb sqs sns rds configure',
    'gcloud': 'auth config compute container projects iam storage run functions sql secrets logging services '
              'components',
    'az': 'login account group vm aks acr storage keyvault ad webapp functionapp monitor network resource role',
}.items()}
for _k in ('pip3',):
    SUBCOMMANDS[_k] = SUBCOMMANDS['pip']
for _k in ('gradle', 'mvn'):
    SUBCOMMANDS[_k] = {'clean', 'compile', 'test', 'package', 'install', 'verify', 'deploy', 'build', 'check',
                       'assemble', 'run', 'dependencies', 'tasks'}
SUBCOMMANDS['just'] = SUBCOMMANDS['make']
SUBCOMMANDS['tofu'] = SUBCOMMANDS['terraform']
for _k in ('nx', 'turbo'):
    SUBCOMMANDS[_k] = {'run', 'build', 'test', 'lint', 'affected', 'graph', 'serve', 'dev', 'generate', 'format',
                       'show', 'reset', 'daemon', 'prune'}
# Package runners show the tool they run when it is on the program allowlist ("npx vitest").
RUNNERS = {'npx', 'pnpx', 'bunx', 'uvx'}
PY_MODULES = set('''
    pytest pip venv unittest json.tool http.server py_compile compileall mypy black ruff isort flake8 pylint build
    twine coverage pdb cProfile timeit doctest site ensurepip tokenize ast zipfile tarfile webbrowser pydoc uvicorn
    gunicorn flask django ipykernel jupyter nbconvert poetry pipx tox nox pre_commit playwright
'''.split())
PY_RE = re.compile(r'^python(?:3(?:\.\d{1,2})?)?$')
# Options are keyed by program because the same spelling can take a value in one
# command and be a flag in another (for example, npx -p <package> vs git -p).
PROGRAM_OPT_WITH_ARG = {
    'git': {'-C', '-c', '--git-dir', '--work-tree', '--namespace', '--config-env'},
    'gh': {'-R', '--repo', '-H', '--hostname'},
    'npm': {'--prefix', '--workspace', '-w', '--registry', '--cache'},
    'pnpm': {'-C', '--dir', '--filter', '-F', '--config', '-p', '--package', '--reporter', '--loglevel',
             '--workspace-concurrency'},
    'yarn': {'--cwd', '--config', '--registry'},
    'bun': {'--cwd', '--config'},
    'npx': {'-p', '--package'},
    'pnpx': {'-p', '--package'},
    'bunx': {'-p', '--package'},
    'uvx': {'--from', '--with', '--with-editable', '--with-requirements', '--python'},
    'uv': {'-C', '--directory', '--project', '--config-file', '--python', '--with'},
    'cargo': {'-p', '--package', '--manifest-path'},
    'go': {'-C'},
    'docker': {'-H', '--host', '-c', '--context', '--config', '-l', '--log-level'},
    'kubectl': {'-n', '--namespace', '-f', '--filename', '--context', '--kubeconfig', '--cluster', '--user', '-v'},
    'helm': {'-n', '--namespace', '--kubeconfig', '--kube-context', '--repository-config'},
    'terraform': {'-chdir'},
    'tofu': {'-chdir'},
    'pip': {'-c', '--constraint', '-r', '--requirement'},
    'pip3': {'-c', '--constraint', '-r', '--requirement'},
    'make': {'-C', '-f', '--file'},
    'codex': {'-C', '--cd', '-m', '--model', '-p', '--profile', '-c', '--config', '-s', '--sandbox', '-a',
              '--ask-for-approval', '-i', '--image', '--add-dir'},
    'claude': {'--model', '--add-dir', '--session-id', '--permission-mode', '--output-format', '--input-format',
               '--mcp-config', '--settings', '--fallback-model', '--append-system-prompt'},
    'pre-commit': {'-c', '--config'},
    'deno': {'--config', '--import-map', '--lock'},
    'rustup': {'-t', '--toolchain'},
    'conda': {'-n', '--name', '-p', '--prefix', '-c', '--channel'},
    'swift': {'--package-path'},
    'xcrun': {'--sdk', '--toolchain'},
    'aws': {'--profile', '--region', '--endpoint-url'},
    'gcloud': {'--project', '--configuration', '--account', '--verbosity', '--format'},
    'az': {'--subscription', '--output'},
    'nx': {'--project', '--configuration'},
    'turbo': {'--filter', '--scope'},
}
PROGRAM_OPT_FLAGS = {
    'git': {'--no-pager', '-P', '--paginate', '--bare', '--no-replace-objects', '--literal-pathspecs', '-p',
            '--no-optional-locks'},
    'npm': {'--silent', '-s', '--offline', '--dry-run', '--force', '-y', '--yes', '--workspaces', '-ws',
            '--include-workspace-root', '--if-present'},
    'pnpm': {'--silent', '-s', '--offline', '--frozen', '--frozen-lockfile', '--locked', '--recursive', '-r',
             '--dry-run', '--force', '-y', '--yes', '-w', '--workspace-root', '--workspace', '--parallel',
             '--stream', '--if-present', '--aggregate-output', '--prefer-offline'},
    'yarn': {'--silent', '--offline', '--immutable', '--frozen-lockfile', '--verbose'},
    'bun': {'--silent', '--offline', '--frozen-lockfile', '--verbose'},
    'npx': {'--yes', '-y', '--no-install', '--offline'},
    'pnpx': {'--yes', '-y', '--no-install', '--offline'},
    'bunx': {'--yes', '-y', '--no-install'},
    'cargo': {'--locked', '--offline', '--frozen', '--verbose', '-v', '--quiet', '-q'},
    'docker': {'--tls', '--tlsverify', '--debug'},
    'terraform': {'-json', '-no-color', '-input=false'},
    'tofu': {'-json', '-no-color', '-input=false'},
    'pip': {'--quiet', '-q', '--verbose', '-v'},
    'pip3': {'--quiet', '-q', '--verbose', '-v'},
    'codex': {'--yolo', '--full-auto', '--dangerously-bypass-approvals-and-sandbox', '--oss', '--search'},
    'claude': {'-p', '--print', '-c', '--continue', '--verbose', '--dangerously-skip-permissions'},
    'deno': {'--cached-only', '--frozen', '--no-check'},
    'conda': {'-y', '--yes', '--offline'},
    'aws': {'--no-cli-pager', '--no-sign-request'},
    'gcloud': {'--quiet', '-q', '--no-user-output-enabled'},
}
COMMON_OPT_FLAGS = {'--help', '-h', '--version', '-v', '--verbose', '-q', '--quiet'}

# Prefix commands that run the next word. Options taking a value are wrapper-specific;
# wrapper arguments are never considered candidate program names.
WRAPPERS = {'sudo', 'doas', 'time', 'nice', 'nohup', 'env', 'command', 'builtin', 'exec', 'noglob',
            'timeout', 'gtimeout', 'stdbuf', 'caffeinate', 'xargs'}
WRAPPER_OPT_WITH_ARG = {
    'sudo': {'-u', '--user', '-g', '--group', '-C', '-D', '--chdir', '-h', '--host', '-p', '--prompt',
             '-r', '--role', '-t', '--type', '--close-from', '-U', '--other-user', '-T', '--command-timeout'},
    'doas': {'-u', '--user', '-C'},
    'nice': {'-n', '--adjustment'},
    'env': {'-u', '--unset', '-C', '--chdir', '-S', '--split-string', '-P'},
    'exec': {'-a'},
    'timeout': {'-s', '--signal', '-k', '--kill-after'},
    'gtimeout': {'-s', '--signal', '-k', '--kill-after'},
    'stdbuf': {'-i', '--input', '-o', '--output', '-e', '--error'},
    'caffeinate': {'-t', '-w'},
    'xargs': {'-a', '--arg-file', '-I', '-n', '--max-args', '-P', '--max-procs', '-L',
              '-d', '--delimiter', '-s', '--max-chars', '-E', '-J', '-R', '-S', '--process-slot-var'},
    'time': {'-o', '--output', '-f', '--format'},
}
# Wrapper options that take no value. An option in neither table may take a value, so the word after it
# could be that value rather than the program: such segments are labelled Unparsed.
WRAPPER_OPT_FLAGS = {
    'sudo': {'-A', '--askpass', '-B', '--bell', '-b', '--background', '-E', '--preserve-env', '-e', '--edit',
             '-H', '--set-home', '-i', '--login', '-K', '--remove-timestamp', '-k', '--reset-timestamp', '-l',
             '--list', '-N', '--no-update', '-n', '--non-interactive', '-P', '--preserve-groups', '-S', '--stdin',
             '-s', '--shell', '-V', '--version', '-v', '--validate'},
    'doas': {'-n', '-s', '-L'},
    'time': {'-p', '--portability', '-v', '--verbose', '-a', '--append', '-l', '-q', '--quiet'},
    'env': {'-i', '--ignore-environment', '-0', '--null', '-v', '--debug', '-'},
    'command': {'-p', '-v', '-V'},
    'exec': {'-c', '-l'},
    'timeout': {'--foreground', '--preserve-status', '-v', '--verbose', '-f', '-p'},
    'gtimeout': {'--foreground', '--preserve-status', '-v', '--verbose'},
    'caffeinate': {'-d', '-i', '-m', '-s', '-u'},
    'xargs': {'-0', '--null', '-r', '--no-run-if-empty', '-t', '--verbose', '-p', '--interactive', '-x', '--exit',
              '-o', '--open-tty', '-e', '-i', '-l', '--eof', '--replace', '--max-lines'},
    # GNU -e/-i/-l and --eof/--replace/--max-lines take an optional value only when attached (-i{}, --replace={})
}
WRAPPER_OPT_ATTACHED_ONLY = {'xargs': {'-e', '-i', '-l'}, 'nice': set()}
NICE_NUM_RE = re.compile(r'^-\d+$')
WRAPPER_SPLIT_OPTS = {'env': {'-S', '--split-string'}}
WRAPPER_POSITIONAL = {'timeout': 1, 'gtimeout': 1}
# Segments that only set up state: the label goes to the next segment when there is one.
SETUP_CMDS = {'cd', 'pushd', 'popd', 'export', 'set', 'unset', 'source', '.', 'true', ':', 'shopt', 'ulimit',
              'umask', 'trap', 'alias', 'local', 'declare', 'typeset', 'readonly', 'eval', '[', 'test', 'continue',
              'break', 'return', 'shift', 'let', 'setopt'}
SHELL_KEYWORDS = {'if', 'then', 'else', 'elif', 'while', 'until', 'do', '!', '{', '(', '[[', '{{'}
SKIP_SEGMENTS = {'fi', 'done', 'esac', '}', 'for', 'case', 'select', 'function', ')', ']]'}
SEPARATORS = {';', '&&', '||', '|', '&', '|&', ';;', '(', ')', '\n'}
REDIRECTS = {'>', '>>', '<', '<<', '<<<', '>&', '<&', '&>', '&>>', '>|', '<>'}
ENV_ASSIGN_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*(?:\[[^\]]*\])?\+?=')
# shlex joins adjacent operator characters (");", ")|"); this splits such runs back into operators.
OPERATOR_RE = re.compile(r'\(\)|;;|&&|\|\||\|&|&>>|&>|>>|>&|<&|>\||<>|<<<|<<|[;&|()<>]')


def _subcommand(prog, rest):
    """The allowlisted subcommand after the program's options, else None.
    A word right after an option neither table knows may be that option's argument, so the next word is
    tried instead ("docker --tlsdir d ps"). When that next word is not allowlisted, or both words are
    ("docker --tlsdir login ps"), it is unclear which one is the subcommand, so none is shown."""
    allowed = PROGRAMS if prog in RUNNERS else SUBCOMMANDS[prog]
    takes_arg = PROGRAM_OPT_WITH_ARG.get(prog, set())
    flags = PROGRAM_OPT_FLAGS.get(prog, set()) | COMMON_OPT_FLAGS
    words, j, after_opt = [], 0, False
    while j < len(rest) and len(words) < 2:
        t = rest[j]
        if t.startswith(('-', '+')):
            option = t.split('=', 1)[0]
            if option in takes_arg:
                j += 1 if '=' in t else 2
                after_opt = False
            else:
                j += 1
                after_opt = '=' not in t and option not in flags
            continue
        words.append((t, after_opt))
        if not after_opt:
            break
        after_opt = False
        j += 1
    if not words:
        return None
    (first, maybe_arg), second = words[0], words[1][0] if len(words) > 1 else None
    if not maybe_arg:
        return first if first in allowed else None
    if second in allowed:
        return None if first in allowed else second
    return None


def _strip_heredocs(s):
    """Drop here-document bodies; their text is data, not commands."""
    lines, out, i = s.split('\n'), [], 0
    while i < len(lines):
        out.append(lines[i])
        delims = [m.group(2) for m in HEREDOC_RE.finditer(lines[i])]
        i += 1
        for d in delims:
            while i < len(lines) and lines[i].strip() != d:
                i += 1
            i += 1
    return '\n'.join(out)


def _shell_words(s, arith=False):
    """Shell words with control operators as separate items; None when the line can't be tokenized.
    With arith, a run starting with two adjacent "((" keeps them as one "((" item, which may open
    arithmetic; "( (" stays two subshell openers."""
    s = _strip_heredocs(s.replace('\r', '')).replace('\\\n', ' ')
    try:
        lx = shlex.shlex(s.replace('\n', ' \n '), posix=True, punctuation_chars=';&|()<>')
        lx.whitespace = ' \t'
        lx.whitespace_split = True
        lx.commenters = ''
        words = list(lx)
    except ValueError:
        return None
    out = []
    for w in words:
        if arith and w.startswith('(('):
            out.append('((')
            out.extend(OPERATOR_RE.findall(w[2:]))
        elif w and not w.strip(';&|()<>') and w not in SEPARATORS and w not in REDIRECTS and w != '()':
            out.extend(OPERATOR_RE.findall(w))
        else:
            out.append(w)
    return out


def _segments(words):
    seg = []
    skip_next = False
    in_array = in_case = in_arith = False
    prev = None
    words, i = list(words), 0
    while i < len(words):
        w = words[i]
        i += 1
        if w == '((' and not in_arith:
            if not in_array and (not seg or seg == ['for'] or seg[-1].endswith('$')):
                # $(( … )) is an expansion inside the current command; (( … )) and for (( … )) are not commands
                in_arith, prev = True, None
                seg = seg if seg and seg[-1].endswith('$') else []
                continue
            words[i:i] = ['(']  # not arithmetic: two subshell openers
            w = '('
        if skip_next:
            skip_next = False
            continue
        if in_arith:  # for (( init; test; step )) and (( expr )): arithmetic, not commands
            in_arith = not (w == ')' and prev == ')')
            prev = w
            continue
        prev = w
        if in_array:  # NAME=(a b c): the elements are data
            in_array = w != ')'
            continue
        if w == '(' and seg and len(seg) == 1 and seg[0].endswith('=') and ENV_ASSIGN_RE.match(seg[0]):
            in_array = True
            continue
        if seg and seg[0] == 'case':
            in_case = True
        if w == 'esac' and not seg:
            in_case = False
        if w in SEPARATORS:
            if seg and not (in_case and w == ')' and seg[0] != 'case'):  # case patterns are not commands
                yield seg
            seg = []
        elif w in REDIRECTS or (w and w[0] in '<>' and not w.strip('<>&|')):
            if seg and seg[-1].isdigit():
                seg.pop()  # the file descriptor of "2>file"
            skip_next = True
        else:
            seg.append(w)
    if seg:
        yield seg


def _wrapper_option(w, t):
    """Words consumed by wrapper w's option word t (1 or 2), or None when t is not a known option.
    Short options may be clustered ("-0r") and may carry their value ("-n1", "-I{}")."""
    takes_arg, flags = WRAPPER_OPT_WITH_ARG.get(w, set()), WRAPPER_OPT_FLAGS.get(w, set())
    option = t.split('=', 1)[0]
    if t.startswith('--'):
        if option in takes_arg:
            return 1 if '=' in t else 2
        return 1 if option in flags else None
    if option in takes_arg and '=' not in t:
        return 2
    if option in flags:
        return 1
    if w == 'nice' and NICE_NUM_RE.match(t):
        return 1
    for j in range(1, len(t)):
        o = '-' + t[j]
        if o in takes_arg:
            return 1 if j + 1 < len(t) else 2
        if o in WRAPPER_OPT_ATTACHED_ONLY.get(w, ()):
            return 1
        if o not in flags:
            return None
    return 1


def _segment_label(seg, depth):
    """(label or None, is_setup) for one simple command."""
    i = 0
    while i < len(seg) and (seg[i] in SHELL_KEYWORDS or ENV_ASSIGN_RE.match(seg[i])):
        i += 1
    while i < len(seg) and os.path.basename(seg[i]) in WRAPPERS:
        w = os.path.basename(seg[i])
        i += 1
        npos = WRAPPER_POSITIONAL.get(w, 0)
        while i < len(seg):
            t = seg[i]
            if t == '--':
                i += 1
                while npos and i < len(seg):
                    npos -= 1
                    i += 1
                break
            if t.startswith('-'):
                option = t.split('=', 1)[0]
                if w in WRAPPER_SPLIT_OPTS and option in WRAPPER_SPLIT_OPTS[w]:
                    if '=' in t:
                        split_string, after = t.split('=', 1)[1], i + 1
                    elif i + 1 < len(seg):
                        split_string, after = seg[i + 1], i + 2
                    else:
                        return CMD_UNPARSED, False
                    try:
                        split_words = shlex.split(split_string)
                    except ValueError:
                        return CMD_UNPARSED, False
                    seg = seg[:i] + split_words + seg[after:]
                    continue
                step = _wrapper_option(w, t)
                if step is None:
                    return CMD_UNPARSED, False
                i += step
            elif ENV_ASSIGN_RE.match(t):
                i += 1
            elif npos:
                npos -= 1
                i += 1
            else:
                break
    if i >= len(seg):
        return None, True
    word = seg[i]
    if word in SKIP_SEGMENTS:
        return None, True
    if word.startswith(('$', '`', '"', "'")) or '$(' in word or not word.strip():
        return CMD_UNPARSED, False
    if ':' in word:  # URLs, host:path and drive paths are never programs
        return CMD_REJECTED, False
    prog = os.path.basename(word.rstrip('/'))
    rest = seg[i + 1:]
    if rest[:1] == ['()'] or (prog == 'function' and rest):
        return None, True  # function definition; its body is the next segment
    if prog in SHELLS and depth < 3:
        for j, t in enumerate(rest):
            if t.startswith('-') and not t.startswith('--') and 'c' in t[1:] and j + 1 < len(rest):
                return _line_label(rest[j + 1], depth + 1), False
    if prog in SETUP_CMDS:
        return ('test' if prog == '[' else prog), True
    sub = None
    if PY_RE.match(prog):
        for j, t in enumerate(rest):
            if t == '-m' and j + 1 < len(rest):
                sub = '-m ' + rest[j + 1] if rest[j + 1] in PY_MODULES else '-m'
                break
            if t == '-c':
                sub = '-c'
                break
            if not t.startswith('-'):
                break
    elif prog not in PROGRAMS and prog not in SHELLS:
        return CMD_REJECTED, False
    elif prog in SUBCOMMANDS or prog in RUNNERS:
        sub = _subcommand(prog, rest)
    return (prog + ' ' + sub if sub else prog), False


def _line_label(line, depth=0):
    words = _shell_words(line, arith=True) if isinstance(line, str) else None
    if not words:
        return CMD_UNPARSED
    setup = None
    funcs, defining = {}, None
    for seg in _segments(words):
        if len(seg) >= 2 and (seg[1] == '()' or seg[0] == 'function'):
            defining = seg[1] if seg[0] == 'function' else seg[0]
            seg = [w for w in seg[2:] if w not in ('()', '{')]
            if not seg:
                continue
        if not defining and seg[0] in funcs:
            return funcs[seg[0]]
        label, is_setup = _segment_label(seg, depth)
        if label is None:
            continue
        if defining:
            if not is_setup:
                funcs[defining], defining = label, None
            continue
        if not is_setup:
            return label
        setup = setup or label
    return setup or CMD_UNPARSED


def command_label(cmd):
    """Privacy-safe label for a shell command: the program's basename plus a subcommand for multi-command
    tools. cmd is a command line or an argv list. Arguments, paths and quoted text never appear."""
    if isinstance(cmd, (list, tuple)):
        argv = [str(x) for x in cmd]
        if not argv:
            return CMD_UNPARSED
        if os.path.basename(argv[0]) in SHELLS and len(argv) >= 3 and argv[-2].startswith('-') and 'c' in argv[-2]:
            return _line_label(argv[-1])
        return _line_label(' '.join(shlex.quote(a) for a in argv))
    return _line_label(cmd)


# Codex exec scripts are JavaScript. A label is given only when the script provably calls a tool first:
# _JsFirstCall reads an allowlisted prefix and then the call; anything outside that grammar is a reject.
class _JsReject(Exception):
    """The script falls outside the grammar; the argument is a fixed reason (no script text)."""


_JS_LINE_ENDS = '\n\r  '
_JS_SPACE = '\t\v\f \xa0﻿'
_JS_NAME = re.compile(r'[A-Za-z_$][A-Za-z0-9_$]*')
_JS_NUMBER = re.compile(r'0[xX][0-9a-fA-F]+|(?:(?:0|[1-9][0-9]*)(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?')
_JS_TEMPLATE_NAME = re.compile(r'\$\{([A-Za-z_$][A-Za-z0-9_$]*)\}')
_JS_ESCAPES = {'n': '\n', 't': '\t', 'r': '\r', 'b': '\b', 'f': '\f', 'v': '\v'}
_JS_HEX = set('0123456789abcdefABCDEF')
# Names that may reach the global object, rewrite objects or load code. Rejected as a code token in the
# parsed prefix and anywhere in the raw text after the call.
_JS_FORBIDDEN = ('eval', 'Function', 'constructor', 'globalThis', 'global', 'self', 'window', 'this', 'with',
                 'Proxy', 'Reflect', 'defineProperty', 'defineProperties', 'assign', 'setPrototypeOf',
                 '__proto__', 'prototype', 'import', 'require', 'arguments', 'export')
# A binding may not take one of these names: keywords, strict-mode reserved words and names whose
# declaration is an early error or shadows a global the argument could not use.
_JS_RESERVED = set(_JS_FORBIDDEN) | {
    'await', 'break', 'case', 'catch', 'class', 'const', 'continue', 'debugger', 'default', 'delete', 'do',
    'else', 'enum', 'extends', 'false', 'finally', 'for', 'function', 'if', 'in', 'instanceof', 'new', 'null',
    'return', 'super', 'switch', 'throw', 'true', 'try', 'typeof', 'var', 'void', 'while', 'yield', 'let',
    'static', 'implements', 'interface', 'package', 'private', 'protected', 'public', 'undefined', 'NaN',
    'Infinity', 'tools', 'async', 'of', 'get', 'set', 'text', 'Promise'}
# Names that may start the expression holding the call, and names that may start a statement on a new line
# after a declaration without continuing it.
_JS_CALL_STARTS = ('await', 'tools', 'text', 'Promise')
_JS_STATEMENT_STARTS = ('const', 'let', 'var') + _JS_CALL_STARTS


def _js_is_space(ch):
    return ch in _JS_SPACE or unicodedata.category(ch) == 'Zs'


def _js_line_end(src, i):
    return min((j for j in (src.find(t, i) for t in _JS_LINE_ENDS) if j >= 0), default=len(src))


def _js_unescape(body):
    """The value of a string or template body without interpolations, or None for an escape that is a
    syntax error in strict code (legacy octal, \\8, \\9, a malformed \\x or \\u)."""
    out, i, n = [], 0, len(body)
    while i < n:
        ch = body[i]
        i += 1
        if ch != '\\':
            out.append(ch)
            continue
        if i >= n:
            return None
        e = body[i]
        i += 1
        if e in _JS_ESCAPES:
            out.append(_JS_ESCAPES[e])
        elif e == '0' and body[i:i + 1] not in tuple('0123456789'):
            out.append('\0')
        elif e in '0123456789':
            return None
        elif e == 'x':
            h = body[i:i + 2]
            if len(h) != 2 or not set(h) <= _JS_HEX:
                return None
            out.append(chr(int(h, 16)))
            i += 2
        elif e == 'u':
            if body[i:i + 1] == '{':
                j = body.find('}', i)
                h = body[i + 1:j] if j > 0 else ''
                if not h or not set(h) <= _JS_HEX or int(h, 16) > 0x10FFFF:
                    return None
                i = j + 1
            else:
                h = body[i:i + 4]
                if len(h) != 4 or not set(h) <= _JS_HEX:
                    return None
                i += 4
            out.append(chr(int(h, 16)))
        elif e == '\r':
            i += body[i:i + 1] == '\n'
        elif e not in '\n  ':  # a line continuation adds nothing
            out.append(e)
    return ''.join(out)


class _JsDyn(str):
    """A string the argument builds from literals and runtime values, each value written as a $X shell word."""


# A bare name in the argument that no prefix binding holds: a value only the runtime knows.
_JS_RUNTIME = _JsDyn('$X')


class _JsFirstCall:
    """Reads the start of a script with an allowlist grammar, up to the closing ')' of its first call.
    run() gives (tool name, argument value or None) and leaves self.end after the ')', or raises _JsReject.

    Prefix: empty statements and const/let declarations whose initializers are literals (see value). A
    declaration ends at ';', or at a line end when the next token is const, let, var, await or tools, where
    no expression can continue it. Call: an expression statement or the only remaining initializer of a
    const/let/var declaration (a name or a flat [...]/{...} pattern), tools.NAME(ARG?, ?) reached as wrapped()
    describes. Tokens: JavaScript whitespace,
    line ends and comments; ASCII names; the punctuators { } [ ] ( ) , : ; . = +; strings; templates whose
    only interpolations are ${NAME}; decimal and hex numbers. Any other character, a backslash outside a
    literal, '==', '++', '--', '+=' or '-=' is a reject; '=>' is accepted only in a .map callback (see callback)."""

    def __init__(self, src, i):
        self.src, self.i = src, i
        self.bound = {}
        self.declared = set()  # names the call's statement declares for its result
        self.wrappers = set()  # text and Promise when the call is read through them
        self.open = 0  # brackets opened before the call in its expression
        self.runtime = set()  # unbound names the argument reads
        self.exec_arg = False  # the next object is exec_command's argument
        self.dyn_ok = False  # an unbound name may be read: only inside exec_command's top-level cmd string
        self.sync = False  # inside a callback that is not async, where await is a syntax error
        self.mapped = False  # the call is reached through a .map callback
        self.mapped_from = set()  # the prefix binding whose .map the call is reached through
        self.params = set()  # parameter names of the .map callback, which hold its first element
        self.local = set()  # names the callback body declares for the call's result
        self.scope = None  # brackets open where the callback's scope starts: its body block, or .map's '('
        self.repeats = False  # the .map runs the callback, and so the call, once per element of a longer array
        self.advance()

    def advance(self):
        src, i, n, nl = self.src, self.i, len(self.src), False
        while i < n:
            ch = src[i]
            if ch in _JS_LINE_ENDS:
                nl, i = True, i + 1
            elif _js_is_space(ch):
                i += 1
            elif src.startswith('//', i):
                i = _js_line_end(src, i)
            elif src.startswith('/*', i):
                j = src.find('*/', i + 2)
                if j < 0:
                    raise _JsReject('comment never closes')
                nl = nl or any(t in src[i:j] for t in _JS_LINE_ENDS)
                i = j + 2
            else:
                break
        self.nl = nl
        self.tok, self.i = self.token(i)

    def token(self, i):
        src = self.src
        if i >= len(src):
            return ('eof', None), i
        ch = src[i]
        m = _JS_NAME.match(src, i)
        if m:
            nxt = src[m.end():m.end() + 1]
            if nxt == '\\' or (nxt and ord(nxt) > 127 and not _js_is_space(nxt) and nxt not in _JS_LINE_ENDS):
                raise _JsReject('name outside the grammar')
            if m.group() in _JS_FORBIDDEN:
                raise _JsReject('forbidden name')
            return ('name', m.group()), m.end()
        if ch in '"\'':
            j = i + 1
            while j < len(src) and src[j] != ch:
                if src[j] in '\n\r':
                    raise _JsReject('string outside the grammar')
                j += 2 if src[j] == '\\' else 1
            value = _js_unescape(src[i + 1:j]) if j < len(src) else None
            if value is None:
                raise _JsReject('string outside the grammar')
            return ('str', value), j + 1
        if ch == '`':
            parts, j, start = [], i + 1, i + 1
            while j < len(src) and src[j] != '`':
                if src[j] == '\r':
                    raise _JsReject('template outside the grammar')
                if src.startswith('${', j):
                    m = _JS_TEMPLATE_NAME.match(src, j)
                    if not m:
                        raise _JsReject('template outside the grammar')
                    parts += [_js_unescape(src[start:j]), ('name', m.group(1))]
                    j = start = m.end()
                    continue
                j += 2 if src[j] == '\\' else 1
            if j >= len(src):
                raise _JsReject('template outside the grammar')
            parts.append(_js_unescape(src[start:j]))
            if None in parts or any(isinstance(p, tuple) and p[1] in _JS_FORBIDDEN for p in parts):
                raise _JsReject('template outside the grammar')
            return ('tpl', parts), j + 1
        m = _JS_NUMBER.match(src, i)
        if m:
            if re.match(r'[\w$.\\]', src[m.end():m.end() + 1]) or ord(src[m.end():m.end() + 1] or ' ') > 127:
                raise _JsReject('number outside the grammar')
            t = m.group()
            v = int(t, 16) if t[:2] in ('0x', '0X') else float(t) if '.' in t or 'e' in t.lower() else int(t)
            if abs(v) > 2 ** 53 - 1:  # a JS number would round it, so the argument would differ
                raise _JsReject('number outside the grammar')
            return ('num', v), m.end()
        if src.startswith('=>', i):
            return ('p', '=>'), i + 2
        if ch in '{}[](),:;.=+-':
            if src[i:i + 2] in ('==', '++', '+=', '--', '-='):
                raise _JsReject('operator outside the grammar')
            return ('p', ch), i + 1
        raise _JsReject('character outside the grammar')

    def expect(self, kind, value, reason):
        if self.tok != (kind, value):
            raise _JsReject(reason)
        self.advance()

    def run(self):
        while True:
            kind, v = self.tok
            if self.tok == ('p', ';'):
                self.advance()
            elif kind == 'name' and v in ('const', 'let', 'var'):
                self.advance()
                while True:
                    if self.tok in (('p', '['), ('p', '{')):
                        self.declared |= self.pattern()
                        self.expect('p', '=', 'declaration outside the grammar')
                        return self.wrapped()
                    name = self.binding_name()
                    self.expect('p', '=', 'declaration outside the grammar')
                    if self.tok[0] == 'name' and self.tok[1] in _JS_CALL_STARTS or self.tok in (('p', '('), ('p', '[')) and self.leads_to_call():
                        self.declared.add(name)
                        return self.wrapped()
                    if v == 'var':
                        raise _JsReject('var before the call')
                    self.bound[name] = self.value(init=True)
                    if self.tok != ('p', ','):
                        break
                    self.advance()
                if self.tok == ('p', ';'):
                    self.advance()
                elif not (self.nl and self.tok[0] == 'name' and self.tok[1] in _JS_STATEMENT_STARTS):
                    raise _JsReject('no call' if self.tok[0] == 'eof' else 'statement end outside the grammar')
            elif kind == 'name' and v in _JS_CALL_STARTS or self.tok in (('p', '('), ('p', '[')):
                return self.wrapped()
            else:
                raise _JsReject('no call' if kind == 'eof' else 'statement before the call')

    def leads_to_call(self):
        """Whether the '(' or '[' at the current token opens a chain of '(' and '[' that reaches a call start."""
        n = _JS_NAME.match(self.src, re.compile(r'[\s(\[]*').match(self.src, self.i).end())
        return bool(n) and n.group() in _JS_CALL_STARTS

    def pattern(self):
        """An array or object binding pattern of plain names for the call result: [a, b] or {a, b}."""
        close = ']' if self.tok == ('p', '[') else '}'
        self.advance()
        seen = set()
        while self.tok != ('p', close):
            if close == ']' and self.tok == ('p', ','):
                self.advance()
                continue
            name = self.binding_name()
            if close == '}' and self.tok == ('p', ':'):
                self.advance()
                name = self.binding_name()
            if name in seen:
                raise _JsReject('declaration outside the grammar')
            seen.add(name)
            if self.tok == ('p', ','):
                self.advance()
            elif self.tok != ('p', close):
                raise _JsReject('declaration outside the grammar')
        self.advance()
        return seen

    def wrapped(self):
        """The first call of an expression that evaluates nothing before it: tools.NAME(...) reached through
        await, a '(' group, an array or object literal whose earlier entries are literals or reads of bound
        names, the argument of text(...), the argument or array argument of Promise.all/allSettled/race/any,
        or the callback of NAME.map(...) over a bound non-empty literal array. Reading stops at the call's ')'."""
        while True:
            if self.tok == ('name', 'await'):
                if self.sync:
                    raise _JsReject('call outside the grammar')
                self.advance()
            elif self.tok == ('p', '('):
                self.open += 1
                self.advance()
            elif self.tok in (('p', '['), ('p', '{')):
                self.open += 1
                self.entries(']' if self.tok == ('p', '[') else '}')
            elif self.tok == ('name', 'text'):
                self.wrappers.add('text')
                self.advance()
                self.expect('p', '(', 'call outside the grammar')
                self.open += 1
            elif self.tok == ('name', 'Promise'):
                self.wrappers.add('Promise')
                self.advance()
                self.expect('p', '.', 'call outside the grammar')
                if self.tok not in (('name', 'all'), ('name', 'allSettled'), ('name', 'race'), ('name', 'any')):
                    raise _JsReject('call outside the grammar')
                self.advance()
                self.expect('p', '(', 'call outside the grammar')
                self.open += 1
            elif self.tok[0] == 'name' and isinstance(self.bound.get(self.tok[1]), list) and self.bound[self.tok[1]] \
                    and not self.mapped:
                first = self.bound[self.tok[1]][0]
                self.mapped_from.add(self.tok[1])
                self.repeats = len(self.bound[self.tok[1]]) > 1
                self.advance()
                self.expect('p', '.', 'call outside the grammar')
                self.expect('name', 'map', 'call outside the grammar')
                self.expect('p', '(', 'call outside the grammar')
                self.open += 1
                self.mapped = True
                return self.callback(first)
            else:
                return self.call()

    def entries(self, close):
        """Skips the '[' or '{' at the current token and the entries before the one that holds the call. Each
        skipped entry is a string, a number or a read of a bound name (an object entry: KEY: that value, or a
        shorthand bound NAME), so evaluating it runs nothing. The entry that holds the call follows."""
        self.advance()
        while True:
            if close == '}':
                kind, key = self.tok
                if kind not in ('name', 'str') or key in ('__proto__', 'tools'):
                    raise _JsReject('object outside the grammar')
                self.advance()
                if self.tok == ('p', ',') and kind == 'name' and key in self.bound:
                    self.advance()
                    continue
                self.expect('p', ':', 'object outside the grammar')
            kind, v = self.tok
            if not (kind in ('str', 'num') or kind == 'name' and v in self.bound):
                return
            j = self.i
            while j < len(self.src) and (self.src[j] in _JS_LINE_ENDS or _js_is_space(self.src[j])):
                j += 1
            if self.src[j:j + 1] != ',':
                return  # the entry continues (v.x, v + x, v(...)): it is the one that must hold the call
            self.advance()
            self.advance()

    def callback(self, first):
        """Reads the arrow function passed to .map up to the start of its body and binds its parameters to the
        first element, the value of its first invocation. Parameters: a name, or in parentheses a name or a flat
        [a, b] / {a, k: b} pattern of plain names, then at most an index name. The body is an expression that
        holds the call, or a block whose first statement does: a declaration, a return or an expression."""
        if self.tok == ('name', 'async'):
            self.advance()
            if self.nl:
                raise _JsReject('call outside the grammar')
        else:
            self.sync = True
        names = []
        if self.tok == ('p', '('):
            self.advance()
            if self.tok != ('p', ')'):
                if self.tok in (('p', '['), ('p', '{')):
                    names += self.element_pattern(first)
                else:
                    names.append((self.binding_name(), first))
                if self.tok == ('p', ','):
                    self.advance()
                    if self.tok != ('p', ')'):
                        names.append((self.binding_name(), 0))
            self.expect('p', ')', 'call outside the grammar')
        else:
            names.append((self.binding_name(), first))
        if len({n for n, _ in names}) != len(names) or self.nl or self.tok != ('p', '=>'):
            raise _JsReject('call outside the grammar')
        self.advance()
        self.bound.update(names)
        self.params.update(n for n, _ in names)
        if self.tok != ('p', '{'):
            self.scope = self.open
            return self.wrapped()
        self.open += 1
        self.scope = self.open
        self.advance()
        while self.tok == ('p', ';'):
            self.advance()
        kind, v = self.tok
        if kind == 'name' and v in ('const', 'let', 'var'):
            self.advance()
            if self.tok in (('p', '['), ('p', '{')):
                self.local |= self.pattern()
            else:
                self.local.add(self.binding_name())
            self.expect('p', '=', 'declaration outside the grammar')
        elif self.tok == ('name', 'return'):
            self.advance()
            if self.nl:
                raise _JsReject('call outside the grammar')
        elif not (kind == 'name' and v in _JS_CALL_STARTS or self.tok in (('p', '('), ('p', '['))):
            raise _JsReject('statement before the call')
        return self.wrapped()

    def element_pattern(self, value):
        """[(name, value)] for a flat [a, , b] or {a, k: b} parameter pattern destructuring value. An array
        pattern needs an array long enough, an object pattern an object with each key, so no name is undefined."""
        close = ']' if self.tok == ('p', '[') else '}'
        if not isinstance(value, list if close == ']' else dict):
            raise _JsReject('call outside the grammar')
        self.advance()
        out, k = [], 0
        while self.tok != ('p', close):
            if close == ']' and self.tok == ('p', ','):
                k += 1
                self.advance()
                continue
            name = key = self.binding_name()
            if close == '}' and self.tok == ('p', ':'):
                self.advance()
                name = self.binding_name()
            if close == ']':
                if k >= len(value):
                    raise _JsReject('call outside the grammar')
                out.append((name, value[k]))
                k += 1
            else:
                if key not in value:
                    raise _JsReject('call outside the grammar')
                out.append((name, value[key]))
            if self.tok == ('p', ','):
                self.advance()
                if close == ']':
                    continue
            elif self.tok != ('p', close):
                raise _JsReject('call outside the grammar')
        self.advance()
        return out

    def binding_name(self):
        kind, name = self.tok
        if kind != 'name' or name in _JS_RESERVED or name in self.bound:
            raise _JsReject('declaration outside the grammar')
        self.advance()
        return name

    def call(self):
        self.expect('name', 'tools', 'call outside the grammar')
        self.expect('p', '.', 'call outside the grammar')
        kind, name = self.tok
        if kind != 'name':
            raise _JsReject('call outside the grammar')
        self.advance()
        if self.tok != ('p', '('):
            raise _JsReject('call outside the grammar')
        self.advance()
        arg = None
        if self.tok != ('p', ')'):
            self.exec_arg = name == 'exec_command'
            arg = self.value()
            if self.tok == ('p', ','):
                self.advance()
        if self.tok != ('p', ')'):
            raise _JsReject('argument outside the grammar')
        self.end = self.i  # the ')' is the current token; nothing after it is read
        return name, arg

    def value(self, init=False):
        """A literal: a string, a template, a number, true, false, null, or an object or array of those.
        Strings and templates may be joined by '+'. In the call argument, a name, a ${name} and a shorthand
        key read a prefix binding; a name joined by '+' or in a template must hold a string."""
        v = self.primary(init)
        if self.tok != ('p', '+'):
            return v
        parts = [v]
        while self.tok == ('p', '+'):
            self.advance()
            parts.append(self.primary(init))
        if not all(isinstance(p, str) for p in parts) or isinstance(parts[0], _JsDyn) and len(parts[0]) == 2:
            raise _JsReject('concatenation outside the grammar')  # a runtime value first may be a number
        return self.join(parts)

    @staticmethod
    def join(parts):
        text = ''.join(parts)
        return _JsDyn(text) if any(isinstance(p, _JsDyn) for p in parts) else text

    def primary(self, init):
        kind, v = self.tok
        if kind in ('str', 'num'):
            self.advance()
            return v
        if self.tok == ('p', '-'):
            self.advance()
            if self.tok[0] != 'num':
                raise _JsReject('initializer outside the grammar' if init else 'argument outside the grammar')
            v = -self.tok[1]
            self.advance()
            return v
        if kind == 'tpl':
            self.advance()
            parts = [p if isinstance(p, str) else self.read(p[1], init) for p in v]
            if not all(isinstance(p, str) for p in parts):
                raise _JsReject('template outside the grammar')
            return self.join(parts)
        if kind == 'name':
            self.advance()
            return {'true': True, 'false': False, 'null': None}[v] if v in ('true', 'false', 'null') else self.read(v, init)
        if self.tok == ('p', '{'):
            return self.object(init)
        if self.tok == ('p', '['):
            return self.array(init)
        raise _JsReject('initializer outside the grammar' if init else 'argument outside the grammar')

    def read(self, name, init):
        if name in self.bound:
            return self.bound[name]
        if init or not self.dyn_ok or name in _JS_RESERVED or name in self.declared | self.local:
            raise _JsReject('unbound name in an initializer' if init else 'unbound name in the argument')
        self.runtime.add(name)
        return _JS_RUNTIME

    def object(self, init):
        # An unbound name throws a ReferenceError before the call runs, wherever the argument reads it; only
        # exec_command's top-level cmd string may hold one, because the cmd label then names the call.
        # A repeated key keeps the last value but still evaluates the first, so it is a reject.
        self.advance()
        exec_arg, self.exec_arg, dyn_ok = self.exec_arg, False, self.dyn_ok
        obj = {}
        while self.tok != ('p', '}'):
            kind, key = self.tok
            if kind not in ('name', 'str') or key in ('__proto__', 'tools') or key in obj:
                raise _JsReject('object outside the grammar')
            self.advance()
            self.dyn_ok = exec_arg and key == 'cmd'
            if self.tok == ('p', ':'):
                self.advance()
                obj[key] = self.value(init)
            elif kind == 'name' and self.tok in (('p', ','), ('p', '}')):
                obj[key] = self.read(key, init)
            else:
                raise _JsReject('object outside the grammar')
            self.dyn_ok = dyn_ok
            if self.tok == ('p', ','):
                self.advance()
            elif self.tok != ('p', '}'):
                raise _JsReject('object outside the grammar')
        self.advance()
        return obj

    def array(self, init):
        self.advance()
        arr = []
        while self.tok != ('p', ']'):
            arr.append(self.value(init))
            if self.tok == ('p', ','):
                self.advance()
            elif self.tok != ('p', ']'):
                raise _JsReject('array outside the grammar')
        self.advance()
        return arr


# Labels for Codex exec scripts that run no resolvable shell command, by the first tools.* call.
EXEC_TOOL_LABELS = {'write_stdin': 'write_stdin', 'apply_patch': 'apply_patch', 'view_image': 'view_image',
                    'update_plan': 'update_plan'}
EXEC_TOOL_PREFIXES = {'web__': 'web', 'clock__': 'clock'}
MCP_SERVERS = {'t3_code': 'MCP: t3_code'}
# codex_apps serves connected apps; the app is the prefix of the function name.
CODEX_APPS = ('slack', 'atlassian', 'gmail', 'github', 'fathom', 'adobe_workfront', 'google_calendar',
              'google_drive', 'google_docs', 'google_sheets', 'google_slides', 'figma')
MCP_OTHER = 'MCP (other)'
TOOL_DISCOVERY = 'Tool discovery'


def _exec_tool_label(name):
    """Label for a tools.<name>(...) call other than exec_command, or None for an unknown built-in."""
    if name.startswith('mcp__'):
        server, sep, fn = name[5:].partition('__')
        if sep and server == 'codex_apps':
            app = next((a for a in CODEX_APPS if fn == a or fn.startswith(a + '_')), None)
            return 'MCP: ' + app if app else MCP_OTHER
        return MCP_SERVERS.get(server, MCP_OTHER) if sep else MCP_OTHER
    if name in EXEC_TOOL_LABELS:
        return EXEC_TOOL_LABELS[name]
    return next((v for k, v in EXEC_TOOL_PREFIXES.items() if name.startswith(k)), None)



# After the call, the raw text (strings and comments included) may not name tools other than as a
# tools.<name>( call, hold a \u escape, a forbidden name or a bare '#', or name a prefix binding other than
# after a '.'. Only hoisting can make later code run first or change tools, and these keep any declaration
# of tools out; the rest of rule c is checked on the raw text because nothing past the call is tokenized.
_JS_LATER_TOOLS_RE = re.compile(r'(?<![\w$])tools(?![\w$])(?!\s*\.\s*[A-Za-z_$][\w$]*\s*\()')
_JS_LATER_FORBIDDEN_RE = re.compile(r'(?<![\w$])(?:' + '|'.join(_JS_FORBIDDEN) + r')(?![\w$])')
# A script that may reach a tool: it names one, or may build a reference to tools without naming it.
_JS_TOOLISH_RE = re.compile(r'(?<![\w$])tools(?![\w$])|exec_command|\\u|' + _JS_LATER_FORBIDDEN_RE.pattern)


def _exec_hashbang_end(script):
    """Offset after a '#!' hashbang at offset 0, which is a comment up to the first line end, else 0."""
    return _js_line_end(script, 0) if script.startswith('#!') else 0


# Declaring keywords. A later declaration of a name the call reads or declares can stop the call: a duplicate
# of a prefix or result binding is an early error, so the script never runs, and a later let, const or class of
# text or Promise (or a var or function of Promise) makes reading it throw before the call.
_JS_DECLARING_KEYWORDS = ('let', 'const', 'var', 'class', 'function', 'using')


def _js_rest_declares(rest, names, open=0):
    """Whether code after the call may declare one of names. True for a name right after a declaring keyword
    anywhere, after 'var' anywhere in the code, as an element of a [...] or {...} pattern whose outermost
    bracket follows let, const, var or using, or after a ',' outside any bracket (let a, NAME). A name read
    elsewhere is a use. open is the number of brackets the call's expression opened before the call; code
    inside them is that expression, not a declaration. Code the scan cannot follow (a slash that is not a comment, an unclosed string) gives
    True when one of names appears in it at all."""
    if not names:
        return False
    rest = rest.replace('\ufeff', ' ')  # JS whitespace that str.isspace and \\s do not cover
    alt = '|'.join(re.escape(n) for n in sorted(names))
    if re.search(r'(?<![\w$])(?:' + '|'.join(_JS_DECLARING_KEYWORDS) + r')\s+(?:' + alt + r')(?![\w$])', rest):
        return True
    if not re.search(r'(?<![\w$])(?:' + alt + r')(?![\w$])', rest):
        return False
    if re.search(r'(?<![\w$.])var(?![\w$])', rest):
        return True
    stack, prev, i, n = [['x', None]] * open, None, 0, len(rest)  # stack: [open bracket, token before it]
    while i < n:
        ch = rest[i]
        if stack and stack[-1][0] == '`':
            if ch == '\\':
                i += 2
            elif ch == '`':
                stack.pop()
                prev, i = 'str', i + 1
            elif rest.startswith('${', i):
                stack.append(['${', prev])
                prev, i = '${', i + 2
            else:
                i += 1
        elif ch.isspace():
            i += 1
        elif rest.startswith('//', i):
            i = _js_line_end(rest, i)
        elif rest.startswith('/*', i):
            j = rest.find('*/', i + 2)
            if j < 0:
                return True
            i = j + 2
        elif ch == '/':
            return True
        elif ch in '"\'':
            j = i + 1
            while j < n and rest[j] != ch:
                if rest[j] in _JS_LINE_ENDS[:2]:
                    return True
                j += 2 if rest[j] == '\\' else 1
            if j >= n:
                return True
            prev, i = 'str', j + 1
        elif ch == '`':
            stack.append(['`', prev])
            i += 1
        elif ch in '([{':
            stack.append([ch, prev])
            prev, i = ch, i + 1
        elif ch in ')]}':
            if not stack:
                return True
            stack.pop()
            prev, i = ch, i + 1
        elif rest.startswith('...', i):
            prev, i = '...', i + 3
        elif ch == '*' and prev == 'function':
            i += 1
        else:
            m = _JS_NAME.match(rest, i)
            if not m:
                prev, i = ch, i + 1
                continue
            word = m.group()
            if word in names and prev != '.':
                if prev in _JS_DECLARING_KEYWORDS or prev == ',' and not stack:
                    return True
                if prev in (',', '[', '{', ':', '...'):
                    k = len(stack)
                    while k and stack[k - 1][0] in '[{':
                        k -= 1
                    if k < len(stack) and stack[k][1] in _JS_DECLARING_KEYWORDS:
                        return True
            prev, i = word, m.end()
    return bool(stack)


def _exec_first_call(script):
    """(tool name, argument, code after the call, whether the call repeats) of the call a Codex exec script provably
    makes before running anything else, or a raise of _JsReject. A call in a .map callback repeats once per further
    element. A '#!' hashbang at offset 0 is a comment up to the first line end; after a
    byte-order mark it is not, and the '#' is outside the grammar."""
    p = _JsFirstCall(script, _exec_hashbang_end(script))
    name, arg = p.run()
    rest = script[p.end:]
    if _JS_LATER_TOOLS_RE.search(rest):
        raise _JsReject('later code names tools other than in a call')
    if '\\u' in rest:
        raise _JsReject('later \\u escape')
    if _JS_LATER_FORBIDDEN_RE.search(rest):
        raise _JsReject('later forbidden name')
    if '#' in rest:
        raise _JsReject("later '#'")
    if any(re.search(r'(?<![\w$.])' + re.escape(b) + r'(?![\w$])', rest) for b, v in p.bound.items()
           if isinstance(v, (dict, list)) and (b not in p.mapped_from or _js_shares(arg, v))):
        raise _JsReject('later use of a prefix binding')
    scoped = p.params | p.local
    if _js_rest_declares(rest, set(p.bound) - p.params | p.declared | p.wrappers | p.runtime, p.open):
        raise _JsReject('later declaration of a name the call reads or declares')
    if scoped:  # the callback's names: only a declaration in its own scope can rebind them or be a duplicate
        end = _js_scope_end(rest, p.open - p.scope + 1)
        if _js_rest_declares(rest if end is None else rest[:end], scoped, p.open - p.scope):
            raise _JsReject('later declaration of a name the call reads or declares')
    return name, arg, rest, p.repeats


def _js_shares(arg, v):
    """Whether arg holds v or an object or array inside v, so mutating v later could change the argument."""
    inner, todo = set(), [v]
    while todo:
        x = todo.pop()
        if isinstance(x, (dict, list)):
            inner.add(id(x))
            todo += x.values() if isinstance(x, dict) else x
    todo = [arg]
    while todo:
        x = todo.pop()
        if isinstance(x, (dict, list)):
            if id(x) in inner:
                return True
            todo += x.values() if isinstance(x, dict) else x
    return False


def _js_scope_end(rest, closes):
    """Offset in rest of the bracket that closes the closes-th bracket still open at its start, skipping strings
    and comments; None when the code cannot be followed (a slash that is not a comment, a template with ${...},
    an unclosed literal) or the bracket never comes."""
    depth, i, n = 0, 0, len(rest)
    while i < n:
        ch = rest[i]
        if rest.startswith('//', i):
            i = _js_line_end(rest, i)
            continue
        if rest.startswith('/*', i):
            j = rest.find('*/', i + 2)
            if j < 0:
                return None
            i = j + 2
            continue
        if ch == '/':
            return None
        if ch in '"\'`':
            j = i + 1
            while j < n and rest[j] != ch:
                if ch != '`' and rest[j] in '\n\r' or ch == '`' and rest.startswith('${', j):
                    return None
                j += 2 if rest[j] == '\\' else 1
            if j >= n:
                return None
            i = j + 1
            continue
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
            if depth == -closes:
                return i
        i += 1
    return None


def _js_rest_closed(rest):
    """Whether code after the call provably ends outside any string, template or comment. A slash that does
    not open a comment may start a regex literal, so it gives False, as does a raw newline in a quote string."""
    stack, i, n = [], 0, len(rest)  # stack holds '`' for an open template and '{' for a brace inside ${...}
    while i < n:
        ch = rest[i]
        if stack and stack[-1] == '`':
            if ch == '\\':
                i += 2
            elif ch == '`':
                stack.pop()
                i += 1
            elif rest.startswith('${', i):
                stack.append('{')
                i += 2
            else:
                i += 1
        elif rest.startswith('//', i):
            i = _js_line_end(rest, i)
        elif rest.startswith('/*', i):
            j = rest.find('*/', i + 2)
            if j < 0:
                return False
            i = j + 2
        elif ch == '/':
            return False
        elif ch in '"\'':
            j = i + 1
            while j < n and rest[j] != ch:
                if rest[j] in _JS_LINE_ENDS[:2]:
                    return False
                j += 2 if rest[j] == '\\' else 1
            if j >= n:
                return False
            i = j + 1
        elif ch == '`':
            stack.append('`')
            i += 1
        else:
            if ch == '{' and stack:
                stack.append('{')
            elif ch == '}' and stack:
                stack.pop()
            i += 1
    return not stack


def _exec_script_verdict(script):
    """(label, reason) for a Codex exec script; reason is a fixed string when the label is Unparsed."""
    try:
        name, arg, rest, _ = _exec_first_call(script)
    except _JsReject as e:
        if _JS_TOOLISH_RE.search(script, _exec_hashbang_end(script)):
            return CMD_UNPARSED, e.args[0]
        return (TOOL_DISCOVERY if re.search(r'(?<![\w$"\'`])ALL_TOOLS(?![\w$"\'`])', script) else None), None
    if name == 'exec_command':
        cmd = arg.get('cmd') if isinstance(arg, dict) else None
        if isinstance(cmd, list) and cmd and all(type(x) is str for x in cmd):
            return command_label(cmd), None
        if isinstance(cmd, _JsDyn):  # a runtime value can hold any shell text, even an unmatched quote
            if cmd != 'git status $X':  # the one approved form: "git status " + value
                return CMD_UNPARSED, 'cmd holds a runtime value'
            return 'git status', None
        if isinstance(cmd, str) and cmd.strip():
            return command_label(cmd), None
        return CMD_UNPARSED, 'cmd not a static string or argv'
    if 'exec_command' in script[_exec_hashbang_end(script):]:
        return CMD_UNPARSED, 'exec_command after another tool call'
    label = _exec_tool_label(name)
    if not label and _JS_TOOLISH_RE.search(rest):
        return CMD_UNPARSED, 'unknown tool before another call'
    return label, None


def _exec_script_label(script):
    """Label of a Codex exec script, or None when it calls no tool and does not list tools. The label comes
    from the call the script provably makes first (_exec_first_call): its command when it is exec_command
    with a static cmd, or the tool's label when the script never names exec_command. A script that may still
    reach a tool gives Unparsed, one that reads ALL_TOOLS gives Tool discovery."""
    return _exec_script_verdict(script)[0]


def tool_command(tool, args):
    """Command label for one built-in shell tool call. args is the tool input: a dict (Claude),
    a JSON string (Codex function calls) or the raw script (Codex exec)."""
    if tool == 'exec' and isinstance(args, str):
        # heuristic: the first call outside comments and strings, even in a branch that never runs
        label = _exec_script_label(args)
        if label:
            return label
        try:
            json.loads(args)
        except ValueError:
            return CMD_NO_SHELL
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return CMD_UNPARSED
    if not isinstance(args, dict):
        return CMD_UNPARSED
    for k in ('command', 'cmd', 'script'):
        v = args.get(k)
        if isinstance(v, (str, list)) and v:
            return command_label(v)
    return CMD_UNPARSED


def cmd_label_for(server, tool, args):
    """Event suffix carrying the command label for shell tools, () for anything else."""
    return (tool_command(tool, args),) if server is None and tool in SHELL_TOOLS else ()


# How an instruction file entered context: the leaf under each file node.
HOW_LAUNCH = 'Autoloaded at session start'
HOW_COMPACT = 'Reloaded after compaction'
HOW_ACCESS = 'Loaded on path match'
HOW_IMPORT = '@-imported'
READ_PROGRAMS = {'cat', 'head', 'tail', 'sed', 'nl', 'less', 'more', 'bat'}
READ_OPT_WITH_ARG = {'head': {'-n', '-c'}, 'tail': {'-n', '-c'}, 'sed': {'-e'}}
READ_OPTS = {'cat': {'-n', '-b', '-s', '-v', '-e', '-t', '-A', '-E', '-T', '-u'}, 'head': {'-q', '-v'},
             'tail': {'-q', '-v'}, 'sed': {'-n', '-E', '-r', '-s', '--posix'},
             'nl': {'-ba', '-bt', '-bn'}, 'less': {'-N', '-R', '-S'}, 'more': set(),
             'bat': {'-p', '-pp', '--plain', '-n', '--number', '-P', '--no-paging'}}
EXPLICIT_INSTR = 'Instruction files'
EXPLICIT_OTHER = 'Other files'
SEVERAL_FILES = 'Several instruction files'


def is_instruction_path(p):
    """Whether a file path names an instruction file Claude Code or Codex can load by itself."""
    p = (p or '').replace('\\', '/')
    b = os.path.basename(p)
    if b in ('CLAUDE.md', 'CLAUDE.local.md', 'AGENTS.md'):
        return True
    if p.endswith('.md') and ('/.claude/rules/' in p or '/.agents/rules/' in p
                              or re.search(r'/\.claude/projects/[^/]+/memory/', p)):
        return True
    return '/.codex/memories/' in p


def _resolve(p, cwd, home=HOME):
    """Absolute, symlink-resolved path for p (relative to cwd); None when it can't be placed."""
    if p.startswith('~/') or p == '~':
        p = home + p[1:]
    elif not os.path.isabs(p):
        if not isinstance(cwd, str) or not os.path.isabs(cwd):
            return None
        p = os.path.join(cwd, p)
    p = os.path.normpath(p)
    return os.path.realpath(p) if os.path.exists(p) else p


def _raw_command(tool, args):
    """(command line, working directory) of one shell tool call, or None. A Codex exec script counts
    only when its proven first call is exec_command, runs once, no exec_command follows it, the code after it provably
    ends outside a literal or comment, and the call has a literal cmd and, if any, a literal workdir."""
    if tool == 'exec' and isinstance(args, str):
        try:
            name, arg, rest, repeats = _exec_first_call(args)
        except _JsReject:
            return None
        if (name != 'exec_command' or repeats or 'exec_command' in rest or not _js_rest_closed(rest)
                or not isinstance(arg, dict)):
            return None
        cmd, wd = arg.get('cmd'), arg.get('workdir', '')
        if type(cmd) is not str or type(wd) is not str:
            return None  # a runtime value leaves the command or its directory unknown
        return cmd, wd or None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if not isinstance(args, dict):
        return None
    if any(args.get(k) is not None and not isinstance(args[k], str) for k in ('workdir', 'cwd')):
        return None  # a malformed working directory leaves relative operands unplaced
    wd = args.get('workdir') or args.get('cwd')
    for k in ('command', 'cmd', 'script'):
        v = args.get(k)
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            v = v[2] if len(v) >= 3 and v[0].rsplit('/', 1)[-1] in SHELLS and v[1] in ('-c', '-lc') else shlex.join(v)
        if isinstance(v, str) and v:
            return v, wd or None
    return None


SED_READ_SCRIPT_RE = re.compile(r'\s*(?:(?:\d+|\$)(?:,(?:\d+|\$))?\s*!?\s*)?[pdqn=]?\s*(?:;\s*(?:(?:\d+|\$)(?:,(?:\d+|\$))?\s*!?\s*)?[pdqn=]?\s*)*')


def _sed_script_ok(script):
    """Whether a sed script only prints: line-number addresses with p, d, q, n or = commands. Any
    other script (regex addresses, s///, w, e, ...) is rejected, harmless or not."""
    return bool(SED_READ_SCRIPT_RE.fullmatch(script))


def _read_operands(seg, cwd, home):
    """File operands of one read-program segment, or None when the segment is not a plain read: an
    option outside the program's read-only allowlist (sed -i, tail -f, ...), a write redirect, a sed
    script that writes or executes, or stdin that is not exactly one '<' file and the only input."""
    if not seg or seg[0] not in READ_PROGRAMS:
        return None
    prog, files, stdin, j = seg[0], [], None, 1
    need_script = prog == 'sed'
    opts = READ_OPTS.get(prog, set())
    while j < len(seg):
        w = seg[j]
        j += 1
        if w in REDIRECTS or (w and w[0] in '<>' and not w.strip('<>&|')):
            fd = files[-1] if files and files[-1].isdigit() and seg[j - 2] == files[-1] else None
            if fd:
                files.pop()  # the file descriptor of "2>file"
            target = seg[j] if j < len(seg) else None
            j += 1
            if w[0] == '<':
                # "n<", "<&", "<>" and a second "<" redirect descriptors; which input wins is not tracked
                if w != '<' or fd or stdin or not target:
                    return None
                stdin = target
            elif not (target == '/dev/null' or w == '>&' and target and (target.isdigit() or target == '-')):
                return None
            continue
        if w == '--':
            files.extend(seg[j:])
            break
        if w.startswith('-') and len(w) > 1:
            if w in READ_OPT_WITH_ARG.get(prog, ()):
                if j >= len(seg):
                    return None
                if prog == 'sed':
                    if not _sed_script_ok(seg[j]):
                        return None
                    need_script = False
                j += 1
            elif not (w in opts or prog in ('head', 'tail') and re.fullmatch(r'-[nc]?\d+', w)
                      or prog == 'bat' and re.match(r'--(style|paging|language|line-range)=', w)):
                return None
            continue
        if need_script:
            if not _sed_script_ok(w):
                return None
            need_script = False  # sed without -e: the first operand is the script
            continue
        files.append(w)
    if stdin:
        return None if files else [stdin]
    return files


GLOB_RE = re.compile(r'[*?\[]|\{[^{}]*,[^{}]*\}')


def explicit_read(cmd, cwd, home=HOME):
    """File label when a shell command line only reads instruction files: one read program (cat, head,
    sed, ...) with read-only options on instruction-file operands, optionally preceded by cd / env
    assignments and piped through read programs without file operands. Several files give
    SEVERAL_FILES; anything else (search tools, '&&' chains, writes, other operands) None."""
    words = _shell_words(cmd)
    if not words:
        return None
    segs, ops, seg = [], [], []
    for w in words:
        if w in SEPARATORS:
            if seg:
                segs.append(seg)
                ops.append(w)
            seg = []
        else:
            seg.append(w)
    if seg:
        segs.append(seg)
        ops.append(None)
    i = 0
    while i < len(segs) and segs[i][0] == 'cd' and ops[i] in ('&&', ';'):
        if len(segs[i]) > 2:
            return None
        cwd = _resolve(segs[i][1], cwd, home) if len(segs[i]) == 2 else home
        if cwd is None:
            return None
        i += 1
    if i >= len(segs) or any(o not in ('|', None) for o in ops[i:]):
        return None
    seg = segs[i]
    while seg and ENV_ASSIGN_RE.match(seg[0]):
        seg = seg[1:]
    files = _read_operands(seg, cwd, home)
    if not files:
        return None
    if any(_read_operands(s, cwd, home) != [] or any(w[:1] == '<' for w in s) for s in segs[i + 1:]):
        return None  # a later stage must only filter its input
    labels = set()
    for f in files:
        r = _resolve(f, cwd, home)
        if r is None or not is_instruction_path(r):
            return None
        # a glob or brace operand may expand to several files; its text is not a file label
        labels.add(SEVERAL_FILES if GLOB_RE.search(f) else file_label(r, home))
    return labels.pop() if len(labels) == 1 else SEVERAL_FILES


def result_label_for(server, tool, args, cwd):
    """Event suffix of a built-in tool's result. Shell results carry the command label, extended with
    ('Instruction files', file label) when the command only read instruction files; Read results are
    labelled ('Instruction files', file label) or ('Other files',)."""
    if server is not None:
        return ()
    if tool == 'Read':
        fp = args.get('file_path') if isinstance(args, dict) else None
        r = _resolve(fp, cwd) if isinstance(fp, str) and fp else None
        return ((EXPLICIT_INSTR, file_label(r)),) if r and is_instruction_path(r) else ((EXPLICIT_OTHER,),)
    if tool not in SHELL_TOOLS:
        return ()
    label = tool_command(tool, args)
    raw = _raw_command(tool, args)
    f = explicit_read(raw[0], raw[1] or cwd) if raw else None
    return ((label, EXPLICIT_INSTR, f),) if f else (label,)


def model_provider(model):
    """Model class for the dashboard's Claude / GPT buttons."""
    m = (model or '').lower()
    if re.match(r'(claude|fable|opus|sonnet|haiku)', m):
        return 'anthropic'
    if re.match(r'(gpt|o\d|codex|chatgpt)', m):
        return 'openai'
    return 'other'


def is_cmd_group(path):
    """Tool call / result nodes whose children are commands (events carry a command label)."""
    return len(path) == 3 and path[0] in (CALLS, RESULTS) and path[1] == 'Built-in tools' and path[2] in SHELL_TOOLS


def instr_file_label(p):
    p = (p or '').replace('\\', '/')
    b = os.path.basename(p)
    if '/.codex/memories/' in p:
        return 'Codex memory summary (imported)'
    if re.search(r'(^|/)\.claude/CLAUDE\.md$', p) or p == '~/.claude/CLAUDE.md':
        return 'User CLAUDE.md'
    if '/memory/' in p:
        return 'Project memory (MEMORY.md)' if b == 'MEMORY.md' else 'Project memory files'
    if b in ('CLAUDE.md', 'CLAUDE.local.md', 'AGENTS.md'):
        return 'Project ' + b
    if '/.claude/rules/' in p or '/rules/' in p:
        return 'Rules files'
    return 'Other instruction files'


def is_file_group(path):
    """Nodes whose children are individual instruction files (events carry a file label)."""
    return (len(path) == 3 and path[:2] == (INSTR, 'Instruction files')) or path == (INSTR, 'AGENTS.md instructions')


def is_explicit_group(path):
    """Tool-result nodes whose children are the instruction files a tool read explicitly."""
    return len(path) > 3 and path[0] == RESULTS and path[-1] == 'Instruction files'


def _gitdir_repo(gitdir):
    """Repository name from a worktree's gitdir: '<repo>/.git/worktrees/<w>', '<repo>.git/worktrees/<w>'
    or '<repo>/worktrees/<w>' (bare layout)."""
    g = re.sub(r'/worktrees/[^/]+/?$', '', gitdir.rstrip('/'))
    g = re.sub(r'/\.git$|\.git$', '', g)
    return os.path.basename(g)


@functools.lru_cache(maxsize=None)
def _repo_root(d, home):
    """(checkout dir, repo name) for the nearest enclosing git checkout below `home`, else None.
    A worktree's .git file names the main repository, so worktrees show the repo's name."""
    while d and d not in ('/', home):
        g = os.path.join(d, '.git')
        if os.path.isdir(g):
            return d, os.path.basename(d), True
        if os.path.isfile(g):
            try:
                with open(g, encoding='utf-8', errors='replace') as fh:
                    m = re.match(r'gitdir:\s*([^\r\n]+)', fh.read(4096).strip())
            except OSError:
                m = None
            gd = os.path.normpath(os.path.join(d, m.group(1).strip())) if m else ''
            return d, (_gitdir_repo(gd) if gd else '') or os.path.basename(d), False
        d = os.path.dirname(d)
    return None


@functools.lru_cache(maxsize=None)
def _worktree_project_repo(project_dir, home):
    """Repo name shared by the live worktrees in a T3 project directory (for deleted worktrees)."""
    try:
        names = sorted(os.listdir(project_dir))
    except OSError:
        return None
    for n in names:
        r = _repo_root(os.path.join(project_dir, n), home)
        if r and r[0] == os.path.join(project_dir, n):
            return r[1]
    return None


def file_label(p, home=HOME):
    """Display label for one instruction file: '<repo>/<path in repo>' inside a git checkout,
    '~/<path>' under a home directory, otherwise '…/<parent>/<name>'. Never contains a home path."""
    p = (p or '').replace('\\', '/')
    if not p:
        return 'Unknown file'
    if p.startswith('~/'):
        p = home + p[1:]
    if WIN_ABS_RE.match(p):
        # Windows drive or UNC path: never resolved against this machine's checkouts
        p = WIN_HOME_RE.sub('~', p) if WIN_HOME_RE.match(p) else '…/' + '/'.join(p.rstrip('/').split('/')[-2:])
    elif os.path.isabs(p):
        p = os.path.normpath(p)
        root = _repo_root(os.path.dirname(p), home)
        if root and root[2] and root[0].startswith(home + '/.') and not root[0].startswith(home + '/.t3/worktrees/'):
            root = None  # a git-tracked config dir such as ~/.codex/memories (a plain repo) reads better as ~/...
        wt = re.match(re.escape(home) + r'/\.t3/worktrees/([^/]+)/[^/]+/(.+)$', p)
        if root:
            # in-repo worktree directories (live or deleted) count as the repository's own file
            p = root[1] + '/' + WT_DIR_RE.sub('', os.path.relpath(p, root[0]))
        elif wt:
            # a worktree that no longer exists: name it after its project's repository
            p = ((_worktree_project_repo(os.path.join(home, '.t3', 'worktrees', wt.group(1)), home) or wt.group(1))
                 + '/' + WT_DIR_RE.sub('', wt.group(2)))
        elif p.startswith(home + '/'):
            p = '~' + p[len(home):]
        elif HOME_RE.match(p):
            p = HOME_RE.sub('~', p)
        else:
            p = '…/' + '/'.join(p.split('/')[-2:])
    # project directories such as ~/.claude/projects/-Users-name-repo encode the home path
    return HOME_SLUG_RE.sub('~', p.replace(home.replace('/', '-').replace('.', '-'), '~'))


def split_t3(text, base_label):
    """Split a prompt section at T3 markers. Returns [(path, chars)]."""
    hits = sorted((text.find(m), lab) for m, lab in T3_MARKERS if text.find(m) >= 0)
    if not hits:
        return [((INSTR, 'System prompt', base_label), len(text))]
    out = []
    if hits[0][0] > 0:
        out.append(((INSTR, 'System prompt', base_label), hits[0][0]))
    for i, (pos, lab) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        out.append(((INSTR, 'T3 Code', lab), end - pos))
    return out


def sp_section_label(sec, idx):
    h = sec.lstrip()[:160]
    if h.startswith('__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__'):
        return 'Harness guidance'
    if h.startswith('# Memory'):
        return 'Memory instructions'
    if h.startswith('# Environment') or h.startswith('<runtime_info>'):
        return 'Environment'
    if idx == 0 or h.startswith('You are'):
        return 'Harness base prompt'
    return 'Harness guidance'


def split_text(text, kind):
    """Classify a user-visible text block. Returns [(path, chars)] covering len(text)."""
    out, rest = [], text
    for rx, path in ((SYSREM_RE, (USER, 'System reminders')), (PASTE_RE, (USER, 'Pasted content')),
                     (NOTIF_RE, (USER, 'Task / subagent notifications')),
                     (ENVCTX_RE, (INSTR, 'Environment context')),
                     (PERM_RE, (INSTR, 'Codex permissions'))):
        n = sum(len(m.group(0)) for m in rx.finditer(rest))
        if n:
            out.append((path, n))
            rest = rx.sub('', rest)
    m = ROLE_RE.search(rest)
    if m:
        span = PROFILE_SPAN_RE.match(rest, m.start())
        n = len(span.group(0)) if span else len(rest)
        out.append(((INSTR, 'Subagent role profile', role_label(m.group(1))), n))
        rest = rest[:span.start()] + rest[span.end():] if span else ''
    if rest.strip():
        out.append(((USER, kind), len(rest)))
    return out


# ---------------------------------------------------------------- Claude parser
class Stats(dict):
    def inc(self, k, n=1):
        self[k] = self.get(k, 0) + n


def parse_claude(path, is_sub):
    st = Stats()
    recs = []
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except ValueError:
                st.inc('unparseable_lines')
                continue
            if not isinstance(o, dict):
                st.inc('skipped_lines')
                continue
            if o.get('isSidechain') and not is_sub:
                st.inc('skipped_lines')
                continue
            recs.append(o)
    # pass 1: dedupe assistant lines; last line carries the final usage
    final_usage, first_idx, msg_lines = {}, {}, defaultdict(list)
    for i, o in enumerate(recs):
        if o.get('type') != 'assistant':
            continue
        m = o.get('message') or {}
        key = claude_call_key(o, m, i)
        first_idx.setdefault(key, i)
        msg_lines[key].append(i)
        if m.get('usage'):
            final_usage[key] = (m.get('usage'), m.get('model') or '', o.get('timestamp'))
    role = None
    for o in recs:
        if o.get('type') == 'user' and not o.get('isMeta'):
            mt = ROLE_RE.search(_text_only((o.get('message') or {}).get('content')))
            if mt:
                role = mt.group(1)
                break
    tool_names = {}
    deferred_sizes = {}
    events = []
    done = set()
    for i, o in enumerate(recs):
        t = o.get('type')
        m = o.get('message') or {}
        if t == 'system' and o.get('subtype') == 'compact_boundary':
            events.append(('R',))
            st['_compacted'] = True
            st.inc('compactions')
            continue
        if t == 'attachment':
            _claude_attachment(o, events, deferred_sizes, st)
            continue
        if t == 'user':
            _claude_user(o, m, events, tool_names, deferred_sizes)
            continue
        if t != 'assistant':
            continue
        key = claude_call_key(o, m, i)
        if key in done:
            continue
        done.add(key)
        for c in _content_blocks(recs, msg_lines[key]):
            if c.get('type') == 'tool_use':
                srv, tl = tool_key(c.get('name'))
                tool_names[c.get('id')] = (srv, tl) + result_label_for(srv, tl, c.get('input'), o.get('cwd'))
        u = final_usage.get(key)
        if not u:
            continue
        usage, raw_model, ts = u
        if raw_model.startswith('<'):
            st.inc('synthetic_calls_skipped')
            continue
        inp = int(usage.get('input_tokens') or 0)
        cw = int(usage.get('cache_creation_input_tokens') or 0)
        cr = int(usage.get('cache_read_input_tokens') or 0)
        out = int(usage.get('output_tokens') or 0)
        cc = usage.get('cache_creation') or {}
        cw1 = int(cc.get('ephemeral_1h_input_tokens') or 0)
        cw5 = max(0, cw - cw1)
        if inp + cw + cr + out == 0:
            st.inc('empty_usage_skipped')
            continue
        outs = []
        for c in _content_blocks(recs, msg_lines[key]):
            ct = c.get('type')
            if ct == 'thinking':
                outs.append((P_THINK, len(c.get('thinking') or ''), 0))
            elif ct == 'redacted_thinking':
                outs.append((P_THINK, len(c.get('data') or ''), 0))
            elif ct == 'text':
                outs.append((P_TEXT, len(c.get('text') or ''), 0))
            elif ct == 'tool_use':
                srv, tl = tool_key(c.get('name'))
                outs.append((call_path(srv, tl), len(json.dumps(c.get('input') or {})) + len(c.get('name') or ''), 0)
                            + cmd_label_for(srv, tl, c.get('input')))
        events.append(('C', ts_epoch(ts), model_id(raw_model), inp + cw + cr, cr, cw5, cw1, out, None, outs, 'claude'))
        st.inc('calls')
    if role is None:
        role = 'native' if is_sub else 'top'
    st.pop('_last_tools', None)
    st.pop('_compacted', None)
    return {'src': 'claude', 'role': role, 'events': events}, st


def claude_call_key(o, m, i):
    """One API call per message.id: streamed lines of one message may differ in requestId
    (the first can lack it). requestId, then uuid, only when the message has no id."""
    return m.get('id') or o.get('requestId') or o.get('uuid') or 'line%d' % i


def _text_only(content):
    if isinstance(content, str):
        return content
    return ''.join(c.get('text') or '' for c in content or [] if isinstance(c, dict) and c.get('type') == 'text')


def _content_blocks(recs, idxs):
    out = []
    for i in idxs:
        for c in (recs[i].get('message') or {}).get('content') or []:
            if isinstance(c, dict):
                out.append(c)
    return out


def _claude_user(o, m, events, tool_names, deferred_sizes):
    content = m.get('content')
    if o.get('isCompactSummary') or (isinstance(content, str) and content.startswith(COMPACT_PREFIX)):
        events.append(('B', (USER, 'Compaction summaries'), chars_of(content), 0, B_EST))
        return
    if not (isinstance(content, list) and any(isinstance(x, dict) and x.get('type') == 'tool_result' for x in content)):
        # A new human turn: the API stops resending earlier turns' thinking (thinking_drop records
        # report thinkingBlocksSent=1), so drop carried thinking from the replayed context.
        events.append(('T',))
    kind = 'Typed prompts'
    if o.get('isMeta'):
        kind = 'Skill / command content'
    blocks = [content] if isinstance(content, str) else (content or [])
    for c in blocks:
        if isinstance(c, str):
            for p, n in split_text(c, kind):
                events.append(('B', p, n, 0, B_EST))
            continue
        if not isinstance(c, dict):
            continue
        ct = c.get('type')
        if ct == 'text':
            for p, n in split_text(c.get('text') or '', kind):
                events.append(('B', p, n, 0, B_EST))
        elif ct == 'image':
            events.append(('B', (USER, 'Images'), 0, IMAGE_TOKENS, B_IMG))
        elif ct == 'tool_result':
            srv, tl, *cmd = tool_names.get(c.get('tool_use_id'), (None, '?'))
            rc = c.get('content')
            imgs = count_images(rc)
            if imgs:
                events.append(('B', (RESULTS, 'Screenshots & images'), 0, IMAGE_TOKENS * imgs, B_IMG))
            if isinstance(rc, list) and rc and all(isinstance(x, dict) and x.get('type') == 'tool_reference' for x in rc):
                # ToolSearch loads deferred schemas; size them from the deferred_tools_record entries.
                for x in rc:
                    s2, t2 = tool_key(x.get('tool_name') or '')
                    n = deferred_sizes.get(x.get('tool_name'))
                    if n:
                        events.append(('B', (TOOLCTX, 'Loaded tool schemas', s2 or 'Built-in'), n, 0, B_EST))
                events.append(('B', result_path(srv, tl), sum(len(x.get('tool_name') or '') for x in rc), 0, B_EST))
                continue
            events.append(('B', result_path(srv, tl), chars_of(rc), 0, B_EST, *cmd))


def _claude_attachment(o, events, deferred_sizes, st):
    a = o.get('attachment') or {}
    t = a.get('type')
    if t == 'prompt_snapshot':
        pieces = []
        for idx, sec in enumerate(a.get('systemPrompt') or []):
            if not isinstance(sec, str):
                continue
            label = sp_section_label(sec, idx)
            pieces.extend(split_t3(sec, label))
        tools = a.get('tools')
        if tools is None:
            # snapshot without a tools list: the previous schemas are still sent
            pieces.extend(st.get('_last_tools', []))
            tools = []
        tool_pieces = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            srv, tl = tool_key(tool.get('name') or '?')
            if srv is None and tool.get('server'):
                srv = str(tool.get('server'))
                if re.match(r'^[0-9a-f-]{32,}$', srv):
                    srv = 'Unnamed server'
            n = len(tool.get('name') or '') + len(tool.get('description') or '') + len(json.dumps(tool.get('schema') or {}))
            tool_pieces.append((schema_path(srv, tl), n))
        if a.get('tools') is not None:
            st['_last_tools'] = tool_pieces
        pieces.extend(tool_pieces)
        if pieces:
            events.append(('P', pieces))
            st.inc('prompt_snapshots')
        return
    if t == 'deferred_tools_record':
        for e in a.get('entries') or []:
            if isinstance(e, dict) and e.get('name'):
                deferred_sizes[e['name']] = (len(e.get('name') or '') + len(e.get('description') or '')
                                             + len(json.dumps(e.get('input_schema') or {})))
        return
    rendered = o.get('rendered')
    rchars = sum(chars_of(r.get('content')) for r in rendered if isinstance(r, dict)) if isinstance(rendered, list) else None
    if t == 'nested_memory':
        c = a.get('content') or {}
        n = rchars if rchars is not None else chars_of(c.get('content'))
        fp = c.get('path') or a.get('displayPath')
        events.append(('B', (INSTR, 'Instruction files', instr_file_label(a.get('displayPath') or c.get('path'))), n, 0, B_EST,
                       (file_label(fp), HOW_IMPORT if c.get('parent') else HOW_ACCESS)))
    elif t == 'instructions':
        # older logs have no reason; then any instructions attachment after a compaction is the reload
        reason = a.get('reason')
        how = HOW_COMPACT if reason == 'compaction' or (reason is None and st.get('_compacted')) else HOW_LAUNCH
        for f in a.get('files') or []:
            if isinstance(f, dict):
                events.append(('B', (INSTR, 'Instruction files', instr_file_label(f.get('path'))), chars_of(f.get('content')), 0, B_EST,
                               (file_label(f.get('path')), how)))
    elif t == 'mcp_instructions_delta':
        names, blocks = a.get('addedNames') or [], a.get('addedBlocks') or []
        for i, name in enumerate(names):
            n = len(blocks[i]) if i < len(blocks) and isinstance(blocks[i], str) else 0
            events.append(('B', (INSTR, 'MCP server instructions', server_label(str(name))), n, 0, B_EST))
    elif t == 'skill_listing':
        events.append(('B', (INSTR, 'Skills listing'), rchars if rchars is not None else chars_of(a.get('content')), 0, B_EST))
    elif t == 'invoked_skills':
        for s in a.get('skills') or []:
            if isinstance(s, dict):
                events.append(('B', (INSTR, 'Invoked skills'), chars_of(s.get('content')), 0, B_EST))
    elif rchars is None:
        st.inc('unrendered_attachments')
    elif t == 'deferred_tools_delta':
        sd = len(json.dumps(a.get('surfacedDefinitions'))) if a.get('surfacedDefinitions') else 0
        sd = min(sd, rchars)
        if sd:
            events.append(('B', (TOOLCTX, 'Loaded tool schemas', 'Surfaced definitions'), sd, 0, B_EST))
        events.append(('B', (TOOLCTX, 'Deferred-tool name lists'), rchars - sd, 0, B_EST))
    elif t == 'agent_listing_delta':
        events.append(('B', (TOOLCTX, 'Agent-type list'), rchars, 0, B_EST))
    elif t in ('hook_success', 'hook_additional_context', 'hook_blocking_error'):
        events.append(('B', (USER, 'Hook output'), rchars, 0, B_EST))
    elif t in ('file', 'edited_text_file', 'compact_file_reference', 'directory'):
        events.append(('B', (USER, 'Attached files'), rchars, 0, B_EST))
    elif t == 'queued_command':
        events.append(('B', (USER, 'Typed prompts'), rchars, 0, B_EST))
    else:
        events.append(('B', (USER, 'System reminders'), rchars, 0, B_EST))


# ---------------------------------------------------------------- Codex parser
def parse_codex(path):
    st = Stats()
    recs = []
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except ValueError:
                st.inc('unparseable_lines')
                continue
            if isinstance(o, dict):
                recs.append(o)
            else:
                st.inc('skipped_lines')
    role = 'top'
    for o in recs:
        p = o.get('payload') or {}
        if o.get('type') == 'response_item' and p.get('type') == 'message' and p.get('role') == 'user':
            mt = ROLE_RE.search(_codex_text(p.get('content')))
            if mt:
                role = mt.group(1)
                break
    events, model, pending, seen = [], None, [], set()
    names, env = {}, {'cwd': None, 'compacted': False}
    # token_usage_record is per response. Older rollouts only have token_count events; those are
    # used only when a file has no token_usage_record, so the two never add up.
    use_tc = not any(o.get('type') == 'token_usage_record' for o in recs)
    tc_prev = [None]
    for o in recs:
        t = o.get('type')
        p = o.get('payload') or {}
        if t == 'session_meta':
            bi = p.get('base_instructions')
            text = bi.get('text') if isinstance(bi, dict) else bi
            if isinstance(text, str) and text:
                events.append(('P', split_t3(text, 'Codex base instructions')))
                st.inc('prompt_snapshots')
        elif t == 'turn_context':
            model = p.get('model') or model
            cwd = p.get('cwd')
            env['cwd'] = env['cwd'] if cwd is None or cwd == '' else cwd if isinstance(cwd, str) else None
        elif t == 'compacted':
            events.append(('R',))
            st.inc('compactions')
            pending = []
            env['compacted'] = True
            for it in p.get('replacement_history') or []:
                if isinstance(it, dict):
                    _codex_item(it, events, names, None, env)
        elif t == 'response_item':
            _codex_item(p, events, names, pending, env)
        elif t == 'token_usage_record':
            rid = p.get('response_id')
            if rid in seen:
                st.inc('duplicate_usage_skipped')
                continue
            seen.add(rid)
            if _codex_call(p.get('usage') or {}, o, model, pending, events, st):
                pending = []
        elif t == 'event_msg' and use_tc and p.get('type') == 'token_count':
            u = _token_count_usage(p.get('info'), tc_prev, st)
            if u and _codex_call(u, o, model, pending, events, st):
                st.inc('token_count_calls')
                pending = []
    return {'src': 'codex', 'role': role, 'events': events}, st


TC_FIELDS = ('input_tokens', 'cached_input_tokens', 'output_tokens', 'reasoning_output_tokens')


def _token_count_usage(info, prev, st):
    """Per-call usage from a token_count event. total_token_usage is cumulative per session and
    last_token_usage is the latest response; the event repeats after some turns with an unchanged
    total. The call is the change in the total; an unchanged total is a repeat. The first event,
    or a total that went down, falls back to last_token_usage."""
    if not isinstance(info, dict):
        return None
    tot, last = info.get('total_token_usage'), info.get('last_token_usage')
    if not isinstance(tot, dict):
        return last if isinstance(last, dict) else None
    before, prev[0] = prev[0], tot
    if before is None:
        return last if isinstance(last, dict) else tot
    d = {k: int(tot.get(k) or 0) - int(before.get(k) or 0) for k in TC_FIELDS}
    if not any(d.values()):
        st.inc('duplicate_usage_skipped')
        return None
    if min(d.values()) < 0:
        st.inc('token_count_resets')
        return last if isinstance(last, dict) else None
    return d


def _codex_call(u, o, model, pending, events, st):
    inp = int(u.get('input_tokens') or 0)
    out = int(u.get('output_tokens') or 0)
    if inp + out == 0:
        st.inc('empty_usage_skipped')
        return False
    cr = int(u.get('cached_input_tokens') or 0)
    cw = int(u.get('cache_write_input_tokens') or 0)
    reas = int(u.get('reasoning_output_tokens') or 0)
    outs = [x for x in pending if x[0] != P_THINK]
    events.append(('C', ts_epoch(o.get('timestamp')), model_id(model or 'unknown-codex'), inp, cr, cw, 0,
                   out, reas, outs, 'codex'))
    st.inc('calls')
    return True


def _codex_text(content):
    if isinstance(content, str):
        return content
    return ''.join(c.get('text') or '' for c in content or [] if isinstance(c, dict))


def _codex_item(p, events, names, pending, env=None):
    """Input items become 'B' events; model-produced items go to `pending` (outputs of the
    next usage record). Inside replacement history (pending None) everything is input."""
    pt = p.get('type')

    def emit_out(path, n, *cmd):
        if pending is None:
            events.append(('B', path, n, 0, B_EST, *cmd))
        else:
            pending.append((path, n, 0, *cmd))

    if pt == 'message':
        r = p.get('role')
        text = _codex_text(p.get('content'))
        if r == 'assistant':
            emit_out(P_TEXT, len(text))
        elif r == 'developer':
            for path, n, *sub in _codex_developer(text, env):
                events.append(('B', path, n, 0, B_EST, *sub))
        else:
            imgs = count_images(p.get('content'))
            if imgs:
                events.append(('B', (USER, 'Images'), 0, IMAGE_TOKENS * imgs, B_IMG))
            doc = AGENTS_DOC_RE.match(text)
            if doc:
                for path, n, *sub in _codex_agents_md(doc.group(0), env):
                    events.append(('B', path, n, 0, B_EST, *sub))
                text = text[doc.end():]
            for path, n in split_text(text, 'Typed prompts'):
                events.append(('B', path, n, 0, B_EST))
    elif pt == 'reasoning':
        if pending is None:
            events.append(('B', P_THINK, chars_of(p.get('summary')), 0, B_EST))
        else:
            pending.append((P_THINK, 0, 0))
    elif pt in ('function_call', 'custom_tool_call'):
        srv, tl = tool_key(p.get('name') or '?', p.get('namespace'))
        cmd = cmd_label_for(srv, tl, p.get('arguments') or p.get('input'))
        names[p.get('call_id')] = (srv, tl) + result_label_for(srv, tl, p.get('arguments') or p.get('input'),
                                                               (env or {}).get('cwd'))
        emit_out(call_path(srv, tl), len(p.get('arguments') or p.get('input') or '') + len(p.get('name') or ''), *cmd)
    elif pt in ('function_call_output', 'custom_tool_call_output'):
        srv, tl, *cmd = names.get(p.get('call_id'), (None, '?'))
        oo = p.get('output')
        imgs = count_images(oo)
        if imgs:
            events.append(('B', (RESULTS, 'Screenshots & images'), 0, IMAGE_TOKENS * imgs, B_IMG))
        n = len(oo) if isinstance(oo, str) else chars_of(oo)
        events.append(('B', result_path(srv, tl), n, 0, B_EST, *cmd))
    elif pt == 'agent_message':
        events.append(('B', (USER, 'Task / subagent notifications'), chars_of(p.get('content')), 0, B_EST))
    elif pt == 'compaction':
        events.append(('B', (USER, 'Compaction summaries'), chars_of(p.get('summary')), 0, B_EST))


def _codex_developer(text, env=None):
    out, rest = [], text
    n = sum(len(m.group(0)) for m in PERM_RE.finditer(rest))
    if n:
        out.append(((INSTR, 'Codex permissions'), n))
        rest = PERM_RE.sub('', rest)
    if any(mk in rest for mk, _ in T3_MARKERS) or 'T3 Code orchestration' in rest:
        out.extend((p if p[1] == 'T3 Code' else (INSTR, 'Developer messages'), k) for p, k in split_t3(rest, 'Developer messages'))
    else:
        # Only the anchored wrapper is file content; Codex's own instructions merely mention AGENTS.md.
        doc = AGENTS_DOC_RE.match(rest)
        if doc:
            out.extend(_codex_agents_md(doc.group(0), env))
            rest = rest[doc.end():]
        if rest.strip():
            out.append(((INSTR, 'Developer messages'), len(rest)))
    return out


def _codex_agents_md(text, env=None):
    """Codex sends ~/.codex/AGENTS.md, then '--- project-doc ---', then the project docs for the cwd
    named in the header. Returns [(path, chars, (file label, how it entered))]."""
    path = (INSTR, 'AGENTS.md instructions')
    how = HOW_COMPACT if (env or {}).get('compacted') else HOW_LAUNCH
    m = AGENTS_HDR_RE.search(text)
    project = file_label(m.group(1).strip().rstrip('/') + '/AGENTS.md') if m else 'Project AGENTS.md'
    cut = text.find(PROJECT_DOC_SEP)
    if cut < 0:
        return [(path, len(text), (project, how))]
    return [(path, cut, (file_label('~/.codex/AGENTS.md'), how)), (path, len(text) - cut, (project, how))]


# ---------------------------------------------------------------- file discovery
def discover(claude_root, codex_root, min_mtime):
    jobs = []
    croot = os.path.expanduser(claude_root)
    for dp, _dn, fn in os.walk(croot):
        for f in fn:
            if f.endswith('.jsonl'):
                p = os.path.join(dp, f)
                if os.path.getmtime(p) >= min_mtime:
                    jobs.append(('claude', p, os.sep + 'subagents' + os.sep in p + os.sep))
    xroot = os.path.expanduser(codex_root)
    for dp, _dn, fn in os.walk(xroot):
        for f in fn:
            if f.endswith('.jsonl'):
                p = os.path.join(dp, f)
                if os.path.getmtime(p) >= min_mtime:
                    jobs.append(('codex', p, False))
    jobs.sort()
    return jobs


def parse_job(job):
    src, path, is_sub = job
    try:
        return parse_claude(path, is_sub) if src == 'claude' else parse_codex(path)
    except OSError:
        st = Stats()
        st.inc('unreadable_files')
        return None, st
    except Exception:  # unexpected record shape: count it in meta rather than abort the run
        st = Stats()
        st.inc('unparseable_files')
        return None, st


# ---------------------------------------------------------------- calibration
def calibrate(sessions):
    """Chars/token per source from context growth between consecutive calls.

    For calls k-1, k with no compaction or prefix change in between, the context grows by
    the previous call's output plus the blocks appended. ratio = text chars appended /
    (ΔC − previous output − fixed-size image tokens). Median over pairs with ≥2,000 chars.
    """
    ratios = defaultdict(list)
    for s in sessions:
        prev, chars, fixed, broken = None, 0, 0, False
        for ev in s['events']:
            k = ev[0]
            if k == 'B':
                chars += ev[2]
                fixed += ev[3]
            elif k in ('R', 'P', 'T'):
                broken = True
            elif k == 'C':
                C, out = ev[3], ev[7]
                if prev is not None and not broken and chars >= 2000:
                    dt = C - prev[0] - prev[1] - fixed
                    if dt > 0:
                        ratios[s['src']].append(chars / dt)
                prev, chars, fixed, broken = (C, out), 0, 0, False
    result = {}
    for src in ('claude', 'codex'):
        r = sorted(ratios.get(src, []))
        if r:
            med = r[len(r) // 2]
            lo, hi = CALIBRATION_BOUNDS
            used = med if lo <= med <= hi else FALLBACK_CHARS_PER_TOKEN
            result[src] = {'chars_per_token': round(used, 4), 'median_observed': round(med, 4),
                           'pairs': len(r), 'p25': round(r[len(r) // 4], 4), 'p75': round(r[3 * len(r) // 4], 4),
                           'fallback': used != med}
        else:
            result[src] = {'chars_per_token': FALLBACK_CHARS_PER_TOKEN, 'pairs': 0, 'fallback': True}
    return result


# ---------------------------------------------------------------- attribution
def apportion(items, total):
    """Integer apportionment (largest remainder) of float weights to an exact integer total."""
    if total <= 0 or not items:
        return [0] * len(items)
    s = sum(items)
    if s <= 0:
        out = [0] * len(items)
        out[-1] = total
        return out
    raw = [w * total / s for w in items]
    base = [int(x) for x in raw]
    rem = total - sum(base)
    order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - base[i]), i))
    for i in order[:rem]:
        base[i] += 1
    return base


def rates(model, src, cw5, cw1, prompt):
    """Per-token (input, cache read, cache write, output) rates for one call; prompt is the call's
    total input (uncached + cache reads + cache writes) and picks the tier."""
    pr = PRICING.get(model)
    if not pr:
        return None
    if 'long' in pr and prompt > pr['long']['above']:
        pr = pr['long']
    rin, rcr = pr['input'] / 1e6, pr['cache_read'] / 1e6
    if src == 'claude' and cw5 + cw1:
        rcw = rin * (cw5 * CACHE_WRITE_MULT['5m'] + cw1 * CACHE_WRITE_MULT['1h']) / (cw5 + cw1)
    else:
        rcw = rin
    return rin, rcr, rcw, pr['output'] / 1e6


def size_bin(tokens):
    """Index of the size bin (HIST_EDGES) for one occurrence of the given size."""
    return bisect.bisect_right(HIST_EDGES, tokens)


def _bin_add(d, q, b, tok):
    g = d.setdefault(q, {})
    g[b] = g.get(b, 0.0) + tok


def _leaf(path, sub):
    """Leaf path of a labelled block: a label is one node, a tuple label (file, how it entered) nests."""
    return path + (sub if type(sub) is tuple else (sub,))


def split_share(fs, total):
    """Apportions a group's integer share among its labels: [(label, tokens)]. Tuple labels are split
    by their first part first, then within it, so each first-level node gets exactly the share it gets
    without the deeper labels (per-file totals don't depend on how the file entered context)."""
    names = list(fs)
    if not any(type(f) is tuple and len(f) > 1 for f in names):
        return list(zip(names, apportion([max(0.0, fs[f]) for f in names], total)))
    heads = {}
    for f in names:
        heads.setdefault(f[0] if type(f) is tuple else f, []).append(f)
    out = []
    for h, ha in zip(heads, apportion([sum(max(0.0, fs[f]) for f in heads[h]) for h in heads], total)):
        out.extend(zip(heads[h], apportion([max(0.0, fs[f]) for f in heads[h]], ha)))
    return out


class Days:
    """Local calendar days first..last. A call at ts belongs to the day it falls on if start_ts <= ts < end_ts."""

    def __init__(self, first, last, start_ts, end_ts):
        self.first, self.last, self.start_ts, self.end_ts = first, last, start_ts, end_ts

    def __len__(self):
        return (self.last - self.first).days + 1

    def hits(self, ts):
        if not self.start_ts <= ts < self.end_ts:
            return []
        i = (datetime.date.fromtimestamp(ts) - self.first).days
        return [i] if 0 <= i < len(self) else []

    def iso(self, i):
        return (self.first + datetime.timedelta(days=i)).isoformat()


def local_midnight(day):
    return datetime.datetime.combine(day, datetime.time()).timestamp()


def window_hits(windows, ts):
    """Indexes of the windows holding ts: a Days, or a list of (start, end) spans."""
    if ts is None:
        return []
    if isinstance(windows, Days):
        return windows.hits(ts)
    return [i for i, (a, b) in enumerate(windows) if a <= ts < b]


def attribute(session, ratios, windows, acc, recon, counters, basis, fits=None, hacc=None):
    """Replays one session. acc[(path, day, model, role)] = [billed, unique, usd], day being the index of
    the window that holds the call. hacc, if given, gets the same cells split by occurrence size:
    hacc[(path, day, model, role)] = {size bin: [occurrences, billed, unique, usd]}."""
    role = session['role']
    ratio = ratios[session['src']]['chars_per_token']
    ctx, new, pend_u = {}, {}, {}
    # Per-file shares of file-group paths ({path: {file: tokens}}). The group is allocated as one
    # path, exactly as without files, and its integer share is then apportioned among its files.
    fctx, fnew, fpend = {}, {}, {}
    prefix, pmax = [], {}
    # The same tokens by occurrence size, keyed by leaf (path, or path + (file,)): {leaf: {bin: tokens}}.
    # bpend also counts the occurrences whose unique tokens the next call credits: {leaf: {bin: [tokens, n]}}.
    bctx, bnew, bpend = {}, {}, {}
    pbins, pfresh = {}, False   # prompt prefix sections: {path: {bin: [tokens, sections]}}
    for ev in session['events']:
        k = ev[0]
        if k == 'B':
            path, chars, fixed, b = ev[1:5]
            sub = ev[5] if len(ev) > 5 else (CMD_UNPARSED if is_cmd_group(path) else None)
            tok = chars / ratio + fixed
            if tok <= 0:
                continue
            ctx[path] = ctx.get(path, 0.0) + tok
            new[path] = new.get(path, 0.0) + tok
            pend_u[path] = pend_u.get(path, 0.0) + tok
            q, hb = (path if sub is None else _leaf(path, sub)), size_bin(tok)
            _bin_add(bctx, q, hb, tok)
            _bin_add(bnew, q, hb, tok)
            c = bpend.setdefault(q, {}).setdefault(hb, [0.0, 0])
            c[0] += tok
            c[1] += 1
            if sub is not None:
                for d in (fctx, fnew, fpend):
                    g = d.setdefault(path, {})
                    g[sub] = g.get(sub, 0.0) + tok
                basis[_leaf(path, sub)].add(b)
            else:
                basis[path].add(b)
        elif k == 'R':
            ctx, new, pend_u = {}, {}, {}
            fctx, fnew, fpend = {}, {}, {}
            bctx, bnew, bpend = {}, {}, {}
        elif k == 'T':
            ctx.pop(P_THINK, None)
            new.pop(P_THINK, None)
            bctx.pop(P_THINK, None)
            bnew.pop(P_THINK, None)
        elif k == 'P':
            agg = {}
            pbins, pfresh = {}, True
            for path, chars in ev[1]:
                agg[path] = agg.get(path, 0.0) + chars / ratio
                if chars > 0:
                    c = pbins.setdefault(path, {}).setdefault(size_bin(chars / ratio), [0.0, 0])
                    c[0] += chars / ratio
                    c[1] += 1
            prefix = list(agg.items())
            for path, _ in prefix:
                basis[path].add(B_EST)
        elif k == 'C':
            _, ts, model, C, cr, cw5, cw1, out, reas, outs, src = ev
            ris = window_hits(windows, ts)
            E = sum(ctx.values())
            P = sum(v for _, v in prefix)
            if fits is not None and C:
                fits[src].append((E + P) / C)
            sm, sp = 1.0, 1.0
            if C < E:
                sm, sp = (C / E if E else 0.0), 0.0
                counters['clamp_events'] += 1
            elif C < E + P:
                sp = (C - E) / P
                counters['prefix_scaled_events'] += 1
            resid = max(0.0, C - E * sm - P * sp)
            paths = [p for p, _ in prefix] + [UNATTRIB]
            weights = [v * sp for _, v in prefix] + [resid]
            files = [None] * len(paths)
            kinds = ['p'] * len(prefix) + ['u']
            # cache reads cover the oldest context: prefix, then carried blocks, then this turn's new blocks
            for p, v in ctx.items():
                old = v - new.get(p, 0.0)
                if old > 0:
                    paths.append(p)
                    weights.append(old * sm)
                    kinds.append('o')
                    fn = fnew.get(p, {})
                    files.append({f: w - fn.get(f, 0.0) for f, w in fctx[p].items()} if p in fctx else None)
            for p, v in new.items():
                if v > 0:
                    paths.append(p)
                    weights.append(v * sm)
                    kinds.append('n')
                    files.append(fnew.get(p))
            alloc = apportion(weights, C)
            # split each file group's integer share among its files; the files' spans stay contiguous
            # inside the group's span, so cache-read and cache-write pricing adds up to the group's
            spans, span_kinds = [], []
            for p, a, fs, kd in zip(paths, alloc, files, kinds):
                if fs:
                    for f, fa in split_share(fs, a):
                        if fa:
                            spans.append((_leaf(p, f), fa))
                            span_kinds.append(kd)
                else:
                    spans.append((p, a))
                    span_kinds.append(kd)
            rt = rates(model, src, cw5, cw1, C)
            cw = cw5 + cw1
            if ris:
                recon_add(recon, ris, model, C + out)
                counters['calls_in_range'] += 1
                if rt is None:
                    counters['unpriced_calls:' + model] += 1
            off = 0
            n_prefix = len(prefix) + 1
            uniq, prefix_u = {}, {}
            for i, (p, a, fs) in enumerate(zip(paths, alloc, files)):
                if fs:
                    continue  # file groups: unique tokens are added per file below
                if i < n_prefix:
                    u = max(0, a - pmax.get(p, 0))
                    if a > pmax.get(p, 0):
                        pmax[p] = a
                    prefix_u[p] = u
                else:
                    u = 0
                uniq[p] = uniq.get(p, 0) + u
            for p, v in pend_u.items():
                if p in fpend:
                    for f, w in fpend[p].items():
                        uniq[_leaf(p, f)] = uniq.get(_leaf(p, f), 0) + w * sm
                else:
                    uniq[p] = uniq.get(p, 0) + v * sm
            usd_by = defaultdict(float)
            for p, a in spans:
                if rt and a:
                    lo, hi = off, off + a
                    c_cr = max(0, min(hi, cr) - max(lo, 0))
                    c_cw = max(0, min(hi, cr + cw) - max(lo, cr))
                    usd_by[p] += c_cr * rt[1] + c_cw * rt[2] + (a - c_cr - c_cw) * rt[0]
                off += a
            billed_by = defaultdict(int)
            for p, a in spans:
                billed_by[p] += a
            hcall = _call_bins(spans, span_kinds, pbins, bctx, bnew, rt, cr, cw, counters) \
                if hacc is not None and ris else None
            if hcall is not None:
                _call_unique_bins(hcall, prefix_u, pbins, pfresh, pend_u, fpend, bpend, sm, alloc[len(prefix)])
            # output: measured reasoning first, the rest split by text size
            o_paths, o_w, o_fixed = [], [], []
            if reas:
                o_paths.append(P_THINK)
                basis[P_THINK].add(B_MEAS)
            rest_out = out - min(reas or 0, out)
            if outs and sum(x[1] for x in outs) > 0:
                o_alloc = apportion([x[1] for x in outs], rest_out)
                o_list = [(x[0], a, x[3] if len(x) > 3 else None) for x, a in zip(outs, o_alloc)]
            else:
                o_list = [(P_TEXT, rest_out, None)] if rest_out else []
            if reas:
                o_list = [(P_THINK, min(reas, out), None)] + o_list
            new, fnew, bnew = {}, {}, {}
            for p, a, sub in o_list:
                if not a:
                    continue
                if sub is None and is_cmd_group(p):
                    sub = CMD_UNPARSED
                q = p if sub is None else _leaf(p, sub)
                billed_by[q] += a
                uniq[q] = uniq.get(q, 0) + a
                if rt:
                    usd_by[q] += a * rt[3]
                hb = size_bin(a)
                _bin_add(bctx, q, hb, a)
                _bin_add(bnew, q, hb, a)
                if hcall is not None:  # an output item is one occurrence, billed and unique in full
                    c = hcall.setdefault(q, {}).setdefault(hb, [0, 0, 0.0, 0.0])
                    c[0] += 1
                    c[1] += a
                    c[2] += a
                    c[3] += a * rt[3] if rt else 0.0
                if p != P_THINK or not reas:
                    basis[q].add(B_OUT if src == 'claude' or p != P_THINK else B_MEAS)
                ctx[p] = ctx.get(p, 0.0) + a
                new[p] = new.get(p, 0.0) + a
                if sub is not None:
                    for d in (fctx, fnew):
                        g = d.setdefault(p, {})
                        g[sub] = g.get(sub, 0.0) + a
            pend_u, fpend, bpend = {}, {}, {}
            pfresh = False
            basis[UNATTRIB].add(B_RESID)
            if ris:
                for p in set(billed_by) | set(uniq):
                    b, u, d = billed_by.get(p, 0), uniq.get(p, 0), usd_by.get(p, 0.0)
                    for ri in ris:
                        key = (p, ri, model, role)
                        cell = acc.get(key)
                        if cell is None:
                            acc[key] = [b, u, d]
                        else:
                            cell[0] += b
                            cell[1] += u
                            cell[2] += d
            for q, bins in (hcall or {}).items():
                for ri in ris:
                    hc = hacc.setdefault((q, ri, model, role), {})
                    for hb, x in bins.items():
                        c = hc.get(hb)
                        if c is None:
                            hc[hb] = list(x)
                        else:
                            for j in range(4):
                                c[j] += x[j]


def _call_bins(spans, span_kinds, pbins, bctx, bnew, rt, cr, cw, counters):
    """One call's input split by occurrence size: {leaf: {bin: [occurrences, billed, unique, usd]}}.
    Each span's integer share is apportioned among the size bins of the items it covers, in bin order,
    so the bins' cache-read, cache-write and input spans stay inside the leaf's span."""
    out, off = {}, 0
    for (q, a), kd in zip(spans, span_kinds):
        if kd == 'p':
            w = {b: c[0] for b, c in pbins.get(q, {}).items()}
        elif kd == 'u':
            w = {size_bin(a): 1.0}
        elif kd == 'n':
            w = dict(bnew.get(q, {}))
        else:
            bn = bnew.get(q, {})
            w = {b: max(0.0, t - bn.get(b, 0.0)) for b, t in bctx.get(q, {}).items()}
        bins = sorted(b for b in w if w[b] > 0)
        if a and not bins:
            counters['hist_fallback'] += 1
            bins, w = [size_bin(a)], {size_bin(a): 1.0}
        g = out.setdefault(q, {})
        for b, ba in zip(bins, apportion([w[b] for b in bins], a)):
            c = g.setdefault(b, [0, 0, 0.0, 0.0])
            c[1] += ba
            if rt and ba:
                lo, hi = off, off + ba
                c_cr = max(0, min(hi, cr) - max(lo, 0))
                c_cw = max(0, min(hi, cr + cw) - max(lo, cr))
                c[3] += c_cr * rt[1] + c_cw * rt[2] + (ba - c_cr - c_cw) * rt[0]
            off += ba
    return out


def _call_unique_bins(out, prefix_u, pbins, pfresh, pend_u, fpend, bpend, sm, resid):
    """Adds one call's unique tokens and new occurrences to its size bins (see attribute)."""
    for p, u in prefix_u.items():
        if p == UNATTRIB:
            if resid:
                c = out.setdefault(p, {}).setdefault(size_bin(resid), [0, 0, 0.0, 0.0])
                c[0] += 1
                c[2] += u
            continue
        sec = pbins.get(p, {})
        bins = sorted(sec)
        for b, bu in zip(bins, apportion([sec[b][0] for b in bins], u)):
            c = out.setdefault(p, {}).setdefault(b, [0, 0, 0.0, 0.0])
            c[2] += bu
            if pfresh:
                c[0] += sec[b][1]
    for p in pend_u:
        for q in ([_leaf(p, f) for f in fpend[p]] if p in fpend else [p]):
            for b, (t, n) in bpend.get(q, {}).items():
                c = out.setdefault(q, {}).setdefault(b, [0, 0, 0.0, 0.0])
                c[0] += n
                c[2] += t * sm


def recon_add(recon, ris, model, n):
    for ri in ris:
        recon[(ri, model)] = recon.get((ri, model), 0) + n


# ---------------------------------------------------------------- tree building
def build_tree(acc, model_ids, role_ids, basis, top_files=FILE_TOP_N, top_cmds=CMD_TOP_N, hacc=None):
    """The drill-down tree. A leaf's 'v' lists cells with metrics or occurrences as [day, model, role, billed, unique, usd].
    With hacc (see attribute), each leaf also gets 'h', the size histogram of each 'v' cell in the same order."""
    mi = {m: i for i, m in enumerate(model_ids)}
    ki = {r: i for i, r in enumerate(role_ids)}
    leaves, hists = {}, {}
    for (path, day, model, role), cell in acc.items():
        v = leaves.get(path)
        if v is None:
            v = leaves[path] = {}
            hists[path] = {}
        key = (day, mi[model], ki[role])
        c = v.get(key)
        if c is None:
            v[key] = [cell[0], cell[1], cell[2]]
        else:
            c[0] += cell[0]
            c[1] += cell[1]
            c[2] += cell[2]
    for (path, day, model, role), bins in (hacc or {}).items():
        _merge_bins(hists[path].setdefault((day, mi[model], ki[role]), {}), bins)
    root = {'children': {}}
    # Longer paths first, so a path that is a prefix of another becomes its "(direct)" child.
    for path, v in sorted(leaves.items(), key=lambda kv: (-len(kv[0]), kv[0])):
        node = root
        for part in path[:-1]:
            node = node['children'].setdefault(part, {'children': {}})
        last = path[-1]
        leaf = {'v': v, 'basis': set(basis.get(path, set())), 'h': hists[path]}
        if last in node['children']:
            node['children'][last]['children'][''] = leaf
        else:
            node['children'][last] = leaf
    def total(n):
        if 'v' in n:
            return sum(c[0] for c in n['v'].values())
        return sum(total(c) for c in n['children'].values())

    def leaf_cells(n):
        if 'v' in n:
            return [(n['v'], n['basis'], n['h'])]
        out = []
        for c in n['children'].values():
            out.extend(leaf_cells(c))
        return out

    def merge(nodes):
        v, h, b = {}, {}, set()
        for n in nodes:
            for lv, lb, lh in leaf_cells(n):
                b |= lb
                for key, x in lv.items():
                    c = v.get(key)
                    if c is None:
                        v[key] = list(x)
                    else:
                        for j in range(3):
                            c[j] += x[j]
                for key, bins in lh.items():
                    _merge_bins(h.setdefault(key, {}), bins)
        return {'v': v, 'basis': b, 'h': h}

    def cum_round(v):
        """Unique tokens and dollars per (day, model, role) cell, rounded along the days so that every run of
        consecutive days sums to within one unit of its raw total, and all days to the rounded total."""
        out, run = {}, {}
        for key in sorted(v, key=lambda k: (k[1], k[2], k[0])):
            c, (u, d, ur, dr) = v[key], run.get(key[1:], (0.0, 0.0, 0, 0))
            u, d = u + c[1], d + c[2]
            nu, nd = round(u), round(round(d, 4) * 10 ** 4)
            out[key] = [c[0], nu - ur, (nd - dr) / 10 ** 4]
            run[key[1:]] = (u, d, nu, nd)
        return out

    def round_split(vs, target_v=None):
        """Round the children's unique tokens and dollars so they add up to the rounded parent, the value the
        unsplit leaf would serialize (cum_round), or to the already rounded target_v. Each child carries its
        rounding along the days, so its running total stays within two units of its raw running total, and its
        row for a day with no negative value is never negative."""
        zero = (0, 0.0, 0.0)
        if target_v is None:
            target_v = {}
            for v in vs:
                for key, x in v.items():
                    t = target_v.setdefault(key, [0, 0.0, 0.0])
                    for j in range(3):
                        t[j] += x[j]
            target_v = cum_round(target_v)
        carry = {}
        for key in sorted(set(target_v).union(*vs), key=lambda k: (k[1], k[2], k[0])):
            for j, scale in ((1, 1), (2, 10 ** 4)):
                raw, given = carry.setdefault((key[1:], j), ([0.0] * len(vs), [0] * len(vs)))
                day = [v.get(key, zero)[j] * scale for v in vs]
                for i, x in enumerate(day):
                    raw[i] += x
                xs = [r - g for r, g in zip(raw, given)]
                fl = [math.floor(x) if d < 0 else max(math.floor(x), 0) for x, d in zip(xs, day)]
                total_ = target_v.get(key, zero)[j]
                target = round(total_) if j == 1 else round(round(total_, 4) * scale)
                extra = int(target) - sum(fl)
                for i in sorted(range(len(xs)), key=lambda i: fl[i] - xs[i])[:max(extra, 0)]:
                    fl[i] += 1
                for i in sorted(range(len(xs)), key=lambda i: (fl[i] <= 0, xs[i] - fl[i]))[:max(-extra, 0)]:
                    fl[i] -= 1
                for i, x in enumerate(fl):
                    given[i] += x
                for v, x in zip(vs, fl):
                    if key not in v:
                        if not x:
                            continue
                        v[key] = [0, 0.0, 0.0]
                    v[key][j] = x if j == 1 else x / scale

    def round_leaves(n, target_v=None):
        """Rounds every leaf under an internal node, which once was one leaf, to that leaf's rounded value."""
        round_split([lv for lv, _, _ in leaf_cells(n)], target_v)

    def file_like(path):
        """File nodes (children of a file group) and the Read result node: split leaves that keep their unsplit sums."""
        return (len(path) > 1 and (is_file_group(path[:-1]) or is_explicit_group(path[:-1]))) or path == READ_RESULT

    counter = [0]

    def emit(label, n, path, parent_label):
        counter[0] += 1
        nid = 'n%d' % counter[0]
        kids = sorted(n.get('children', {}).items(), key=lambda kv: (-total(kv[1]), kv[0]))
        if 'v' in n:
            if not label:
                label = 'Other output' if len(path) > 2 and is_cmd_group(path[:-2]) else '%s (direct)' % parent_label
            cells = cum_round(n['v'])
            leaf = {'id': nid, 'label': label, 'basis': ' + '.join(sorted(n['basis'])) or B_EST,
                    'v': compact(cells, n['h'])}
            if hacc is not None:
                leaf['h'] = compact_hist(n['h'], cells)
            if len(path) > 1 and is_explicit_group(path[:-1]) and label != SEVERAL_FILES and not label.startswith('Other ('):
                leaf['xf'] = label
            if len(path) > 1 and is_file_group(path[:-1]) and not label.startswith('Other ('):
                leaf['fl'] = label
            return leaf
        if n.get('rounded') is None and file_like(path):
            round_leaves(n)
        if not path:
            kids.sort(key=lambda kv: TOP_ORDER.index(kv[0]) if kv[0] in TOP_ORDER else 99)
        elif is_file_group(path) or is_explicit_group(path):
            if len(kids) > top_files:
                rest = kids[top_files:]
                kids = kids[:top_files] + [('Other (%d files)' % len(rest), merge([c for _, c in rest]))]
        elif is_cmd_group(path):
            # Unparsed / no-command buckets stay visible; rejected labels always fold into Other
            pinned = [kv for kv in kids if kv[0] in (CMD_UNPARSED, CMD_NO_SHELL)]
            named = [kv for kv in kids if kv[0] not in (CMD_UNPARSED, CMD_NO_SHELL, CMD_REJECTED)]
            rest = named[top_cmds:] + [kv for kv in kids if kv[0] == CMD_REJECTED]
            kids = named[:top_cmds] + pinned
            if rest:
                lab = CMD_REJECTED if [k for k, _ in rest] == [CMD_REJECTED] else 'Other (%d commands)' % len(rest)
                kids.append((lab, merge([c for _, c in rest])))
            kids.sort(key=lambda kv: (-total(kv[1]), kv[0]))
            # a command with explicit instruction-file reads is internal: round it as one value, then its leaves
            vs = [c['v'] if 'v' in c else merge([c])['v'] for _, c in kids]
            round_split(vs)
            for (_, c), v in zip(kids, vs):
                if 'v' not in c:
                    round_leaves(c, v)
                    c['rounded'] = True
        elif len(kids) > MAX_CHILDREN:
            keep, rest = kids[:MAX_CHILDREN - 1], kids[MAX_CHILDREN - 1:]
            kids = keep + [('Other %s' % (label or '').lower(), merge([c for _, c in rest]))]
        node = {'id': nid, 'label': label, 'children': [emit(k, c, path + (k,), label) for k, c in kids]}
        if len(path) > 1 and is_file_group(path[:-1]):
            node['fl'] = label
        return node

    def kept(v, h):
        return [k for k in sorted(v) if v[k][0] or round(v[k][1]) or round(v[k][2], 4)
                or any(b[0] for b in h.get(k, {}).values())]

    def compact(v, h):
        return [[*k, int(v[k][0]), int(round(v[k][1])), round(v[k][2], 4)] for k in kept(v, h)]

    def compact_hist(h, v):
        """Per compact() cell, [[bin, occurrences, billed, unique, usd], ...]. Unique tokens and dollars are
        rounded (largest remainder) to add up to the cell compact() serializes; billed is exact. A cell with no
        calls of its own, which a split leaf's carried rounding can give tokens, uses the bins of its (model,
        role) pair's latest earlier day with calls, with no occurrences or billed tokens."""
        out, last = [], {}
        for key in sorted(kept(v, h), key=lambda k: (k[1], k[2], k[0])):
            bins, c = h.get(key, {}), v[key]
            keys = sorted(b for b, x in bins.items() if any(x))
            if keys:
                last[key[1:]] = bins
            elif key[1:] in last:
                bins = {b: [0, 0, x[2], x[3]] for b, x in last[key[1:]].items()}
                keys = sorted(b for b, x in bins.items() if any(x))
            us = apportion([max(0.0, bins[b][2]) for b in keys], int(round(c[1])))
            ds = apportion([max(0.0, bins[b][3]) for b in keys], int(round(round(c[2], 4) * 10 ** 4)))
            out.append((key, [[b, bins[b][0], bins[b][1], u, round(d / 10 ** 4, 4)] for b, u, d in zip(keys, us, ds)]))
        return [x for _, x in sorted(out)]

    return emit('All tokens', root, (), '')['children']


def _merge_bins(into, bins):
    for b, x in bins.items():
        c = into.get(b)
        if c is None:
            into[b] = list(x)
        else:
            for j in range(4):
                c[j] += x[j]


def leaf_sums(tree):
    """Billed leaf sums: {(day, model index, role index): tokens}."""
    s = defaultdict(int)

    def walk(n):
        if 'children' in n:
            for c in n['children']:
                walk(c)
            return
        for d, m, k, b, _, _ in n['v']:
            s[(d, m, k)] += b
    for n in tree:
        walk(n)
    return s


# ---------------------------------------------------------------- independent oracle
def _oracle_time(s):
    # Deliberately not parse_ts: fromisoformat after rewriting Z (Python 3.9 rejects a bare Z).
    try:
        d = datetime.datetime.fromisoformat(s.replace('Z', '+00:00'))
    except (AttributeError, ValueError):
        return None
    return (d if d.tzinfo else d.replace(tzinfo=datetime.timezone.utc)).timestamp()


def oracle_file(job):
    """Minimal re-read of one log: {call key: (epoch, raw model, input + cache + output)} and the role.
    Shares only the role regex and the model alias table with the parser."""
    src, path, is_sub = job
    calls, role, model, tc, last_total = {}, None, None, [], None
    try:
        with open(path, encoding='utf-8', errors='replace') as fh:
            rows = [json.loads(x) for x in fh if x.strip().startswith('{')]
    except (OSError, ValueError):
        return None, {}, 1
    for o in rows:
        t, m, p = o.get('type'), o.get('message') or {}, o.get('payload') or {}
        if role is None and t == 'user' and not o.get('isMeta'):
            c = m.get('content')
            text = c if isinstance(c, str) else ' '.join(
                b.get('text') or '' for b in c or [] if isinstance(b, dict) and b.get('type') == 'text')
            hit = ROLE_RE.search(text)
            role = hit.group(1) if hit else None
        if role is None and t == 'response_item' and p.get('type') == 'message' and p.get('role') == 'user':
            hit = ROLE_RE.search(' '.join(b.get('text') or '' for b in p.get('content') or [] if isinstance(b, dict)))
            role = hit.group(1) if hit else None
        if t == 'assistant' and m.get('usage') and not str(m.get('model') or '').startswith('<'):
            u = m['usage']
            n = sum(int(u.get(k) or 0) for k in ('input_tokens', 'cache_creation_input_tokens',
                                                 'cache_read_input_tokens', 'output_tokens'))
            calls['claude:%s' % (m.get('id') or o.get('requestId') or o.get('uuid'))] = (
                _oracle_time(o.get('timestamp')), m.get('model') or '', n)
        elif t == 'turn_context':
            model = p.get('model') or model
        elif t == 'token_usage_record':
            u = p.get('usage') or {}
            calls['codex:%s' % p.get('response_id')] = (
                _oracle_time(o.get('timestamp')), model or 'unknown-codex',
                int(u.get('input_tokens') or 0) + int(u.get('output_tokens') or 0))
        elif t == 'event_msg' and p.get('type') == 'token_count' and isinstance(p.get('info'), dict):
            total = p['info'].get('total_token_usage')
            if total != last_total:  # a repeated cumulative total is the same call
                last = p['info'].get('last_token_usage') or {}
                tc.append((_oracle_time(o.get('timestamp')), model or 'unknown-codex',
                           int(last.get('input_tokens') or 0) + int(last.get('output_tokens') or 0)))
            last_total = total
    if src == 'codex' and not calls:
        calls = {'codex-tc:%s:%d' % (path, i): v for i, v in enumerate(tc)}
    return role or ('native' if is_sub else 'top'), calls, 0


def run_oracle(jobs, windows, results):
    """Sums logged usage per (day index, model, role). Duplicate call keys across files count once
    (first file in sorted order wins)."""
    out, seen, dup_keys, unreadable = defaultdict(int), set(), 0, 0
    for (role, calls, bad) in results:
        unreadable += bad
        for key, (ts, raw_model, n) in calls.items():
            if key in seen:
                dup_keys += 1
                continue
            seen.add(key)
            for d in window_hits(windows, ts) if n else ():
                out[(d, model_id(raw_model), role)] += n
    return out, {'calls': len(seen), 'cross_file_duplicate_keys': dup_keys, 'unreadable_files': unreadable}


def compare_oracle(oracle, sums, days, model_ids, role_ids):
    """Rows per (model, role) over all days, and the (day, model, role) cells where the two sums differ."""
    cells = defaultdict(lambda: [0, 0])
    for key, n in oracle.items():
        cells[key][0] += n
    for (d, mi, ki), n in sums.items():
        cells[(d, model_ids[mi], role_ids[ki])][1] += n
    totals, bad = defaultdict(lambda: [0, 0]), []
    for (d, m, k), (o, leaf) in sorted(cells.items()):
        t = totals[(m, k)]
        t[0] += o
        t[1] += leaf
        if o != leaf:
            bad.append({'day': days.iso(d), 'model': m, 'role': k, 'oracle': o, 'leaf_sum': leaf, 'diff': leaf - o})
    rows = [{'model': m, 'role': k, 'oracle': o, 'leaf_sum': leaf, 'diff': leaf - o}
            for (m, k), (o, leaf) in sorted(totals.items())]
    return rows, bad


# ---------------------------------------------------------------- main
def _tracked(it, total, phase, progress):
    """Yields from it, reporting progress(phase, done, total) once the consumer is done with each item."""
    if progress:
        progress(phase, 0, total)
    for i, x in enumerate(it, 1):
        yield x
        if progress:
            progress(phase, i, total)


def run(args, progress=None):
    """Builds the dashboard data. progress(phase, done, total), if given, is called as work advances."""
    t0 = time.time()
    if args.end:
        end = parse_ts(args.end, naive_tz=None)
        if end is None:
            raise SystemExit('--end: not an ISO 8601 time: %r' % args.end)
    else:
        end = datetime.datetime.now()
    # Days are local calendar days, whatever offset --end was written in; the last one holds --end.
    end = end.astimezone()
    end_ts = end.timestamp()
    last = end.date()
    if args.lookback is None:
        start_ts = float('-inf')
    else:
        start_ts = local_midnight(last - datetime.timedelta(days=max(args.lookback, 1) - 1))
    jobs = discover(args.claude_root, args.codex_root, max(start_ts, 0))
    stats = Stats()
    sessions = []
    workers = args.jobs or min(8, os.cpu_count() or 1)
    if workers > 1 and len(jobs) > 4:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(_tracked(ex.map(parse_job, jobs, chunksize=4), len(jobs), 'parse', progress))
    else:
        results = list(_tracked(map(parse_job, jobs), len(jobs), 'parse', progress))
    t_oracle = time.time()
    if workers > 1 and len(jobs) > 4:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            o_results = list(_tracked(ex.map(oracle_file, jobs, chunksize=4), len(jobs), 'check', progress))
    else:
        o_results = list(_tracked(map(oracle_file, jobs), len(jobs), 'check', progress))
    t_oracle = time.time() - t_oracle
    stamps = [e[1] for sess, _ in results if sess for e in sess['events'] if e[0] == 'C' and e[1] is not None]
    stamps += [ts for _, calls, _ in o_results for ts, _, _ in calls.values() if ts is not None]
    first = min([datetime.date.fromtimestamp(ts) for ts in stamps if start_ts <= ts < end_ts]
                + [last - datetime.timedelta(days=MIN_DAYS - 1)])
    windows = Days(first, last, start_ts, end_ts)
    oracle, oracle_stats = run_oracle(jobs, windows, o_results)
    for (src, _p, _s), (sess, st) in zip(jobs, results):
        stats.inc('%s_files' % src)
        for k, v in st.items():
            stats.inc('%s_%s' % (src, k), v)
        if sess and any(e[0] == 'C' for e in sess['events']):
            sessions.append(sess)
    t_parse = time.time() - t0
    cal = calibrate(sessions)
    acc, recon, basis, hacc = {}, {}, defaultdict(set), {}
    counters = defaultdict(int)
    fits = defaultdict(list)
    for s in _tracked(sessions, len(sessions), 'attribute', progress):
        attribute(s, cal, windows, acc, recon, counters, basis, fits, hacc)
    for src, f in fits.items():
        f.sort()
        cal[src]['estimate_fit'] = {'note': '(estimated blocks + prefix) / measured context per call',
                                    'p10': round(f[len(f) // 10], 3), 'p50': round(f[len(f) // 2], 3),
                                    'p90': round(f[9 * len(f) // 10], 3)}
    models_seen = sorted({k[2] for k in acc} | {m for _, m in recon})
    model_ids = [m for m in PRIMARY_MODELS] + [m for m in models_seen if m not in PRIMARY_MODELS]
    role_ids = role_order({k[3] for k in acc} | {k[2] for k in oracle} | {s['role'] for s in sessions})
    tree = build_tree(acc, model_ids, role_ids, basis, args.top_files, getattr(args, 'top_commands', CMD_TOP_N),
                      hacc)
    sums = leaf_sums(tree)
    by_day = defaultdict(lambda: [0, 0])
    for (d, m), n in recon.items():
        by_day[(d, m)][0] += n
    for (d, mi, _), n in sums.items():
        by_day[(d, model_ids[mi])][1] += n
    rec_totals, rec_bad = defaultdict(lambda: [0, 0]), []
    for (d, m), (logged, leaf) in sorted(by_day.items()):
        t = rec_totals[m]
        t[0] += logged
        t[1] += leaf
        if logged != leaf:
            rec_bad.append({'day': windows.iso(d), 'model': m, 'logged_total': logged, 'leaf_sum': leaf,
                            'diff': leaf - logged})
    reconciliation = [{'model': m, 'logged_total': rec_totals[m][0], 'leaf_sum': rec_totals[m][1],
                       'diff': rec_totals[m][1] - rec_totals[m][0]} for m in model_ids if m in rec_totals]
    oracle_rows, oracle_bad = compare_oracle(oracle, sums, windows, model_ids, role_ids)
    unpriced = sorted(m for m in models_seen if m not in PRICING)
    unpriced_calls = {m: counters['unpriced_calls:' + m] for m in unpriced}
    n_unpriced = sum(unpriced_calls.values())
    coverage = 'Costs use official API rates (%s; fetched %s).' % (', '.join(PRICING_SOURCES), PRICING_FETCHED)
    if n_unpriced:
        coverage += ' %d calls on unpriced models (counted as $0): %s' % (n_unpriced, ', '.join(
            '%s (%d)' % (m, n) for m, n in sorted(unpriced_calls.items(), key=lambda kv: -kv[1]) if n))
    sess_by = defaultdict(int)
    for s in sessions:
        sess_by['%s:%s' % (s['src'], s['role'])] += 1
    srcs = [n for n, k in (('Claude', 'claude_calls'), ('Codex', 'codex_calls')) if stats.get(k)]
    gen = datetime.datetime.fromtimestamp(time.time()).astimezone()
    meta = {
        'generated': gen.isoformat(timespec='seconds'),
        'end': end.isoformat(timespec='seconds'),
        'collected_from': None if args.lookback is None else
        datetime.datetime.fromtimestamp(start_ts).astimezone().isoformat(timespec='seconds'),
        'pill': '%s · generated %s' % (' + '.join(srcs) or 'No sources', gen.strftime('%Y-%m-%d %H:%M')),
        'pillTitle': '%d Claude files, %d Codex files, %d sessions, %d calls in range. Context split estimated at %s chars/token (Claude) and %s (Codex); see meta.reconciliation.' % (
            stats.get('claude_files', 0), stats.get('codex_files', 0), len(sessions), counters['calls_in_range'],
            cal['claude']['chars_per_token'], cal['codex']['chars_per_token']),
        'tokenizer': 'none installed; calibrated chars/token per source',
        'calibration': cal,
        'image_tokens_estimate': IMAGE_TOKENS,
        'pricing_per_mtok': PRICING,
        'pricing_sources': list(PRICING_SOURCES),
        'pricing_fetched': PRICING_FETCHED,
        'cache_write_multipliers_anthropic': CACHE_WRITE_MULT,
        'unpriced_models': unpriced,
        'unpriced_calls': unpriced_calls,
        'coverageNote': coverage,
        'counts': dict(sorted(stats.items())),
        'sessions_by_role': dict(sorted(sess_by.items())),
        'sessions': len(sessions),
        'calls_in_range': counters['calls_in_range'],
        'clamp_events': counters['clamp_events'],
        'prefix_scaled_events': counters['prefix_scaled_events'],
        'size_bin_edges': list(HIST_EDGES),
        'size_bin_fallbacks': counters['hist_fallback'],
        'reconciliation': reconciliation,
        'reconciliation_mismatches': rec_bad,
        'reconciled': not rec_bad,
        'oracle': {
            'method': 'Separate minimal re-read of every log: usage fields summed per (day, model, role), '
                      'deduplicated by message.id / response_id (token_count: repeated totals skipped).',
            'policy': 'Same role regex and model aliases as the parser. Duplicate call keys across files '
                      'count once in the oracle but once per file in the parser; a nonzero '
                      'cross_file_duplicate_keys therefore shows up as a mismatch.',
            **oracle_stats,
            'rows': oracle_rows,
            'mismatches': oracle_bad,
            'matched': not oracle_bad,
        },
        'runtime_seconds': {'parse': round(t_parse, 2), 'oracle': round(t_oracle, 2),
                            'total': round(time.time() - t0, 2)},
    }
    data = {
        'meta': meta,
        'days': {'first': first.isoformat(), 'last': last.isoformat(), 'count': len(windows)},
        'models': [{'id': m, 'label': MODEL_LABELS.get(m, m), 'provider': model_provider(m)} for m in model_ids],
        'roles': [{'id': r, 'label': role_label(r)} for r in role_ids],
        'tree': tree,
    }
    return data


def write_outputs(data, out, html, template):
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, separators=(',', ':'), ensure_ascii=False)
    if html:
        with open(template, encoding='utf-8') as fh:
            tpl = fh.read()
        if '/*__DATA__*/null' not in tpl:
            raise SystemExit('template is missing the /*__DATA__*/null placeholder')
        blob = json.dumps(data, separators=(',', ':'), ensure_ascii=False).replace('</', '<\\/')
        with open(html, 'w', encoding='utf-8') as fh:
            fh.write(tpl.replace('/*__DATA__*/null', blob))


def main(argv=None, progress=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=os.path.join(here, '..', 'token-data.json'))
    ap.add_argument('--html', default=None, help='write the dashboard HTML here')
    ap.add_argument('--template', default=os.path.join(here, 'template.html'))
    ap.add_argument('--end', default=None, help='calls at or after this ISO time are left out (default: now)')
    ap.add_argument('--lookback', type=int, default=None,
                    help='read only the last N local calendar days up to --end (default: all history)')
    ap.add_argument('--claude-root', default=CLAUDE_ROOT)
    ap.add_argument('--codex-root', default=CODEX_ROOT)
    ap.add_argument('--jobs', type=int, default=0, help='parser processes (default: min(8, CPUs))')
    ap.add_argument('--top-commands', type=int, default=CMD_TOP_N,
                    help='commands shown per shell tool node; the rest are merged (default %d)' % CMD_TOP_N)
    ap.add_argument('--top-files', type=int, default=FILE_TOP_N,
                    help='files shown per instruction-file node; the rest are merged (default %d)' % FILE_TOP_N)
    ap.add_argument('--allow-mismatch', action='store_true',
                    help='write outputs even when the leaf sums disagree with the oracle')
    args = ap.parse_args(argv)
    data = run(args, progress)
    m = data['meta']
    bad = m['oracle']['mismatches'] + m['reconciliation_mismatches']
    for r in bad:
        print('MISMATCH %s' % json.dumps(r), file=sys.stderr)
    if bad and not args.allow_mismatch:
        print('leaf sums disagree with the logged usage; nothing written (use --allow-mismatch to write anyway)',
              file=sys.stderr)
        return 2
    write_outputs(data, args.out, args.html, args.template)
    print('sessions=%d calls=%d clamps=%d reconciled=%s oracle_matched=%s oracle_rows=%d runtime=%.1fs' % (
        m['sessions'], m['calls_in_range'], m['clamp_events'], m['reconciled'], m['oracle']['matched'],
        len(m['oracle']['rows']), m['runtime_seconds']['total']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
