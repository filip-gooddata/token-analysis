# Token dashboard

A local dashboard of Claude Code and Codex token use and API-rate cost over the last 30 days. `token-analysis` reads the session logs on this Mac, shows its progress in the terminal, writes a drill-down stacked-bar chart, and opens it in your default browser. The outputs hold aggregate numbers and labels only, never message text or file contents.

## Quick start

Build the executable once:

```sh
./build.sh
```

The script creates `.venv/`, installs PyInstaller there, and writes `dist/token-analysis`. It needs Python 3 and network access for the install. Set `PYTHON=/path/to/python3` to choose the interpreter.

Run it:

```sh
./dist/token-analysis
```

It prints a progress bar for each phase (reading session logs, cross-checking usage, attributing tokens), then prints the dashboard path and opens it in the browser. A run takes from a few seconds to about a minute, depending on how many logs you have.

To run it as `token-analysis` from any directory, put a link to it on your `PATH`, for example:

```sh
ln -s "$PWD/dist/token-analysis" ~/.local/bin/token-analysis
```

macOS can block the unsigned binary on first launch. If it does, allow it in System Settings > Privacy & Security. The binary is built for the architecture of the Mac that builds it.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--out-dir PATH` | `~/Library/Application Support/token-analysis` | Where `token-data.json` and `token-drilldown.html` are written. Each run replaces them. |
| `--no-browser` | off | Collect and print the dashboard path, but do not open the browser. |

Other flags go to the collector, for example `--lookback 7`, `--top-files 20`, or `--end 2026-10-20T12:00:00+02:00`. Use `python3 -I collector/collect.py --help` for the full list. `--out`, `--html`, and `--template` are not accepted, because `token-analysis` sets them; use `--out-dir` to choose where the files go.

If collection fails, `token-analysis` exits with a nonzero status and does not open the browser.

## Run from source

You can run the same command without building it:

```sh
python3 -I collector/token_analysis.py
```

To run only the collector, with no progress display or browser, run this from the repository root:

```sh
python3 -I collector/collect.py --out token-data.json --html token-drilldown.html
```

## Dashboard

- **Models**: each model with usage has a checkbox with its tokens and cost for the current range, metric and role. **All** and **None** select every model or no model. **Claude** and **GPT** select only that class. Shift-click **Claude** or **GPT** to add or remove that class and keep the rest of the selection. Every number in the chart, the drill-down levels and the legend is computed from the selected models only. With no model selected, the chart shows a message instead of bars. The browser remembers the excluded models in `localStorage` (`tkd.models`).
- **Shell commands**: under **Tool calls** or **Tool results** → **Built-in tools**, the `Bash` and `exec` nodes break down by command, such as `git status`, `cat` or `rg`. Labels come only from fixed allowlists of programs and subcommands, so they never contain arguments, paths or the names of your own scripts. Programs that are not on the list are grouped as **Other commands**. A Codex `exec` script with no shell command shows its first MCP or built-in tool call instead, such as **MCP: t3_code**, **MCP (other)**, `write_stdin` or **Tool discovery**, using fixed server, app and tool lists only. See `collector/README.md` for the rules.
- **Instruction files**: each CLAUDE.md, AGENTS.md or rules file splits by how it entered the context: autoloaded at session start, reloaded after compaction, loaded on path match, or @-imported. When the model reads such a file itself with Read, `cat` or `sed`, those tokens stay under **Tool results**, and the detail panel links the two places. See `collector/README.md` for the rules.

## Tests

Run from the repository root:

```sh
python3 -I collector/test_collect.py
python3 -I collector/test_token_analysis.py
```

## Files

| File | Purpose |
|---|---|
| `collector/collect.py` | Reads session logs and generates aggregate token and cost data. |
| `collector/token_analysis.py` | The `token-analysis` command: progress display, output location, browser launch. |
| `collector/template.html` | HTML template used to build the dashboard. |
| `collector/test_collect.py` | Tests for the collector. |
| `collector/test_token_analysis.py` | Tests for the command. |
| `collector/README.md` | Detailed collector methodology and reference. |
| `build.sh` | Builds `dist/token-analysis` with PyInstaller. |

See [`collector/README.md`](collector/README.md) for pricing sources, pricing tiers, attribution rules, and known limitations.
