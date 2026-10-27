# Token drill-down collector

`collect.py` reads local Claude Code and Codex session logs and writes aggregate token and cost data for the drill-down dashboard. It uses only the Python standard library and makes no network or LLM calls. It never writes to the log directories.

## Run

From the repository root:

```sh
python3 -I collector/collect.py --out token-data.json --html token-drilldown.html
```

| Option | Default | Meaning |
|---|---|---|
| `--end ISO` | now | End of the collected data (exclusive). Its local day is the last day on the timeline. |
| `--lookback N` | all history | Read only the last N local calendar days up to `--end` |
| `--claude-root`, `--codex-root` | `~/.claude/projects`, `~/.codex/sessions` | Log roots |
| `--template` | `collector/template.html` | HTML template. The data is embedded with `</` escaped. |
| `--jobs N` | CPU count, max 8 | Parser processes |
| `--top-files N` | 12 | Files shown per instruction-file group. The rest fold into "Other (k files)". |
| `--top-commands N` | 12 | Commands shown under each shell tool. The rest fold into "Other (k commands)". |
| `--allow-mismatch` | off | Write the outputs even when a reconciliation or oracle check fails |

The exit code is 2 when a check fails. The disagreeing rows are printed to stderr, and nothing is written unless you pass `--allow-mismatch`.

Run the tests with `python3 -I collector/test_collect.py`. The `-m unittest` form does not work under `-I`.

## Sources

- Claude: `~/.claude/projects/**/*.jsonl`, subagent files included. Each API call is deduplicated by `message.id`, falling back to `requestId`, then `uuid`, and the last usage line wins. A `requestId` that appears on only some lines of a message does not split it.
- Codex: `~/.codex/sessions/**/*.jsonl`. `token_usage_record` lines are deduplicated by response id. A file with no `token_usage_record` lines falls back to `token_count` events. Each event contributes the difference between its cumulative `total_token_usage` and the previous one. A repeated total is skipped. If the total drops, which means a reset, the event's `last_token_usage` is used.
- Role: taken only from the logs. No role configuration file or T3 setup is read or needed; without profile tags every session is `top` or `native`. The role comes from the first subagent-profile tag in any user message (Claude: non-meta user text), so a tag in a later message still counts. If there is no tag, a Claude subagent file is `native` and anything else is `top`. A role value is a lowercase slug (`[a-z][a-z0-9-]{0,31}`); a tag with any other value is not a profile tag and stays prompt text. `roles` lists the roles found in the parsed sessions: `top`, `envoy`, `drone`, `review` and `native` in that order, then every other role, sorted, labelled from its slug (`envoy-peer` shows as **Envoy peer**). Role ids and these labels are the one output text not drawn from a fixed set. The slug pattern bounds them to 32 lowercase letters, digits and hyphens.

## Accounting

For each call, the measured context is C = input + cache read + cache write. Measured output is reported separately. Per model, the leaf sums equal the logged totals exactly, and so do the sums for every day. `meta.reconciliation` holds this check for every model, and `meta.reconciliation_mismatches` lists every disagreeing (day, model) pair.

`meta.oracle` is an independent check. A second, minimal pass re-reads every log and sums the usage fields per (day, model, role). It has its own timestamp parsing and dedup, and it does not use the replay or the attribution. Each row compares the oracle with the leaf sums. Duplicate call keys across files count once in the oracle but once per file in the parser, so a nonzero `cross_file_duplicate_keys` would show up as a mismatch.

Timestamps keep their offsets: `Z` and `±HH:MM` are honoured, and stamps without an offset are read as UTC. Each call is counted on its local calendar day, in the collector's time zone, so a day around a DST change has 23 or 25 hours. Data runs up to, but not including, `--end`.

### Timeline

`days` describes the timeline: `first` and `last` local ISO dates and `count` days. `last` is the `--end` day. `first` is the earliest day with data, but at least 30 days before `last` (`MIN_DAYS` = 31 days in all), so the presets always fit. With `--lookback N`, only calls from local midnight of day `last − (N − 1)` on are read, and `meta.collected_from` holds that start; otherwise it is `null`. Calls older than the logs themselves (Claude Code and Codex delete old sessions) are not available, so the left end of the timeline is the coverage limit, not a limit you set.

Each leaf has `v`, a sorted list of sparse rows `[day, model, role, billed, unique, usd]`. `day` indexes from `days.first`, `model` indexes `models` and `role` indexes `roles`. Rows are left out only when all three serialized values and their histogram occurrence counts are 0. `h` is parallel to `v`: for each row, the list of non-empty size bins `[bin, occurrences, billed, unique, usd]`. Unique tokens and dollars are rounded along the days of each leaf and (model, role) pair, so the rows of any run of consecutive days add up to that run's own total within one rounding unit (1 token, $0.0001), and the rows of all days add up to the rounded total. Leaves split from one larger entry, such as shell commands and files, share that entry's rounded rows: each stays within two units of its raw running total, so a run of days is off by less than three units, and a day with no negative value never gets a negative row. A row with no calls of its own, which such rounding can create, takes its bins from the latest earlier day with calls for the same model and role, with 0 occurrences and 0 billed tokens. The dashboard sums the rows inside the selected days, so no period is precomputed. The output has no `ranges` key and no `today`/`7d`/`30d` cells; a template from an older version cannot read it.

The JSON grows with the number of active days. On a synthetic fixture with activity every day (5 roles, 5 models, about 15 commands), it was:

| History | Before (3 ranges) | Day rows |
|---|---|---|
| 30 days | 49 KB (12 KB gzip) | 48 KB (14 KB gzip) |
| 90 days | 62 KB (17 KB gzip) | 139 KB (41 KB gzip) |
| 400 days | 79 KB (25 KB gzip) | 625 KB (180 KB gzip) |

Use `--lookback` to cap the size. Moving a handle on the 400-day file took under 3 ms per input event in Chrome.

The collector replays each session in order:

1. Every content block goes into a running context with an estimated token count.
2. Every call bills the whole context again, so a re-read counts on every call.
3. A compaction resets the context. A new user turn drops the previous thinking, because Claude sends only the current turn's thinking.
4. The prefix (system prompt sections plus tool schemas) comes from the latest `prompt_snapshot` (Claude) or the base instructions (Codex). A snapshot without a `tools` list keeps the previous tool schemas.
5. Each call's C is split as follows. Cache reads are charged to the oldest tokens first: prefix, then carried blocks, then this turn's new blocks.
   - If estimates + prefix ≤ C, the remainder goes to **Unattributed prefix**.
   - If estimates ≤ C < estimates + prefix, the prefix is scaled down to fit. This is counted in `prefix_scaled_events`.
   - If C < estimates, all estimates are scaled down and the prefix gets 0. This is the clamp, counted in `clamp_events`.
6. Integer totals are apportioned by largest remainder.

Each leaf's tooltip shows its basis strings:

| Basis | Meaning |
|---|---|
| `measured` | Logged directly: total output, Codex reasoning tokens |
| `estimated (chars/token)` | Text length ÷ calibrated ratio |
| `estimated (fixed per image)` | 1,600 tokens per image |
| `estimated split of measured output` | Claude output split between thinking, text, and tool calls by character share |
| `residual (measured context − estimates)` | Unattributed prefix |

### Instruction files

Each group under **Agent instructions → Instruction files** (Rules files, Project AGENTS.md, User CLAUDE.md, and the rest) and the Codex **AGENTS.md instructions** node has one child per file. A group with a single file keeps that file as its one child, so the filename stays visible. Codex sends AGENTS.md as a user message that starts with `# AGENTS.md instructions for <dir>` and ends at `</INSTRUCTIONS>`; that span goes to the AGENTS.md node, split into `~/.codex/AGENTS.md` and the project file at `--- project-doc ---`, and the rest of the message is classified as before. A developer message is treated the same way only when it starts with that wrapper. Developer text that merely mentions AGENTS.md, as Codex's own instructions do, stays under **Developer messages**. File provenance is inferred from the wrapper alone, so a pasted copy of the exact wrapper is counted as a file.

- The group is first allocated as one path, exactly as before. Its integer share of each call is then apportioned among its files by their token weights. The file spans stay contiguous inside the group's span, so cache-read and cache-write pricing also adds up. Billed tokens of the files sum exactly to the old group leaf in every (day, model, role) cell. Unique tokens and dollars are rounded per leaf, so their sums can differ by rounding.
- File sizes come from the content logged when the file was injected, not from disk. A file edited during the window is counted at the size of each version when it was sent, so no extra estimate is needed. The basis stays `estimated (chars/token)`.
- Codex sends `~/.codex/AGENTS.md`, then `--- project-doc ---`, then the project docs. The text before the separator goes to `~/.codex/AGENTS.md`, the rest to the project file named in the header.
- Labels: a file in a git checkout shows as `<repo>/<path in repo>`. A worktree takes its main repository's name from its `.git` file, and an in-repo `.worktrees/<name>/` or `worktrees/<name>/` directory is dropped, so one file aggregates across worktrees. A deleted `~/.t3/worktrees/<project>/<name>/` worktree takes the repository name of a live sibling, else `<project>`. A relative `gitdir:` is resolved against the checkout, and paths with spaces are kept whole. Other files under a home directory show as `~/<path>`. So does a plain git repository in a hidden home directory, such as `~/.codex/memories`, but a worktree there keeps its repo label. Anything else shows as `…/<parent>/<name>`. Home paths of any user, Windows drive and UNC paths included, and `-Users-<name>` project slugs are replaced with `~`. A Windows path outside a home directory shows as `…/<parent>/<name>`.

#### Attribution rules

Each file node has one child per way the file entered the context. The children split the file's share by token weight, the same way the group is split among files, so a file's total does not change.

| Child | Rule |
|---|---|
| **Autoloaded at session start** | A Claude `instructions` attachment whose `reason` is `session_start`, or whose `reason` is absent before any compaction, and every Codex AGENTS.md wrapper before a compaction. A resumed session logs `session_start` again partway through the file, and that reload counts here too. A file that the root CLAUDE.md pulls in with `@path` is part of that same attachment, so it also counts here. |
| **Reloaded after compaction** | A Claude `instructions` attachment with `reason: compaction`. Older logs have no `reason`; for them, an attachment after a `compact_boundary` counts here. Also a Codex AGENTS.md wrapper after a `compacted` record. |
| **Loaded on path match** | A Claude `nested_memory` attachment with no `parent` field. Claude Code loads it when the session touches a file under the rule's directory or `paths:` glob. |
| **@-imported** | A Claude `nested_memory` attachment that has a `parent` field. This is inferred: the logs have no explicit import marker, and the field names the file that pulled it in. |

A file the model reads itself is not an instruction-file load. Its tokens stay under **Tool results**, where the model paid for them, and are never counted twice. The tool nodes split those results as follows:

- **Read**: **Instruction files**, one child per file, and **Other files**. A path counts as an instruction file when its name is `CLAUDE.md`, `CLAUDE.local.md` or `AGENTS.md`, when it is a `.md` file under `.claude/rules/`, `.agents/rules/` or `~/.claude/projects/<slug>/memory/`, or when it is under `~/.codex/memories/`. Relative paths are resolved against the call's working directory, and symlinks are resolved. A working directory that is not a string leaves relative paths unattributed.
- **Bash, exec and the other shell tools**: under each command (`cat`, `sed`, …), **Instruction files** with one child per file, or **Several instruction files** when one command reads more than one or names them with a glob or brace pattern (`cat .claude/rules/*.md`), and **Other output** for the rest. A command counts only when it does nothing but read: one of `cat`, `head`, `tail`, `sed`, `nl`, `less`, `more` or `bat`, with only instruction files as operands. Input from `<` counts only when it is one plain `<` redirect and the command's only input (`cat < AGENTS.md`); `<&`, `<>`, a numbered `n<`, a second `<`, or `<` next to file operands makes the command not count.
  - Only options on that program's `READ_OPTS` list are allowed, so `sed -i`, `tail -f` and unknown options do not count. A `sed` script counts only when it is made of line-number addresses with `p`, `d`, `q`, `n` or `=` (`1,80p`, `5q`). Any other script, including `s///` and regex addresses, does not count, even when it would only print.
  - Output may go to `/dev/null` or to another descriptor (`2>&1`), but not to a file.
  - The command may be preceded by `cd` and environment assignments. Its output may be piped only into read programs that name no file and take no input redirect of their own (`| head -n 5`). `| tee`, `|&`, `| cat README.md` and `| cat <&3` do not count.
  - `rg`, `grep`, `&&` chains and mixed operands do not count.
  - A Codex `exec` script counts only when the grammar under [Shell commands](#shell-commands) proves that its first call is `exec_command`, no other `exec_command` follows it, and the code after the call ends outside any string, template or comment. The call's `cmd` must be a string built only from literals and prefix bindings, and a `workdir`, if present, too. A runtime value in `cmd` (`"…" + name`) means the command does not count, and a runtime value anywhere else makes the script **Unparsed**. The command is then resolved against that directory. A shell argv such as `["bash", "-lc", "…"]` in the other shell tools uses the script as written.

The explicit-read leaves carry `xf` (the file label) in the JSON, and file nodes carry `fl`. The detail panel uses the pair to cross-link the two places (see [Detail panel](#detail-panel)). Explicit reads use the resolved path, while loads use the path Claude Code logged, so a symlinked rules file can get two different labels; those two places are then not linked.

### Shell commands

The built-in shell tools (Claude `Bash`, Codex `exec` and `exec_command`, and `shell`, `shell_command` and `local_shell`) have one child per command, under both **Tool calls** and **Tool results**. Each tool call carries a command label, and the tool's share is apportioned among its labels the same way as instruction files. So billed tokens of the commands sum exactly to the tool, per (day, model, role) cell.

- The label is the program's basename plus, for multi-command tools such as `git`, `gh`, `pnpm`, `docker` and `kubectl`, the first bare word after the options: `git status`, `pnpm test`. For `python -m` the module is the subcommand, and `python -c` shows as `python3 -c`.
- A command line is split on `;`, `&&`, `||`, `|`, `&` and newlines. The first command that is not setup labels the line. Setup commands are `cd`, `export`, `source`, `set`, `test` and similar, and variable assignments. Wrappers such as `sudo`, `env`, `timeout`, `nohup`, `time` and `xargs` are skipped, and `bash -c "…"` uses the inner command. Each wrapper's options come from `WRAPPER_OPT_WITH_ARG` and `WRAPPER_OPT_FLAGS`, so an option's value is never taken as the program: `exec -a git python -V` shows as `python` and `xargs -a git echo hi` as `echo`. A wrapper option on neither list makes the command **Unparsed**, because its value could be the program. Here-document bodies, `NAME=( … )` array elements, `case` patterns and `(( … ))` arithmetic, including the header of `for (( … ))`, are skipped, so a loop is labelled by its body. A call to a shell function defined on the same line takes the label of the function body.
- Codex `exec` runs a JavaScript script. The collector does not interpret it. It labels the script only when an allowlist grammar proves which tool call the script makes first and with which argument. A missed label is acceptable, a wrong label or attribution is not, so any script outside the grammar gets no command label:
  - **Unparsed** when the script may still reach a tool: it mentions `tools`, `exec_command` or a `\u` escape, or a name that could build a reference to `tools` (the forbidden names below).
  - **Tool discovery** when it does not, but reads `ALL_TOOLS`.
  - **No shell command** otherwise.

  The grammar reads the script from the start up to the closing `)` of the call:
  - A `#!` hashbang at offset 0 is a comment up to the first LF, CR, U+2028 or U+2029. A `#!` after a byte-order mark is outside the grammar. Node's `.mjs` loader accepts it, but a VM script or module and the `.cjs` loader reject it, so the script gets no label.
  - Tokens are JavaScript whitespace, line ends and comments, ASCII names, the punctuators `{ } [ ] ( ) , : ; . = + -`, strings, templates whose only interpolations are `${name}`, and decimal and hex numbers up to 2^53-1. Any other character, a backslash outside a literal, a non-ASCII name character, `==`, `++`, `--`, `+=` or `-=` is outside the grammar. `=>` is a token only where a `.map` callback allows it. So is any of these names as a token: `eval`, `Function`, `constructor`, `globalThis`, `global`, `self`, `window`, `this`, `with`, `Proxy`, `Reflect`, `defineProperty`, `defineProperties`, `assign`, `setPrototypeOf`, `__proto__`, `prototype`, `import`, `require`, `arguments` and `export`. A string or template escape that strict code rejects (legacy octal, `\8`, a malformed `\x` or `\u`) is also outside it.
  - Before the call, only empty statements and `const`/`let` declarations are allowed. Each initializer is a literal: a string, a template, a number, `true`, `false`, `null`, or an array or object of those with plain keys. Strings may be joined with `+`, and an initializer may read an earlier binding (`const d = c + " -la"`, `` `${c}` ``). A `var` declaration, any other statement, a call or an operator before the call is outside the grammar. A declaration ends at `;`, or at a line end when the next token is `const`, `let`, `var`, `await`, `tools`, `text` or `Promise`. A line end before `(` or `[` never ends it, because JavaScript would continue the statement there.
  - The call is `tools.NAME(ARG)`, as an expression statement or as the initializer of a `const`/`let`/`var` declaration. The declaration may bind a name or a flat `[a, b]` or `{a, k: b}` pattern of unique plain names. The call may be reached through `await`, a `(` group, an array or object literal, `text(…)`, the argument of `Promise.all`, `allSettled`, `race` or `any`, or the callback of `NAME.map(…)`. None of these runs anything before the call. Array or object entries before the one that holds the call must be strings, numbers or names of prefix bindings (`KEY: value` with a plain or quoted key, or a shorthand prefix binding), so evaluating them runs nothing.
  - `NAME.map(callback)` needs `NAME` to be a prefix binding that holds a non-empty array, and only one `.map` may lead to the call. `Array.prototype.map` calls the callback for element 0 first, and changing that needs `prototype`, `defineProperty`, `Proxy`, `Reflect`, `setPrototypeOf` or `__proto__`, all outside the grammar. The callback is an arrow function, `async` or not (`async` with no line end after it). Its parameters are a name, or in parentheses a name or a flat `[a, b]` or `{a, k: b}` pattern of plain names, then at most an index name. They bind to element 0 and index 0, so a pattern needs an array element long enough or an object element with each key. Parameter names must be unique, and no line end may come before `=>`. `await` in a callback that is not `async` is outside the grammar. The body is an expression that holds the call, or a block whose first statement holds it: a `const`, `let` or `var` declaration, a `return`, or an expression statement. When the array has more than one element, the call runs once per element: the script takes the label of element 0, but gives no instruction-file attribution. Bindings may not use keywords, the forbidden names, `undefined`, `NaN`, `Infinity`, `tools`, `text` or `Promise`.
  - The argument follows the initializer rules, with plain keys other than `__proto__` and `tools`, each at most once. JavaScript evaluates the value a repeated key drops, so a repeated key is **Unparsed**. A name or a shorthand key may read a prefix binding. A name that no prefix binding holds is a value only the runtime knows, and reading it throws a `ReferenceError` before the call when it is not defined. Such a name may appear only in the string of `exec_command`'s top-level `cmd` (`{cmd: "git status " + value}`), where the label names the command. Anywhere else, including `workdir`, other keys, nested objects, arrays and other tools' arguments, it makes the script **Unparsed**.
  - A runtime value in `cmd` reads as `$X`. A runtime value can hold any shell text, even an unmatched quote, so the label of the literal words around it is not proof of what runs. The command is labelled only in the approved form `"git status " + value`, which gives `git status`. A `` `git status ${value}` `` template builds the same string and is labelled the same. Every other `cmd` that holds a runtime value is **Unparsed**, for example `"cat -n " + file`, `` `cat -n ${file}` ``, `"git " + value`, `"git status" + value` and `` `git status -- ${f}` ``. The value in the approved form can still break the shell quoting or name another command after `git status`, as approved. A runtime value never gives an instruction-file attribution.

  Code after the call is not tokenized. Hoisting could still change `tools` or a name the call reads, or make the script fail before the call, so the call gives **Unparsed** when the raw text after it, strings and comments included:
  - names `tools` other than in a `tools.NAME(` call, or holds a `\u` escape, a forbidden name or a `#`;
  - names a prefix binding that holds an object or array, except after a `.`. The array that `.map` reads may be named later unless the argument holds one of its objects or arrays, which a later write could change before the call reads it;
  - may declare a prefix binding, a result binding, a runtime name the argument reads, or a `text` or `Promise` wrapper the call uses. A duplicate declaration is an early error, so nothing runs, and a later `let`, `const` or `class` of a name the call reads throws before the call. The check returns true for a name after `let`, `const`, `var`, `class`, `function` or `using`, for any later `var` once one of the names appears, for a name in a destructuring pattern of such a declaration, and for code it cannot follow (a `/` that is not a comment, an unclosed string or bracket). A callback's parameters and the names its body declares are checked only up to the end of the callback, where a declaration of them would be an early error, a TDZ throw or a hoisted function. The check runs on the whole rest when that end cannot be found.

  When the call is `exec_command`, its `cmd` labels the script: a string, or an argv array of strings. Any other `cmd` is **Unparsed**. When the call is another tool, the script takes that tool's label, but it is **Unparsed** if `exec_command` appears anywhere else in it, because a later command could take precedence. `mcp__<server>__…` calls show as **MCP: &lt;server&gt;** for the servers in `MCP_SERVERS`, and `mcp__codex_apps__<app>_…` as **MCP: &lt;app&gt;** for the apps in `CODEX_APPS`. Every other MCP call is **MCP (other)**. The built-ins in `EXEC_TOOL_LABELS` (`write_stdin`, `apply_patch`, `view_image`, `update_plan`) show under their own names, and the prefixes in `EXEC_TOOL_PREFIXES` as `web` and `clock`. An unknown tool name gives no label, or **Unparsed** when the code after it may still reach a tool. Function names and unlisted server or app names never appear in a label.

  Limitations. The collector assumes the script parses. A syntax error after the call, such as `x = )`, means nothing runs, yet the label stands. A script that uses a `for … of` loop, calls a helper or computes a value before its first tool call is **Unparsed**, even when a person could read the command from it. The r13–r17 tests of the older lexer still run, but now only assert that no case gets a wrong label: many of those cases are **Unparsed** under the grammar.
- Billed tokens are apportioned as integers. Unique tokens and dollars are rounded by largest remainder, so the serialized children add up to what the unsplit tool leaf would show.
- **Unparsed** and **No shell command** are always shown. The top `--top-commands` labels are kept and the rest fold into "Other (k commands)". The folded child sorts by its total.

Privacy: labels come only from fixed lists in `collect.py`, never from the command text. The program must be on `PROGRAMS` (common Unix, developer and macOS tools) or be `python`, `python3` or `python3.N`. Any other program becomes **Other commands**, which always folds: your own scripts (`./run.sh`), aliases, functions that are not defined on the same line, and quoted or URL-like words. The subcommand must be on that program's `SUBCOMMANDS` list (for example `git status`, `make test`), a `python -m` module must be on `PY_MODULES`, and `npx`, `pnpx`, `bunx` and `uvx` show the tool they run only when it is on `PROGRAMS`. Anything else is dropped, so `make my-target` shows as `make` and `python3 -m my_module` as `python3 -m`. Known options that take an argument (`-C`, `--config`, `-n` and others in `OPT_WITH_ARG`) are skipped with their argument. Known options without an argument (`--no-pager`, `--silent` and others in `OPT_FLAGS`) are skipped alone. After an option on neither list, the next word may be its argument, so the subcommand is taken from the word after it: `docker --tlsdir certs ps` shows as `docker ps`. If that word is missing or not on the list, or both words are on the list, the subcommand is unclear and none is shown, so `docker --tlscacert login` shows as `docker`. To get a label for a tool, add it to these lists. The tests check that arguments, paths, URLs, quoted text, hashes, user-named scripts, targets and modules, and a fixture secret never reach a label.

### Detail panel

Clicking a segment or a legend item selects it, and the panel to the right of the bars shows its details. A segment with children also expands into a new bar. Clicking an expanded segment collapses it and the panel shows its parent. The panel starts on **All tokens**. It stays in view while the page scrolls, and below 860 px it moves under the bars. Every number follows the current period, role, metric and model filter.

- The header shows the value, its share of the parent and of the total, the occurrence count, and the bin of the median occurrence.
- The donut splits the value by model. Above six models, the smallest fold into "Other (k models)".
- On an instruction file, or any node below it, a line reports how much the model also read the file itself ("Also read explicitly: N tokens / k occurrences (counted under Tool results)"). Its button opens that Read or shell leaf. An explicit-read leaf shows the reverse line, with a button back to the file. The line follows the metric, period, role and model filters, and is hidden when the other side is 0.
- The histogram sorts occurrences into fixed size bins (`meta.size_bin_edges`: below 100, 100–200, 200–500, … 1M and above, in estimated tokens). The top panel counts occurrences per bin. The bottom panel shows each bin's share of the selected metric.

An occurrence is one item entering the context:

- A content block (tool call, tool result, message, image) counts once, at the first call after it arrives. If a compaction resets the context before any call, it has no occurrence.
- Model output counts once, at its own call.
- A prefix section (a system prompt section or the tool schemas) counts once per snapshot, at the first call after the snapshot.
- **Unattributed prefix** counts once per call that has a residual.

The bin is the item's own size: its estimate before any scaling for blocks and prefix sections, its share of the measured output for output, and the residual for Unattributed prefix. Every later call that re-reads the item adds its billed tokens to that same bin but adds no occurrence. So a 400-token `cat` result read on three calls is one occurrence with 1,200 billed tokens in the 200–500 bin. A bin can therefore have tokens but no occurrences in a short period, when the item first arrived in an earlier period. Unique tokens and dollars follow the same bins.

Per leaf and (day, model, role) cell, the bins in `h` add up to the cell: billed tokens exactly, and unique tokens and dollars after the same largest-remainder rounding as the cell. A non-leaf node sums its leaves. `meta.size_bin_fallbacks` counts call spans whose tokens could not be traced to an item and were binned by the span's own size. The bins increase the JSON size.

### Calibration

The chars/token ratio is fitted separately per source. For consecutive calls with no compaction, prefix change, or turn change between them: ratio = appended chars ÷ (ΔC − previous output − image tokens). The fit uses the median over pairs with at least 2,000 chars, bounded to [1.5, 6], with 3.5 as the fallback. `meta.calibration[src].estimate_fit` gives p10/p50/p90 of (estimates + prefix) ÷ C.

## Pricing

`PRICING` in `collect.py` holds the official API rates in USD per million tokens, from <https://platform.claude.com/docs/en/about-claude/pricing> and <https://developers.openai.com/api/docs/pricing>, fetched 2026-10-20. `meta.pricing_sources`, `meta.pricing_fetched` and `meta.coverageNote` repeat the sources and the date.

- Anthropic cache writes cost 1.25× the input rate (5-minute TTL) or 2× (1-hour TTL). Cache reads have their own rate per model.
- Some models have a second, `long` rate set. Haiku 5.5 uses it above 100,000 prompt tokens. GPT-6.1 Sol, GPT-6 Luna and GPT-6 Astra use it above 272,000 input tokens.
- The tier is chosen per call from that call's C (input + cache read + cache write). Each call is priced on its own. For Codex, `input_tokens` already includes the cached tokens, so C is `input_tokens`. Calls read from `token_count` events use the per-call change in the cumulative total. The tier uses that change too.
- Model IDs lose a trailing `[…]` tag (for example `[1m]`) and a `-YYYYMMDD` snapshot date before the alias lookup. For example, `claude-haiku-4-5-20251001` is priced as Haiku 4.5.
- The oracle compares token counts, not dollars. Pricing does not change it.

## Known gaps

- Character-based estimates can exceed the measured context, so the prefix is scaled down. Tool schemas and system text may tokenize differently from tool results, and some context (such as Codex tool schemas) is not in the logs.
- Codex tool schemas and some harness context are not logged, so they show up as Unattributed prefix.
- These Claude attachment types have no rendered text and are counted, not attributed: deferred tool records, hook results, credential notices, permissions, and similar (`counts.claude_unrendered_attachments`).
- Unknown models get their own labels and cost $0. They are listed in `meta.unpriced_models`, with call counts for all collected days in `meta.unpriced_calls`. The dashboard shows `meta.coverageNote` under the legend. The note always cites the pricing sources. When there are unpriced calls, it adds a sentence such as "12 calls on unpriced models (counted as $0): …".
- A Codex file's final `token_count` total can differ from the sum of its `token_usage_record` lines. The collector uses the records. Usage that appears only in `token_count` is not counted for that file.

## Privacy

The output contains only numbers and labels: category names, tool names, MCP server names, instruction-source names, and shell command labels (see [Shell commands](#shell-commands)). It contains no message text or file contents. Instruction-file labels are relative paths (`~/…`, `<repo>/…`, `…/<parent>/<name>`) and never contain a home directory or user name. The end-to-end tests check that a fixture secret, the fixture paths, the home directory, its project slug, `/Users/` and `/home/` do not appear in the JSON or the HTML.
