# WSBench Phase 1

## Running the workbench

From the source directory, install the CPU wheel and WSBench with Python 3.11 or newer.

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.4'
python -m pip install -e .
wsbench fetch all
wsbench serve
```

`wsbench fetch transfo-xl-wt103` adds the 1.1 GB Transformer-XL checkpoint; `all` leaves it out. Open `http://127.0.0.1:8765/` on the same computer. The tiny models work on a fresh offline installation. The default data directory is `~/.cache/wsbench`. Set `WSB_HOME` to choose another directory.

```sh
wsbench check tests/data/tiny_battery.wsl
wsbench run tests/data/tiny_battery.wsl --json /tmp/tiny-verdicts.json
pytest -q
```

Set `WSB_LLM=mock` for the offline assist demonstration. A live session reads `OPENROUTER_API_KEY` from its environment. Run the optional acceptance suites with `pytest -m network`, `pytest -m slow`, and `pytest -m live`. To repeat the three live checks on GitHub, manually dispatch the **OpenRouter live tests** workflow with the `OPENROUTER_API_KEY` Actions repository secret configured.

## Implementation decisions

- The editor opens with a short runnable example for the bundled tiny model.
- The workbench displays implicit statement names as `line<N>` in the API and UI.
- Each synchronous progress callback receives a snapshot that remains valid as subsequent statements run.
- Numeric overflow produces a diagnostic for the affected statement so a run remains JSON-serializable.
- The two-column grid catalogue supplies the descriptions for the Language tab and leaves its arguments column empty.
- The executor retains changed tensors for the five most recently submitted scripts, counting queued and cancelled scripts, and for at most 2 GiB in all (`jobs.STATE_BYTES`, counting each kept tensor once per line, also one shared with the baseline): the states of the oldest runs are dropped first, then the earliest lines of the running script. A dropped state answers 410 `E_EXPIRED`.
- The assist uses OpenRouter's reported cost when available and the dossier prices for known models. A custom model without either source of pricing contributes zero to the estimate.
- Code Golf fixture subsets use the upstream Apache-2.0 licence in `tests/fixtures/golf_mini/LICENSE`.
- NeoX uses one torch thread for the tiny model and two for Pythia, Transformer-XL one for `txl-tiny` and up to four for `transfo-xl-wt103`; the caller's thread count is restored afterward.

## Run history

- Every script run is stored as a JSON file in `WSB_HOME/history/` (the 50 newest are kept): script, verdicts, status and source (`editor`, `arc` or `assist`). A running script is saved at most every two seconds and when it ends.
- `GET /api/runs` lists the 20 newest runs. `GET /api/runs/{run_id}` also answers for runs of an earlier server process, and adds `script`, `source` and `created` to the response.
- After a page reload the Editor shows its most recent run again, including one that is still running, and the Assist tab shows its most recent measured script.
- A run that was in progress when the server stopped is shown as `cancelled`; its unfinished statements get `E_RUNTIME` with the message `interrupted: the server restarted before this statement finished`.
- The Assist tab starts a request with `POST /api/assist/start` and reads its state from `GET /api/assist/{assist_id}` once a second, so no HTTP request stays open while the models answer. Each request is stored in `WSB_HOME/assist_history/` with the model replies (the 20 newest are kept), and `GET /api/assist/latest` returns the newest one, which the tab shows again after a reload. `POST /api/assist` still answers synchronously.
- When the translator finds no testable claim, the Assist tab opens the translator's reply and reports `no testable statement`.
- Tensor views need the edited weights, which are kept in memory only. For a run of an earlier server process they answer 410 `E_EXPIRED`.

## Explore loop

- The Explore tab runs several assist rounds for one idea. A planner model (default: the translate model) writes one testable question per round, sees a digest of the measured verdicts (statements and numbers, no model prose) and writes the next question. A final call writes a short report in the language of the idea.
- The loop stops when the planner sets `done`, after the requested rounds (1–8, default 3), before a round that would probably exceed the budget (default $1, at most $5; the estimate is the most expensive round so far), or on cancel. The first round always runs, and the report call comes on top of the budget.
- The planner may ask the researcher at most two questions per loop. The loop then has status `waiting` until `POST /api/explore/{id}/answer` arrives.
- If the planner model fails or refuses, the critic model is tried for that call.
- Endpoints: `POST /api/explore/start`, `GET /api/explore/{id}`, `GET /api/explore/latest`, `POST /api/explore/{id}/answer`, `POST /api/explore/{id}/cancel`. State is stored in `WSB_HOME/explore_history/` (the 20 newest are kept) and restored after a reload. A loop that was running when the server stopped is reported as interrupted and does not resume.
- Live check on pythia-14m (two rounds, a German idea about interchangeable middle layers): 22 statements measured in 94 s for $0.35; the planner model accepted the measured results as input both times.

## Model providers

- With `ANTHROPIC_API_KEY` set, every model ID that starts with `anthropic/` is called through the Anthropic API with the official `anthropic` package: the Explore planner and report, and the Assist translator. The ID is mapped to the Anthropic form (`anthropic/claude-fable-5.2` → `claude-fable-5-2`). All other models, and Claude models when the key is missing, go through OpenRouter.
- The Anthropic call sends `output_config.effort = low`, no `thinking` and no sampling parameters, and opts into the server-side refusal fallback (`fallbacks="default"`, beta `server-side-fallback-2026-07-01`). When a fallback model answered, the reply carries `served_by` and the cost uses that model's price.
- Costs for Anthropic calls are computed from the token usage and a price table (Fable 5.2: $8 / $40 per 1M tokens); prompt caching is not used, so the estimate is an upper bound.
- `/api/health` reports `anthropic: true|false`, the header shows it as `Claude direct`, and each reply carries `provider`.
- Not verified live: this path was tested with a stand-in client only. No Anthropic key was available in the development environment.

## Research agent (Claude Platform)

- The Agent tab talks to a Claude Managed Agent (beta). Anthropic hosts the agent loop; WSBench executes the agent's tool calls locally, so every measurement is the same deterministic executor the Editor uses. This is the high-agency layer: the agent decides what to test, writes WSL itself and converses over several turns. The Explore tab is the fixed loop in between, and the Editor is fully manual.
- Agent definition (`wsbench/agent/spec.py`): model `claude-fable-5-2` at effort `medium`, a system prompt with the language card, and four custom tools: `wsl_check`, `wsl_run`, `arc_tasks`, `arc_task`. No sandbox tools and no web access.
- The environment and the agent are created once and their IDs are stored in `WSB_HOME/agent.json`; the agent is updated in place when the definition changes. Each conversation is one session with a hard budget (default $2, at most $20), set as a session budget in cents.
- A turn opens the event stream before it sends the message, answers `agent.custom_tool_use` events with `user.custom_tool_result`, ends on an idle event whose stop reason is not `requires_action`, and after a dropped stream reads the missed events from the event list. Stop sends `user.interrupt` and cancels a running script.
- Transcripts are stored in `WSB_HOME/agent_history/` and shown again after a reload. After a server restart a conversation is shown as `closed`: the session still exists on the Claude Platform, but this server cannot continue it.
- A fifth tool, `arxiv_paper`, lets the agent read a paper from arXiv: part 0 gives title, authors, abstract and section headings, parts 1..n give the full text from arXiv's HTML version in pieces of 15,000 characters. The server contacts only arXiv, only for a well-formed arXiv identifier, and caches the text in `WSB_HOME/papers/`. Papers without an HTML version yield the abstract only.
- `lesswrong_post` reads a LessWrong or Alignment Forum post through the forum's GraphQL API (cached in `WSB_HOME/posts/`). `github_repo` clones a public repository once (shallow, into `WSB_HOME/repos/`) and then lists folders, reads files in parts and searches text. Repository code is only read, never executed or imported; paths outside the checkout and the `.git` folder are refused. These three tools are the agent's only access to outside material.
- First real run (2026-11-25, from the owner's Codespace): one turn with `wsl_check` and `wsl_run`, 52 s, $0.19 list cost.
- A conversation survives the server. The session lives on the Claude Platform, so after a server restart or a failed turn the next message first re-attaches: it reads the session's whole event history (all pages), shows what the agent said meanwhile, and settles tool calls nobody answered (a new message is refused while a call is pending). Only a terminated session or a spent budget ends a conversation.
- A tool result counts as delivered only after the platform accepted it. If sending fails, the stored result is sent again after the reconnect, and the tool is not run twice. Sending retries connection errors and overload (408, 409, 429, 5xx) with a growing pause. `retries_exhausted` and `refusal` end the turn but keep the session usable; a refusal shows its category.
- Stop sends `user.interrupt`. If a tool is running at that moment, its result is sent when it finishes and the agent is interrupted again; a result that the platform refuses because the interrupt closed the call is dropped.
- The IDs of the environment and the agent are stored in `WSB_HOME/agent.json` together with a fingerprint of the API key. Another key means another workspace, so the IDs are ignored. They are written after each create, an existing resource of the same name is reused before a new one is created, and a deleted or archived resource is replaced once. Mock mode never touches this file.
- The link to the session in the Claude Console uses the workspace named in `WSB_ANTHROPIC_WORKSPACE` (default `default`).
- With `WSB_LLM=mock` a scripted stand-in platform answers, for offline use.
- Not verified live: no Anthropic key was available in the development environment. Tested with stand-in platforms only (event handling, tool execution, budget, errors, storage) and in Chromium in mock mode.

## Transformer-XL

The owner asked for Transformer-XL with 551M parameters. No public Transformer-XL checkpoint has that size, so WSBench uses the public Transformer-XL large model on WikiText-103 and reads its geometry from `config.json`. Another Transformer-XL checkpoint needs a snapshot entry in `fetch.py` and a `SHAPES` entry in `registry.py`, not new code. Configurations with `div_val` 1, pre-LayerNorm or shared position biases are refused.

### Models and provenance

- `transfo-xl-wt103` is the Hugging Face checkpoint `transfo-xl/transfo-xl-wt103` at revision `40a186da79458c9f9de846edfaea79c412137f97`: 18 layers, d 1024, 16 heads of 64, MLP width 4096, a vocabulary of 267,735 words, and an adaptive embedding and softmax with four clusters (ids below 20,000, 40,000 and 200,000, widths 1024, 256, 64, 16). `txl-tiny` has random weights (4 layers, 2 heads, d 16, vocabulary 64 in four clusters) and works offline. Both belong to the family `txl` (`wsbench/models/txl.py`).
- Pinned files and SHA-256: `config.json` `5ccc2e5ae36a26fa0edccfc9fd16aba633bde30aa3cadca17e2394480b823ada`; `pytorch_model.bin` `edd5e62f5edb8ef65703ef827ab0bcd16ab29f2b853cfd9d11d81faa1bc7a0fb` (1,140,884,800 bytes, the legacy torch format); `vocab.json` `24e5bbda51446a21610bd58bb1ebdefee85766d0b3715f35da09babead595243` (267,735 words; `<eos>` 0, `<unk>` 24).
- `wsbench fetch transfo-xl-wt103` downloads the three files into `WSB_HOME/models/transfo-xl-wt103/`. Each file goes to a temporary file first and is renamed after its hash check. `wsbench fetch all` still means pythia-14m and ARC. The pickled files of the repository (`vocab.pkl`, `vocab.bin`, `corpus.bin`) are never fetched, and the weights are read with `torch.load(weights_only=True)` after the hash check. `/api/health` reports the checkpoint under `models_local`, and the header shows it.
- Size: 249 tensors with 285,205,322 parameters, each stored once. Counting the seven tied tensors twice gives 322,473,146: the four softmax weights are the embedding tables, and three of the four softmax projections are the embedding projections. The paper gives 257M for this model.
- Licence: the Hugging Face model card states no licence. The original code, kimiyoung/transformer-xl, is Apache-2.0, and WikiText-103 is CC BY-SA. The Language tab lists this under Licences and attribution.

### Function and language

- A model state holds each tensor once. The softmax weights and the tied projections are stored under their `transformer.word_emb.*` names, so a block edit of `transformer.word_emb.emb_layers.{c}.weight` changes the embedding and the softmax together. The duplicate names (`crit.out_layers.{c}.weight` and the tied `crit.out_projs.{c}`) are rejected with `E_TARGET`, and the message names the tensor to use. Operators are copy-on-write: they share every tensor they leave alone.
- WSBench measures one segment without memory, with the plain causal mask, as on the first segment of the original evaluation. Hugging Face's TransfoXL called without `mems` prepends 1600 all-zero memory slots to every layer instead. They act as an attention sink and lower the probe loss from 4.7249 to 4.1357 nats. `txl.forward(..., zero_mems=1600)` reproduces that call.
- The probe is the Pythia probe's prose in WikiText-103 tokenisation: 122 tokens, all in the vocabulary. The bench compares the log-probabilities of the full adaptive softmax where it compares logits for NeoX. `max_abs_dlogits` and `rel` are therefore log-probability differences; the largest |log-probability| of the probe is 43.7167.
- The positional frequencies are the checkpoint's float32 table (`transformer.pos_emb.inv_freq`), within 8.2e-8 relative of the float64 formula. The float64 formula would move the probe log-probabilities by up to 6.4e-6.
- Agreement with transformers 4.46.3 in float64: `txl-tiny` agrees within 1.1e-14 in every log-probability, without memory, with zero memories, for a batch of two, with `same_length` off and with other clamp and window lengths. On the probe the largest log-probability difference is 3.9e-12. The loss is 4.724939573022554 against 4.724939573022569, and 4.135668769365871 against 4.135668769365872 with zero memories. The references are in `tests/data/txl_tiny_reference.json` and `tests/data/txl_wt103_probe.json`.
- WSL: `model txl-tiny` and `model transfo-xl-wt103` take the claims gauge, reparam, approx and edit, and the layer operations of NeoX: `permute`, `scale(L[i].mlp, uniform(lo, hi))`, `rotate(L[i].mlp, orth)`, `vo`, `qk` (only on all dims, the default), `delete`, `swap`, `L[i] := ...` with `lerp`, `neg_writers`, `neg` and `align`, and `gridop` on any 2-D parameter. The language card has an "OPS ON Transformer-XL" section and FACTS lines for `transfo-xl-wt103`. The Editor palette, the Assist and Explore target menus and the agent accept the family. `wsbench/examples/transformer_xl.wsl` is the source of the card's numbers.

### What is a symmetry in a post-LayerNorm Transformer-XL

- Exact: permuting MLP neurons; positive per-neuron rescaling, because ReLU(c z) = c ReLU(z) for c > 0 (it is not exact through GELU in NeoX); permuting heads, which moves each head's q, k and v rows, r_net rows, both position biases and o_net columns together; a change of basis N of one head's values (v → N v, o_net → o_net N⁻¹); and a change of basis A of one head's queries and keys on all its dimensions. The Q/K case maps q, r_w_bias and r_r_bias to A·, and k and the head's r_net rows to A⁻ᵀ·. This keeps both the content score (q + r_w_bias)·k and the position score (q + r_r_bias)·r. Aligning a layer to another is a composition of these permutations. On `transfo-xl-wt103` they change the log-probabilities by 6.5e-16 to 1.3e-15 relative in L9, and by up to 9.1e-14 in the other layers measured.
- Not exact: rotating MLP neurons, because ReLU acts per neuron. `neg(L)` is not exact either. In attention every score is a product of two negated factors, and the negated o_net cancels the negated values, so the attention output does not change. The MLP becomes -W₃ ReLU(-z) - b₃, which is not W₃ ReLU(z) + b₃. On the probe, `L9 := neg(L9)` costs +4.73 nats.
- Not defined (`E_MODEL`, with the reason in the message): `fold_ln`, `center_writers`, `ln_to_rms`, `rotate(resid, ...)` and `scale(resid, ...)`. Every sublayer ends in x ← LN(x + f(x)), so the residual stream itself is the output of a LayerNorm. Its gain and bias feed the next layer and the skip connection alike, so no reader can absorb them, and a rotation or scaling of the stream does not pass through the next LayerNorm. `rope_pairs` and `qk(..., dims=pass)` are not defined either: Transformer-XL has relative sinusoidal positions with learned biases and no rotary dimensions.
- `delete(L[i])` zeroes what the layer adds to the residual stream (o_net, CoreNet.3 and its bias). The two post-LayerNorms stay, so a deleted layer maps x to LN2(LN1(x)), not to x.

### Measurements on transfo-xl-wt103

All values are from `wsbench/examples/transformer_xl.wsl` (seed 0) through the executor, in nats on the 122-token probe. The base loss is 4.7249, and L[mid] is L9.

- Exact symmetries (CONFIRMED): MLP permutation and rescaling, head permutation, V/O and Q/K basis changes and align, with relative changes from 7.3e-16 to 1.2e-15.
- Not symmetries (REFUTED): rotating the L9 MLP (+4.72) and `L9 := neg(L9)` (+4.73).
- Edits: deleting L9 gives -0.05 (control +0.75) and deleting L10 alone -0.01. Deleting L8 and L10 together costs +0.70. Swapping L8 and L9 costs +3.53, `L10 := neg_writers(L8)` +4.58, `L9 := lerp(L8, L10, 0.5)` +0.94 and rotating the L9 MLP +4.73. A rot90 of one 64 × 64 block of L9's o_net gives -0.005.
- `pytest -m network tests/test_txl.py` repeats these measurements and checks the card's numbers against them.

### Executor and cost

- The control of an edit puts random noise of the edit's norm on exactly the entries the edit changed, as for NeoX, but it draws the noise differently. The NeoX control draws one generator sequence for every tensor in key order, unchanged tensors included, so the sequence stays stable. On `transfo-xl-wt103` that would be 285M normal draws per edit. The Transformer-XL control draws only for the changed tensors. Each tensor gets its own generator, seeded with ((seed + 999,983) · 2³² + CRC-32 of the tensor name) mod 2⁶⁴, so the noise of a tensor depends only on the statement's seed and the tensor's name. Transformer-XL and NeoX controls are therefore not draw-for-draw comparable.
- The change report checks identity (`a is b`) before it compares values. With copy-on-write an operation replaces 1 to 26 of the 249 tensors, and only those are compared.
- On four CPU threads `transfo-xl-wt103` loads in 4 to 8 s, of which the hash checks take 1.3 s. A forward pass takes 0.9 to 1.2 s. A gauge statement takes 0.6 to 1.5 s (one forward pass), and an edit takes 1.5 to 3 s (the edit and its control). `rotate(L[i].mlp, orth)` adds 2 to 4 s for the 4096 × 4096 orthogonal matrix, and `align` adds 2.5 to 3.5 s for the assignment of 4096 neurons. The 16-statement example takes 31 s after a 4.2 s load. Times varied by a factor of two on the shared development machine.
- Memory: the float64 state needs 2.12 GiB, and the baseline log-probabilities [1, 122, 267,735] another 261 MB. The process peaks at 3.0 GiB after loading and at 3.7 GiB while running the example.
- Each line keeps its changed tensors for the tensor views. On `transfo-xl-wt103` that is 8 to 208 MiB per line: 8 MiB for a gridop on an o_net block, 32 MiB for a V/O or Q/K basis change, 40 MiB for a delete, 64 MiB for an MLP permutation or rotation, 104 MiB for a lerp or neg of a layer, 156 MiB for a gridop on the first embedding table and 208 MiB for a swap. A swap keeps the baseline's own tensors and allocates nothing new. The 16 statements of the example keep 1.13 GiB, of which 0.86 GiB are new tensors; a 20-statement measurement kept 1.35 GiB of new tensors. That is why the kept states now have a 2 GiB budget (see Implementation decisions).
- A tensor view of the 20,000 × 1024 embedding table takes up to 1 s and sends about 700 KiB.

## Acceptance measurements

- The pinned Pythia download matched all three SHA-256 values. The 76-tensor model yielded 149 probe tokens, loss 4.3176623 and maximum absolute logit 25.84748.
- All 28 Pythia statements passed the specified numeric and verdict checks in 13.32 s with training paused. Running independent training and browser experiments concurrently increased the time to 55.99 s.
- The tiny battery reproduced all 19 statuses and classes, including the 7.1558e-6 relative logit change and the −0.3300 layer-interpolation loss delta.

### Modular-addition experiment

The four-thread CPU run used the unmodified float32 trainer. The validation process used passive OpenMP waits and allocator reuse. The table records the first evaluation that reached each threshold.

| Run | Train ≥99% | Test ≥50% | Test ≥90% | Test ≥99% | Wall seconds |
|---|---:|---:|---:|---:|---:|
| A, seed 0 | 150 | 500 | 1,250 | 1,450 | 173.08 |
| F, seed 0 with Grokfast | 200 | 200 | 1,000 | 1,200 | 145.15 |
| B, seed 1 | 150 | 250 | 650 | 850 | 142.75 |
| S, seed 0 with intervention | 150 | 250 | 750 | 850 | 129.15 |

- Grokfast used 20% fewer steps to reach 90% test accuracy. The observed speed-up was 1,250 / 1,000 = 1.25×.
- On P1, test accuracy was 0.4895 at t=0 and 1.0000 at t=1. Training accuracy dipped to 0.7851 at t=0.25.
- Transplanting A's direction into B on P2 gave test accuracy 0.01743 at t=1. C1 measured a cosine of −0.001801 between the two seeds' directions.
- The HTTP cancellation check observed a running train statement through two evaluations, stopped it within two seconds, and verified that it wrote no run.
- Each run stores effective training arguments in `run.json` and publishes its completed checkpoint set by directory rename.
- The complete D.3 script passed all numerical acceptance checks in 593.13 seconds.

### ARC and Code Golf

- The pinned archive supplied 400 task files, 400 raw solutions and 400 submitted solutions.
- All seven single-primitive tasks matched the Appendix F counts for primitive, fixed and gated execution.
- The sequential 400-task Code Golf test confirmed 397/400 in 652.45 seconds. Tasks 157, 205 and 286 returned `error: timeout` at the 60-second limit.
- Some ARC-GEN inputs exceed the compiler's 30×30 canvas. The task browser and primitive and golf evaluators preserve those inputs. Compiled evaluation records an out-of-canvas input as a failed example.

### Browser checklist

| Check | Observed in Chromium |
|---|---|
| Load and run the four examples | D.1, D.2 and D.4 completed. D.3 produced live charts and was cancelled for the UI check. The slow suite runs the full D.3 experiment separately. |
| Inspect R11 | Opened the three heatmaps and switched the selected 16×16 block to three ARC decile grids. |
| Insert a block edit | Inserted a rotation at the editor cursor and obtained a MEASURED verdict. |
| Open task150 and compare networks | Golf passed 266/266. The fixed mirror network passed 39/266. The gated network passed 266/266 and offered seven input shapes. |
| Open task087 in the editor | The selected ARC program opened with its task directive and solve statement. |
| Inspect training and path charts | The D.3 accuracy and log-loss SVGs rendered. Verified a three-point path chart, then the full D.3 P1 chart on the measured A@200 and A@2000 checkpoints. |
| Use Assist | Verified the missing-key response and the mock repair, raw replies, five predicted/measured rows, and combined-script editor action. |
| Narrow display | At 390 px the document fit the viewport. No JavaScript errors occurred during these flows. |

### Assist and final checks

- The mocked three-role pipeline returned all five expected labels, verdicts and agreement values. Its syntax-repair case made one repair call.
- On 2026-11-25, all three default models returned a live completion through OpenRouter in 35.90 seconds. The test reported USD 0.046960 for `anthropic/claude-fable-5.2`, USD 0.000000 for `openai/gpt-6.2-sol` and USD 0.024789 for `google/gemini-4-neon`, totaling USD 0.071749. [GitHub Actions run](https://github.com/Erikiss/WSB/actions/runs/45612629646)
- The complete offline suite passed 97 tests. The three Appendix A–C kernel files matched the supplied source byte for byte. A wheel installation check loaded the CLI, bundled WSL examples and static application.
- The Cloud preview on port 3000 uses mock Assist. The verified modular-addition checkpoints are available in this Cloud computer's default cache; the source archive can regenerate them by running the D.3 example.
