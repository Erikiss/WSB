# WSBench Phase 1

## Running the workbench

From the source directory, install the CPU wheel and WSBench with Python 3.11 or newer.

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.4'
python -m pip install -e .
wsbench fetch all
wsbench serve
```

Open `http://127.0.0.1:8765/` on the same computer. The tiny model works on a fresh offline installation. The default data directory is `~/.cache/wsbench`. Set `WSB_HOME` to choose another directory.

```sh
wsbench check tests/data/tiny_battery.wsl
wsbench run tests/data/tiny_battery.wsl --json /tmp/tiny-verdicts.json
pytest -q
```

Set `WSB_LLM=mock` for the offline assist demonstration. A live session reads `OPENROUTER_API_KEY` from its environment. Run the optional acceptance suites with `pytest -m network`, `pytest -m slow`, and `pytest -m live`.

## Implementation decisions

- The editor opens with a short runnable example for the bundled tiny model.
- The workbench displays implicit statement names as `line<N>` in the API and UI.
- Each synchronous progress callback receives a snapshot that remains valid as subsequent statements run.
- Numeric overflow produces a diagnostic for the affected statement so a run remains JSON-serializable.
- The two-column grid catalogue supplies the descriptions for the Language tab and leaves its arguments column empty.
- The executor retains changed tensors for the five most recently submitted scripts, counting queued and cancelled scripts.
- The assist uses OpenRouter's reported cost when available and the dossier prices for known models. A custom model without either source of pricing contributes zero to the estimate.
- Code Golf fixture subsets use the upstream Apache-2.0 licence in `tests/fixtures/golf_mini/LICENSE`.
- NeoX uses one torch thread for the tiny model and two for Pythia and restores the caller's thread count afterward.

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
- All three optional live OpenRouter checks skipped because the environment had no `OPENROUTER_API_KEY`. No live calls or charges occurred.
- The complete offline suite passed 97 tests. The three Appendix A–C kernel files matched the supplied source byte for byte. A wheel installation check loaded the CLI, bundled WSL examples and static application.
- The Cloud preview on port 3000 uses mock Assist. The verified modular-addition checkpoints are available in this Cloud computer's default cache; the source archive can regenerate them by running the D.3 example.
