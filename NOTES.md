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
