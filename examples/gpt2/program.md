# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the current branch (HEAD), not master.
3. **Read the in-scope files**: The repo is for Triton-XDNA compiler. We are targetting iGPU and NPU. The @examples/gpt2 files are all fair game, as are the files in @amd_triton_npu
4. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a RyzenAI mini pc containing an iGPU and NPU. The gpt2 inference script runs for a **fixed number of 10 generated tokens** using the `hetero` backend:

```bash
cd examples/gpt2
AMD_TRITON_NPU_FUSED_MLP=1 python gpt2_inference.py --backend hetero --max-tokens 10 > run.log 2>&1
```

`--max-tokens 10` is required — the default (0) does a single forward pass, not generation. `AMD_TRITON_NPU_FUSED_MLP=1` enables the load_pdi fused-MLP path — without it, NPU/hetero runs measure an unoptimized fallback (~70x slower prefill), so it must be set on every run. Keep the backend, token count, and this flag fixed across experiments so results are comparable.

**What you CAN do:**
- Modify `examples/gpt2` files — Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.
- Modify `amd_triton_npu/backend/driver.py` file - this is our main compilation driver for NPU. Triton-XDNA is an under development compiler and we want to improve its functionality and performance. Smart and concise changes are allowed to improve the quality of the compiled NPU code.  

**What you CANNOT do:**
- Install new packages or add dependencies. 
- Modify the evaluation harness. 

**Known dead end — do NOT retry:** hetero decode is slow (~8.5s/token) because XRT re-instantiates the hardware context on every NPU launch. Persisting/reusing `hw_context` across launches in `driver.py` was already tried and REVERTED (commit 0ca58b6) — it corrupts other kernels (reads a dirty AIE array, ~75% wrong elements) and breaks the example suite. This was re-investigated and confirmed non-viable on this firmware/XRT stack. Do not re-add context persistence as a quick fix. If you change `driver.py`, run `python scripts/run_tests.py --device aie2p` to confirm you didn't break the suite.

**The goal is simple: get the best heterogeneous (iGPU + NPU) performance for gpt2.** Everything is fair game: optimize the model, fuse operations, change data types, optimize the triton-xdna compiler passes. The only requirement is that we preserve accuracy - the generated tokens should make sense.

**Optimization metric**: minimize **TPOT** (time per output token, ms) as the primary metric, with **total generation time** as the tiebreaker. Lower is better. Energy/power is out of scope for this loop (the script does not emit it; measuring it needs the sudo power-analyzer MCP).

**Accuracy gate**: in generation mode the run prints two blocks — `--- Triton (HETERO) Generation ---` and `--- HuggingFace Reference Generation ---`. Compare the two generated texts: they should match (or be very close and still coherent). A change that speeds things up but makes the Triton output diverge into gibberish is a failure — discard it.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A few ms improvement that adds 20 lines of hacky code? Probably not worth it. A few ms improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the inference script as is.

## Output format

Once the script finishes it prints a `Performance Metrics` block like this:

```
  Performance Metrics:
    Prompt tokens:  4
    Output tokens:  10
    TTFT:           123.4 ms
    TPOT:           56.7 ms
    TPS:            8.9 tokens/sec
    Total time:     0.634 s
    Decode min:     50.1 ms
    Decode max:     70.2 ms
    Steady TPOT:    52.3 ms  (excl. 1st decode)
    Steady TPS:     9.5 tokens/sec
```

Grep `run.log` for `TPOT:` (primary metric) and `Total time:` (tiebreaker). Also check the `LOGITS COMPARISON` block (`Cosine similarity`, `Top-1 match`) for the accuracy gate.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV should have a header row and maintain relevant results organized by git commit hash.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune the example with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment (redirect everything to run.log — do NOT use tee or let output flood your context):
   `cd examples/gpt2 && AMD_TRITON_NPU_FUSED_MLP=1 python gpt2_inference.py --backend hetero --max-tokens 10 > run.log 2>&1`
5. Read out the results: `grep -E "TPOT:|Total time:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If performance or energy efficiency improved (lower), you "advance" the branch, keeping the git commit
9. If perf or energy efficiency is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!