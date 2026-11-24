# Paper

Paper is a 4B-parameter reasoning model made by knowledge distillation: an open-weight teacher that reasons well writes worked solutions, and a small student is fine-tuned to reason the same way.

| | Model | Size | Licence |
|---|---|---|---|
| Teacher | `deepseek-r1:14b` (DeepSeek-R1-Distill-Qwen-14B) | 14.8B, 9 GB at 4-bit | MIT |
| Student | `Qwen/Qwen3-4B` | 4.02B | Apache-2.0 |

DeepSeek's licence text says the R1 series allows "distillation for training other LLMs". No training text in this project is written by anyone but the teacher model.

**This folder is the pipeline, not the finished model.** The dataset and the model are produced by running it on a GPU. Nothing here has been run on a GPU yet; see [What has and has not been tested](#what-has-and-has-not-been-tested).

## What you need

- An NVIDIA GPU with 16 GB of memory (built for an RTX 4070 Ti SUPER).
- Python 3.11 or newer, and [Ollama](https://ollama.com) to serve the teacher.
- About 30 GB of free disk.
- Time. Building the dataset is the slow part, and with a 14B teacher on one card it takes days, not hours. The teacher is accurate but reasons at length: in a first trial on an RTX 4070 Ti SUPER it produced about 59 tokens a second and more than half its replies ran past 2,000 tokens. The pilot in step 2 measures this on your machine and projects the total.

## Run it

All commands are run from this folder.

### 1. Install

```bash
python -m venv .venv
```

Activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on Linux), then:

```bash
pip install -r requirements.txt
```

Start the teacher so that it answers four requests at once, each with a 4,096-token window. On Windows, stop any running Ollama first (`Get-Process ollama* | Stop-Process -Force`), then in PowerShell:

```bash
$env:OLLAMA_CONTEXT_LENGTH=4096; $env:OLLAMA_NUM_PARALLEL=4; ollama serve
```

On Linux:

```bash
OLLAMA_CONTEXT_LENGTH=4096 OLLAMA_NUM_PARALLEL=4 ollama serve
```

Leave that window open. Once the model is loaded, `ollama ps` should show `100% GPU`; if it shows a CPU share, lower `OLLAMA_NUM_PARALLEL` and `teacher.concurrency` to 2.

In a second terminal:

```bash
ollama pull deepseek-r1:14b
```

### 2. Pilot

```bash
python -m paper_distill.build pilot
```

This sends 60 generated problems to the teacher, has it write and solve a few questions of its own, and prints the share of traces kept, the speed, and a projected total time. If the projection is too long, the options are:

- a faster teacher: `deepseek-r1:8b` (5 GB, same licence);
- a hosted endpoint for an openly licensed model, set with `teacher.base_url`;
- more generated problems and fewer teacher-written ones (`dataset.mix`), since generated problems cost nothing to write;
- a smaller `dataset.target_size`, down to the 10,000 minimum.

If the pilot keeps almost nothing, it prints the most common rejection reason and what to change.

### 3. Build the dataset

```bash
python -m paper_distill.build all
```

This can be stopped at any time and run again; it continues where it left off. It ends with an audit that fails if any record is malformed, duplicated, or below the 10,000 minimum. The result is `data/paper_train.jsonl` and `data/paper_val.jsonl`.

While it runs, this shows keep rates so far by source and difficulty, from a second terminal:

```bash
python -m paper_distill.build stats
```

### 4. Train

Install PyTorch with CUDA first. The command depends on your system; take it from the selector at [pytorch.org](https://pytorch.org/get-started/locally/). Then:

```bash
pip install -r requirements-train.txt
```

```bash
python -m paper_distill.train
```

Close Ollama first so the teacher is not holding GPU memory. If the run is interrupted, add `--resume`. If it runs out of memory, lower `train.tokens_per_batch` in `config.yaml`.

### 5. Evaluate

Measure the untouched base model first, then Paper, then the teacher, all on the same held-out problems:

```bash
python -m paper_distill.evaluate --model base
```

```bash
python -m paper_distill.evaluate
```

```bash
python -m paper_distill.evaluate --strategy vote
```

```bash
python -m paper_distill.evaluate --teacher
```

```bash
python -m paper_distill.evaluate --summary
```

Add `--set gsm8k --n 300` to any of them for the GSM8K test split (needs `pip install datasets`).

### 6. Use it, and run it on modest hardware

```bash
python -m paper_distill.infer "A tank fills in 6 hours and drains in 9. With both open, how long to fill?"
```

To run without this code, merge the adapter and convert to GGUF. At 4-bit the file is about 2.5 GB and runs on a laptop CPU.

```bash
python -m paper_distill.export
```

```bash
git clone https://github.com/ggml-org/llama.cpp
```

```bash
pip install -r llama.cpp/requirements.txt
```

```bash
python llama.cpp/convert_hf_to_gguf.py out/paper-merged --outfile out/paper-bf16.gguf --outtype bf16
```

Then quantise with llama.cpp's `llama-quantize` (from a [llama.cpp release](https://github.com/ggml-org/llama.cpp/releases) or your own build):

```bash
llama-quantize out/paper-bf16.gguf out/paper-Q4_K_M.gguf Q4_K_M
```

```bash
cd out && ollama create paper -f Modelfile
```

```bash
ollama run paper
```

If replies in Ollama show raw tags or do not stop, copy the `TEMPLATE` block from `ollama show --modelfile qwen3:4b` into `out/Modelfile` and create the model again. Majority voting works against the exported model too: `python -m paper_distill.infer "..." --http http://localhost:11434/v1 --name paper`.

## The dataset

One JSON object per line. Every record has exactly these keys, in this order:

| Key | Meaning |
|---|---|
| `id` | 16 hex characters, a hash of the question |
| `source` | `procedural`, `teacher_authored` or `public:gsm8k` |
| `domain`, `skill`, `difficulty` | what the problem tests; difficulty is 1 to 5 |
| `mode` | `think`, or `direct` for an answer with no reasoning |
| `system` | the system prompt, identical in every record |
| `question` | the user turn |
| `reasoning` | the teacher's reasoning |
| `answer` | the teacher's visible reply; its last line is always `Final answer: ...` |
| `final_answer` | the text of that last line |
| `verification` | how the answer was checked, and against what |
| `teacher` | model, licence and temperature |
| `stats` | word counts and token count |

The student is trained to produce `<think>\n{reasoning}\n</think>\n\n{answer}`, which is Qwen3's own format. The loss is taken only on that text, never on the question.

Where the questions come from:

- **Procedural (85%).** 27 generators in `paper_distill/seeds/` write multi-step problems and compute the exact answer in code: arithmetic, algebra, number theory, counting and probability, geometry, deduction puzzles with one solution, family and comparison chains, multi-hop facts about invented entities, navigation, object tracking, tables, code tracing, shortest paths, scheduling and search. A trace is kept only if the teacher's final answer equals the computed one.
- **Teacher-authored (15%).** The teacher writes new questions on 40 topics the generators cannot cover (commonsense, causal, science, verbal, strategy), each with an answer key. A question is kept only if a separate solving pass by the teacher reaches the same answer. This is weaker than exact ground truth: it catches inconsistency, not a mistake the teacher makes twice.
- **Public (off by default).** GSM8K's train split can be mixed in with `dataset.mix.public`.

A trace is also rejected if it was cut off, has almost no reasoning, lacks the final-answer line, mixes languages, loops, has a very long paragraph, or names its own maker.

Reasoning length is capped: 900 words for difficulty 1 to 3 and 1,600 for difficulty 4 and 5, with long traces limited to a quarter of the final set. A reply is stopped the moment its reasoning passes the cap, so the teacher's time is not spent on text that would be discarded. A rejected problem is not retried; a fresh one is used.

The solver keeps accepted traces roughly level across generators, so the set does not drift toward whatever the teacher finds easy, but it gives up on a kind of problem the teacher has not once solved within the limits. 600 generated problems are held out and never trained on.

## Why it is built this way

- **Fine-tune on teacher outputs, nothing fancier.** DeepSeek made its own small models by fine-tuning Qwen and Llama checkpoints on 800,000 R1 outputs, with no reinforcement learning. A 32B model distilled this way beat the same model trained with RL (72.6 against 47.0 on AIME 2024). [R1 paper](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)
- **Keep only checked answers.** DeepSeek sampled several responses per prompt and kept the correct ones, also dropping mixed-language reasoning and long paragraphs (same paper). One large study found answer filtering made little difference for big students ([OpenThoughts](https://arxiv.org/abs/2506.04178)); it is kept here because exact checking costs nothing for generated problems.
- **Short traces.** Students of 3B and under do not reliably benefit from long reasoning chains ([Small Model Learnability Gap](https://arxiv.org/abs/2502.12143)), and 100k pruned traces averaging about 720 tokens beat 800k long ones ([arXiv 2505.19716](https://arxiv.org/abs/2505.19716)). The same work recommends about one long trace to four short ones. Hence the caps, the quarter limit on long traces, and the preference for shorter traces when there is a surplus.
- **More than the minimum.** Small models trained on only about 8,000 long traces lost accuracy in one study, while 7B and 14B models recovered at 16,000 to 32,000 ([arXiv 2506.07712](https://arxiv.org/abs/2506.07712)). The default is 12,000 short traces. Raising `dataset.target_size` is the most useful change you can make if you have the time.
- **A mid-size teacher is fine.** Across many teacher and student pairs, a stronger teacher did not predict a better student ([arXiv 2601.14249](https://arxiv.org/abs/2601.14249)).
- **Voting at inference.** DeepSeek's 1.5B distil went from 28.9 to 52.7 on AIME 2024 with majority voting over 64 samples (R1 paper, table 5). Paper samples three solutions and draws two more only if they disagree; stopping early when chains agree is both faster and more accurate than always voting over everything ([arXiv 2505.17813](https://arxiv.org/abs/2505.17813)).
- **A reasoning budget.** If the model is still thinking when its budget runs out, the think block is closed and it must answer ([budget forcing](https://arxiv.org/abs/2501.19393)).
- **No self-checking pass.** Asking a small model to review its own answer mostly makes it worse ([arXiv 2310.01798](https://arxiv.org/abs/2310.01798)).
- **Training settings.** Three epochs, learning rate 1e-4 with cosine decay to a tenth, in line with what DeepSeek reports for its distils ([arXiv 2501.12948](https://arxiv.org/abs/2501.12948)).

The arXiv items above were read as abstracts and search extracts; the R1 paper and the licences were read in full from the official repositories.

## What to expect

Paper will not match the large reasoning models. Its teacher, the 14B distil, scores 69.7 on AIME 2024 and 93.9 on MATH-500, and a student trained on about 12,000 of its traces should not be expected to exceed it. DeepSeek needed 800,000 samples from its full model to get its 1.5B distil to 83.9 on MATH-500.

There is also a real chance of no gain. `Qwen3-4B` already reasons in think tags out of the box, and a small fine-tuning set can lower a small model's accuracy as well as raise it. That is why step 5 measures the base model first. If Paper does not beat the base on the held-out set, the fix is more data, shorter traces, or a teacher from the student's own family (`qwen3:14b`, Apache-2.0, set `system_in_user: false`).

## What has and has not been tested

Tested in a CPU-only container (123 tests):

- Every generator at every difficulty, with independent re-solving of the ordering puzzles and spot checks of the maths.
- The answer checker, on accepted and rejected cases for every answer type.
- The full build (prepare, author, solve, select, audit) against a scripted stand-in teacher, including resume, the call limit, a teacher outage, and the audit catching broken records.
- The teacher client against a local stub server: request shape, streamed and plain replies, stopping over-long reasoning early, retries and failures.
- The training text against Qwen3-4B's own chat template, using its real tokenizer: they match exactly.
- Train, resume, merge, generate and evaluate on a tiny randomly initialised model with the student's architecture. The loss falls, the merged model computes the same outputs as base plus adapter, and budgeted generation and voting run.

Not tested, because it needs a GPU or model weights that could not be downloaded there:

- The real teacher beyond one 60-problem trial: its keep rate and speed under the current settings, how Ollama streams its reasoning, and whether parallel requests fit in memory.
- 4-bit loading, memory use at these settings, and training time.
- The GGUF conversion and Ollama's handling of the exported model.
- Paper's accuracy. Paper does not exist until you run steps 3 and 4.

```bash
pip install pytest && python -m pytest -q
```

## Licences

- Teacher outputs: DeepSeek-R1 is MIT-licensed and its README permits distillation. Its base, Qwen2.5-14B, is Apache-2.0.
- Student: Qwen3-4B is Apache-2.0. If you redistribute Paper, keep that licence notice with it.
- GSM8K (optional): MIT.
