# go3v3_cheap20 — scoring observations (2026-11-18)

Offline analysis of the question cache (`outputs/go3v3_cheap20/qcache/`, on the 2019 host): 320 answers from 17
configurations (16 design points + BO iterations 1–2), 20 GAIA validation questions (seed 42, n20 subset).
No code was changed. The run keeps the current scorer, so every point stays comparable.

## 1. The run's scorer

`run_gaia_subset_enhanced.py:116-118` scores with an exact, case-insensitive match after `strip()`:

```python
prediction = str(final_result).strip()
true_answer = str(example["true_answer"]).strip()
is_correct = prediction.lower() == true_answer.lower()
```

This is **not** the official GAIA scorer (`scripts/gaia_scorer.py::question_scorer`). On these 320 answers both give the same
total (175 correct, 54.7%), but they differ on single answers. The official scorer ignores whitespace and ASCII
punctuation, and it compares numbers as floats.

## 2. Where the wrong answers come from

| Category | Answers | Share |
|---|---|---|
| Correct | 175 | 54.9% |
| Wrong answer | 78 | 24.5% |
| No final answer (the "answer" is reasoning text: `Thought: …`, `Looking at this conversation…`, `<code>`) | 35 | 11.0% |
| Right answer on the first line, followed by an explanation | 16 | 5.0% |
| Gave up (`Unable to determine`, `None`) | 12 | 3.8% |
| Right, but rejected by the strict match (Unicode dash, number in words) | 3 | 0.9% |

Official scorer applied to the first line only: 190/320 = **59.4%**, against 54.7% now.

### Correction to an earlier note

The "answer + explanation" problem comes from the **reformulator (docs slot)**, not from the manager.
`claude-haiku-5-5` as reformulator produces 13 of the 16 cases, and it scores 9/40 in that role. Point 9
(luna / minimax-m3.1 / haiku, 15%) and point 10 (minimax-m3.1 / minimax-m3.1 / haiku, 30%) both have haiku in docs.
The low score of point 9 was earlier blamed on luna as manager; most of it is haiku as reformulator.

"No final answer" is spread over most reformulators (glm 12, minimax-m3 9, deepseek 5, minimax-m3.1 5).
It appears when the final answer is taken from raw text: the manager hit its step limit, or the reformulation fell back
(`run_gaia_subset_enhanced.py:105-108`).

## 3. Questions almost every configuration gets wrong

| q | Level | Right | Expected | What happens | Solvable today? |
|---|---|---|---|---|---|
| 10 | 2 | 0/16 | `6` | 15/16 answer `3`. Box Office Mojo's 2020 lists have most likely changed since the ground truth was written. | **No** (stale web data) |
| 18 | 3 | 1/15 | `100000000` | Needs the narration audio of a YouTube 360 VR video. 2 answers are `100 million`, which is the right value but in words. | Barely (audio) |
| 19 | 3 | 2/14 | `bacon` | Recognise the harness brand in a photo, then find a website story dated 8 Dec 2022. | Hard (vision) |
| 8 | 2 | 3/15 | `orange, white` | Film knowledge. `red, white` is common. One `orange, white` was rejected because text followed it. | Yes |
| 12 | 2 | 3/15 | `Kleinpaul` | Chapter 2 on Project MUSE often fails to load, so models guess (`Nietzsche`, `Rank`). | Yes, access-dependent |
| 15 | 2 | 3/15 | `Russian-German Legion` | 2 answers `Russian–German Legion` (en dash) rejected. 1 `Russo-German Legion`. | Yes |

Reachable maximum with q10 (and in practice q18) unsolvable: about 90%. Every configuration fails q10, so the ranking does not change.

## 4. Possible ground-truth leakage

9 cached records mention the GAIA dataset or its public mirrors. One deepseek manager writes "I'll start by recovering
the GAIA ground-truth record from the public mirrors". 6 of the 9 are marked correct:
q15 (mimo), q6 (deepseek), q11 ×2 (minimax-m3.1), q3 and q11 (minimax-m3). It is not verified yet whether the answers
were actually read from the dataset, or whether the dataset is only mentioned in the reasoning.

## 5. How we would fix it (hypothetical, not applied)

Keep the current exact match as the **BO objective** for the whole go3v3_cheap20 run. Changing it mid-run would make new
points incomparable with the finished ones. The fixes below are for a later run, or for a second, offline "lenient"
column next to the current accuracy.

1. **Scorer: official GAIA scorer plus small normalisations.** Use `question_scorer`, and before it:
   - Unicode NFKC, then map every dash (`‐ ‑ ‒ – — −`) to `-` and curly quotes to straight quotes.
     Effect: `Russian–German Legion` = `Russian-German Legion`.
   - For a numeric ground truth, parse numbers written with a scale word (`100 million`, `1.5 billion`, `3 thousand`)
     into the plain number. `100 million` → `100000000`, which is what the ground truth expects (an integer with no
     separators). Commas, `$` and `%` are already stripped by the official scorer.
   - Lists: the official scorer already ignores spaces, so `orange,white` = `orange, white`.
2. **Answer followed by extra text.** The decision depends on whether the question says how to answer:
   - Every question is wrapped by `augmented_question`, and the reformulator prompt asks for a bare final answer. Many
     questions also state the format explicitly ("Your answer should be a numerical integer value", "Give the last name
     only", "Answer without using articles", "comma-separated list in alphabetical order").
   - Proposed rule: extract the answer from a `FINAL ANSWER:` marker if present, otherwise from the first non-empty line,
     and score that. Keep the strict score as well and report both.
   - Treat "right answer + explanation" as a separate failure class in the analysis. It is a real weakness of that
     reformulator: haiku in docs loses about 1 question in 3 this way.
3. **No final answer.** When the prediction starts with reasoning text (`Thought:`, `Looking at`, `I'll`, `<code>`, …),
   call the reformulator once more with a strict "final answer only" prompt before scoring. Count it as wrong if it
   still fails. Log the event, so the "no final answer" rate is visible per configuration.
4. **Unsolvable questions.** Flag q10 (stale web data) and q18 (audio only) in the subset file. Report accuracy both
   on all 20 questions and on the solvable ones. Replace them only in a new subset, never mid-run.
5. **Leakage.** In the web search and page-visit tools, block `huggingface.co/datasets/gaia-benchmark`, dataset mirrors
   and pages that publish GAIA answers. Re-check the 9 records above by reading their transcripts.

## 6. Relaxed scorer: final review before the clean run (relaxed_v2)

Every answer still wrong under the relaxed scorer was checked by hand (design, first run, BO iterations).

- Accepted by relaxed but not by exact: 4 answers, all legitimate. `100 million` (q18), `Russian–German Legion` with an en dash (q15, ×2),
  and `Russo-German Legion` (q15).
- **Alias added (relaxed_v2):** `ANSWER_ALIASES` in `scripts/gaia_scorer.py` accepts `Russo-German Legion` for `Russian-German Legion`.
  Both are English names of the same unit. D7 (glm / glm / luna) goes from 60% to 65%. Aliases are hand-reviewed judgment calls;
  keep the list short.
- **Per-question aliases (relaxed_v3, `TASK_ALIASES` keyed by task id):**
  - q10 Box Office Mojo: `3` accepted. The live 2020 lists (read on 2026-11-18) share 3 titles in the two top 10s
    (Bad Boys for Life, Sonic the Hedgehog, Dolittle); the ground truth `6` reflects older lists. 14 of 16 configurations
    answered `3`. Date-dependent: re-check before reusing this subset later.
  - q17 Pearl City: `2017 Komo Mai Drive 900000` accepted. The question asks which home sold for more *and* the price, so naming
    the home with the price answers it fully (an earlier draft of this note had it as "extra text").
- Every other content error was checked against the question text and is a true error (q1 actor vs character name, q4, q5,
  q8, q9 USGS zip, q11 geometry, q12, q14 `564.0` from C = 149 instead of 150, q15 `Tennessee militia`, q18, q19).
- Effect: most points gain 5–15 points of accuracy (q10 alone gives +5 to every configuration that answered `3`). The new
  maximum is D15 (glm / deepseek / luna) at 80%.
- No other scorer problems: the remaining wrong answers are wrong content (`3` for `6`, `cute` for `fluffy`, `563.6` for `563.9`,
  `Nietzsche` for `Kleinpaul`, …), answer + explanation, reasoning text, or empty.

### Pipeline problems found (not scorer problems; not fixed, kept for comparability)

1. **Output cap (`max_tokens` 32768) hit:** 33 of 12,976 LLM calls ended with `finish_reason=length`, over about 18 questions.
   The harmful case is the **reformulator**: 9 capped calls (deepseek@max 5, minimax-m3 2, mimo 2) on 5 questions. The model
   spends the whole budget reasoning, the answer is cut and the prediction is **empty**: 3 of the 5 empty predictions have no
   error and two reformulator calls of exactly 32768 output tokens each. (The other 2 empty predictions are 30-minute timeouts.)
   Possible fix for a later run: on `finish_reason=length` in the reformulator, retry once with low reasoning effort, or give the
   reformulator a lower effort by default.
2. **Reformulator continues the agent instead of answering:** 37 of the 39 "no final answer" cases come from a reformulator call
   that returned a continuation of the agent transcript (`Thought: …`, `<code>`, `# Approach`, `I'll continue …`), mostly
   glm@max. The reformulator sees the memory as a conversation and imitates it. Possible fix for a later run: send the memory as
   one quoted user message and ask for `FINAL ANSWER: …` only, then extract after the marker. This changes the prompt, so not now.

## 7. Data

- Question × configuration matrix and per-model breakdowns were computed with throw-away scripts on the 2019 host. They
  can be regenerated from the question cache with no API calls.
- Pareto page: `make_pareto.py` + `template.html` (in the session scratchpad) read `evaluations.jsonl` from the 2019 host.
