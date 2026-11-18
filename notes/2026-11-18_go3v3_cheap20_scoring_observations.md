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

## 6. Data

- Question × configuration matrix and per-model breakdowns were computed with throw-away scripts on the 2019 host. They
  can be regenerated from the question cache with no API calls.
- Pareto page: `make_pareto.py` + `template.html` (in the session scratchpad) read `evaluations.jsonl` from the 2019 host.
