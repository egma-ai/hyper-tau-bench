---
name: search-corpus
description: Search every file in the task materials with the hyper_tau search_corpus tool. Use when you need to find which documents, emails, transcripts, images or spreadsheets bear on a topic, rule, policy, exception or edge case.
---

# Searching the task materials with search_corpus

`search_corpus` (on the hyper_tau MCP server) asks one yes/no question of
every file in the task materials and returns, for each file, the probability
that the answer is yes. A fast decision model reads every file, so one call
covers the whole corpus, including files whose wording would not match a
keyword search.

## Calling it

- `question` (required): a yes/no question about a single file, for example
  "Does this file state a rule about refunds for cancelled flights?" or
  "Does this file describe what to do when a customer cannot verify their
  identity?"
- `min_probability` (optional, default 0.5): the cut-off for the files listed
  in the reply. Lower it (for example to 0.2) to also see borderline files.
- `path` (optional): only search one directory, relative to the workspace
  root.

The reply lists the matching files, highest probability first; long files
appear as line ranges. The probability for every file is saved in
`corpus_search/<timestamp>.json`. Audio and video files are not read; they
are listed as skipped.

## Using the results

- The tool tells you where to look, not what the files say: open the files
  it lists and read them.
- Ask about one topic per call. Narrow questions give sharper probabilities
  than broad ones.
- Probabilities can be wrong in both directions. When a topic matters, ask
  again in different words or lower `min_probability`.
- A call scans the whole corpus and takes from a few seconds to about a
  minute. The limit is 200 calls per task.
