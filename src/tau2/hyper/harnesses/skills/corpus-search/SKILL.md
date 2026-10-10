---
name: corpus-search
description: Ask a yes/no question of every file in the task materials at once, using the hyper_tau search_corpus tool.
---

The hyper_tau search_corpus tool allows you to ask a yes/no question of every file in the task materials at once. A fast & cheap AI model reads each file and returns the probability that the answer is yes for that file. Text files, PDFs, Word/Excel/PowerPoint files, emails and HTML are read as text, images are read as images, and files inside .zip archives are included. Long files are judged in sections and reported with line ranges. Returns the files at or above min_probability, highest first; the result for every file is saved to corpus_search/<timestamp>.json. Audio and video files are not read and are listed as skipped.

**Parameters:**

| Parameter | Required | Description |
|---|---|---|
| `question` | yes | The yes/no question to ask of each file, e.g. 'Does this file state a rule about refunds for cancelled flights?' |
| `min_probability` | no, default 0.5 | Only list files at or above this probability. |
| `path` | no | Only search files under this workspace-relative directory. |
