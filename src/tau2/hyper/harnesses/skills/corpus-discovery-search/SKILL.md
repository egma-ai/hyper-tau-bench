---
name: corpus-discovery
description: A skill teaching how to create a good support agent by discovering the scenarios, procedures and cross-cutting policies a support agent will need to do its job, using the hyper_tau search_corpus tool.
---

The core concepts are Scenarios, Procedures & Policies. To build a good agent - here's what you should be aiming to do:

1. Discover possible scenarios from the corpus that might be encountered in production by the support agent
2. Discover the right procedures for each possible scenario.
3. Discover cross-cutting policies that might apply for any support conversation - things that the support agent should be aware of regardless of the query.

How the search tool might help - it enables you to do proper discovery of scenarios, procedures and policies across the whole corpus (except audio & video & the Client).

It allows you to ask a yes/no question of every file in the task materials at once. A fast & cheap AI model reads each file and returns the probability that the answer is yes for that file. Text files, PDFs, Word/Excel/PowerPoint files, emails and HTML are read as text, images are read as images, and files inside .zip archives are included. Long files are judged in sections and reported with line ranges. Returns the files at or above min_probability, highest first; the result for every file is saved to corpus_search/<timestamp>.json. Audio and video files are not read and are listed as skipped.

**Parameters:**

| Parameter | Required | Description |
|---|---|---|
| `question` | yes | The yes/no question to ask of each file, e.g. 'Does this file state a rule about refunds for cancelled flights?' |
| `min_probability` | no, default 0.5 | Only list files at or above this probability. |
| `path` | no | Only search files under this workspace-relative directory. |
