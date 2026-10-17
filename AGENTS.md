# Working on this experiment

- Keep code readable for a PhD researcher who is new to Python; prefer small named functions over frameworks.
- Do not pass questions, labels, supporting-fact annotations, or reader outputs into a query-blind compressor.
- Never commit API keys, `.env`, `.venv`, raw dataset caches, or raw API response caches.
- Store data, virtual environments, temporary files and results under this repository on D:, not C:.
- Keep `full`, `no_context`, and compressed predictions paired on exactly the same examples.
- Save both directions of InfoGap. Preserve the documented factor-2 convention and the unscaled result.
- Scan all probability breakpoints and one-sided limits; do not substitute a coarse threshold grid.
- Do not replace invalid probability outputs with a default or silently discard failed samples.
- Use a new run name when changing protocol, model, prompts, code or data. Preserve previous runs.
- Run `python -m unittest discover -s tests -v` for changes to metrics, data boundaries, caching or API behavior.
- Real API calls cost money or consume service quota. Keep explicit attempt limits and do not expand sample size without a task-related reason.
- Keep bilingual HTML documentation defaulting to English; source-code comments and README may be Chinese.
