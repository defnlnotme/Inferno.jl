## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2025-05-16 - [LLM Thinking Indicator Pattern]
**Learning:** LLM inference latency (prefill time) can create a "dead" feeling in CLI tools. A subtle "thinking" indicator (dimmed dots `...`) that appears immediately after submission and is cleared by backspaces (`\b\b\b\e[K`) when the first token arrives provides excellent feedback without cluttering the final output. Using `finally` blocks ensures this indicator is always cleared, even on interrupts.
**Action:** Implement the dimmed dot thinking pattern in all streaming inference paths, ensuring TTY checks are used to protect non-interactive environments.
