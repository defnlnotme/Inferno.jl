## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2025-05-16 - [Dimmed Thinking Indicator Pattern]
**Learning:** For interactive LLM inference, providing immediate feedback during the "prefill" phase (before the first token is sampled) significantly reduces perceived latency. A subtle "thinking" indicator using dimmed dots (`\e[2m...\e[0m`) cleared by backspaces (`\b\b\b\e[K`) when the first token arrives provides a clear signal of activity without cluttering the output. This pattern is safe for TTYs and should be implemented in both library-level streaming wrappers and end-user CLI scripts.
**Action:** Implement `is_tty` checks and use the `\e[2m...\e[0m` -> `\b\b\b\e[K` sequence in all streaming output functions to signal processing.
