## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2025-05-16 - [Persistent Thinking Indicators for CLI]
**Learning:** Dimmed dots (`\e[2m...\e[0m`) followed by backspaces (`\b\b\b\e[K`) provide an elegant "thinking" state for CLI-based LLMs, bridging the gap between prompt submission and the first generated token. This pattern prevents the interface from appearing frozen during prefill and sampling. Implementing this at both the library level (streaming functions) and application level (manual loops) ensures a consistent experience.
**Action:** Always implement a TTY-aware 'thinking' state using the `first_token` flag pattern to ensure the indicator is cleared exactly once when generation starts or is interrupted.
