## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2025-05-16 - [Dimmed Thinking Indicators]
**Learning:** For low-latency but not instantaneous CLI operations like LLM token generation, providing a dimmed thinking indicator (`\e[2m...\e[0m`) that is cleared via backspaces (`\b\b\b\e[K`) once the first result arrives significantly improves the perceived responsiveness without cluttering the final output. This pattern should be applied both in interactive loops and library streaming functions when outputting to a TTY.
**Action:** Implement the dimmed dot pattern for "thinking" states and ensure it is TTY-aware to avoid escape sequence pollution in non-interactive environments.
