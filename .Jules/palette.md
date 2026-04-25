## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2025-05-16 - [TTY-Aware Thinking Indicators]
**Learning:** In terminal-based LLM applications, a "thinking" indicator (like dimmed dots) provides essential feedback during long-latency first-token generation. To maintain a clean UI, this indicator must be TTY-aware (only shown if `stdout` is a TTY) and correctly cleared (using `\b\b\b\e[K`) before the first token is printed. Robustness requires clearing the indicator in `catch` blocks as well, ensuring it doesn't persist if the operation is interrupted.
**Action:** Implement `first_token` tracking and TTY-aware indicator clearing in all streaming functions and interactive loops.
