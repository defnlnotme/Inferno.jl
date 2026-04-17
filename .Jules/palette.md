## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2025-05-16 - [Dynamic Thinking Indicators]
**Learning:** For LLM streaming interfaces, providing immediate visual feedback during the latency between prompt submission and the first token (TTFT) significantly improves perceived performance. Using dimmed dots (`\e[2m...\e[0m`) that are cleared via backspaces (`\b\b\b\e[K`) upon receiving the first token provides a seamless transition from "waiting" to "generating" without cluttering the console history.
**Action:** Implement the `first_token` flag pattern with TTY-aware dimmed dots and backspace clearing in all stdout-streaming generation paths. Ensure indicators are also cleared in `catch` blocks for `InterruptException`.
