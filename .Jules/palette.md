## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2025-05-16 - [UX State Scoping & Reusable Thinking Indicators]
**Learning:** In Julia CLI tools with streaming outputs, defining UI state variables (like `is_stdout_tty` or `first_token`) *before* the `try-catch` block is critical. If defined inside `try`, an interruption during the very first operations (like stream initialization) will cause an `UndefVarError` in the `catch` block when attempting to clear UI elements. The "dimmed dots" indicator (`\e[2m...\e[0m`) cleared by backspaces (`\b\b\b\e[K`) is an effective reusable pattern for signaling LLM latency.
**Action:** Always initialize CLI UI state variables before `try` blocks and use the standardized "dots" pattern for time-to-first-token latency.
