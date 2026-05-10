## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2025-05-22 - [Polished CLI Interaction Patterns]
**Learning:** For interactive LLM chat interfaces, visual hierarchy is significantly improved by using distinct, styled prefixes (e.g., bold cyan "You> " and bold green "Assistant> "). Additionally, a dimmed "..." thinking indicator provides crucial "System is working" feedback during initial token generation latency. Using the `\e[2K` ANSI escape sequence for line clearing is more robust across different terminal widths than space-padding.
**Action:** Always implement `Assistant> ` prefixes and TTY-aware thinking indicators in chat loops. Prefer `\e[2K` for terminal line management.
