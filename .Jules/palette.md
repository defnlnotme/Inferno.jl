## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2025-05-16 - [Correct Command Precedence in Julia CLI]
**Learning:** In Julia, the `&&` operator has higher precedence than `||`. When implementing multiple slash commands in a CLI loop, a line like `line == "/quit" || line == "/exit" && (println("Goodbye"); break)` will be parsed as `line == "/quit" || (line == "/exit" && ...)`. This causes the `/quit` command to evaluate to `true` (doing nothing) while only `/exit` actually triggers the `break`.
**Action:** Always wrap logical OR conditions for CLI commands in parentheses: `(line == "/quit" || line == "/exit") && break`.
