## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2026-04-20 - [Thinking Indicator Pattern for LLM Latency]
**Learning:** For LLM inference engines with non-trivial TTFT (Time To First Token), a TTY-aware "thinking" indicator (e.g., dimmed dots `\e[2m...\e[0m`) provides essential feedback that the system is working. This indicator must be precisely cleared with backspaces (`\b\b\b\e[K`) immediately before the first token is printed to maintain a seamless visual transition.
**Action:** Implement the `first_token` flag pattern in streaming loops: print dimmed dots if `isa(stdout, Base.TTY)`, then clear them and reset the flag when the first token arrives. Ensure the `catch` block for `InterruptException` also clears the indicator to prevent visual artifacts on the terminal.
