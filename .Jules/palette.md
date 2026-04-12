## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2026-04-12 - [TTY-Aware Thinking Indicator Pattern]
**Learning:** For interactive LLM CLI tools, providing immediate visual feedback during the latency between prompt submission and first-token arrival significantly improves perceived responsiveness. Using dimmed dots (`\e[2m...\e[0m`) and clearing them with backspaces (`\b\b\b\e[K`) when the first token arrives creates a seamless transition. Distinguishing between `is_stdout_tty` (for indicators) and `is_stdin_tty` (for input echoing) ensures robust behavior in both interactive and piped environments.
**Action:** Implement the `is_stdout_tty` thinking indicator pattern in all interactive loops where generation latency is expected.
