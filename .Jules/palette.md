## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2026-04-06 - [Standardizing Thinking Indicators in CLI]
**Learning:** In interactive CLI applications with high-latency processing (like LLM inference), providing immediate visual feedback through a "thinking" indicator (e.g., dimmed dots) significantly improves the perceived responsiveness. Distinguishing between stdin and stdout TTY states is crucial to avoid echoing input twice when piping commands while still showing indicators on the terminal.
**Action:** Use the dimmed dots \e[2m...\e[0m and backspace-clear \b\b\b\e[K pattern for all interactive components. Always check isa(stdout, Base.TTY) separately from stdin for output-only escape sequences.
