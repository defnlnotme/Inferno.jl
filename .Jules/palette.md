## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2025-05-16 - [LLM Thinking Indicator for CLI]
**Learning:** In LLM CLI applications, "time to first token" (TTFT) creates a perceived lag. A subtle, dimmed "thinking" indicator (e.g., `...` in `\e[2m`) that appears immediately after the prompt and is cleared via backspaces (`\b\b\b\e[K`) when generation begins, significantly improves perceived responsiveness without cluttering the final output.
**Action:** Implement TTY-aware thinking indicators in all streaming generation entry points to bridge the gap between prompt submission and token arrival.
