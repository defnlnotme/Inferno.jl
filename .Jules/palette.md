## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2026-04-09 - [Visualizing LLM Latency with Thinking Indicators]
**Learning:** Local LLM inference can have a significant "Time To First Token" (TTFT) delay. Providing a TTY-aware "thinking" indicator (e.g., dimmed dots `\e[2m...\e[0m`) that is automatically cleared (`\b\b\b\e[K`) when generation begins significantly improves perceived performance and reduces user anxiety about whether the system is hung.
**Action:** Implement the dimmed thinking dots pattern in all streaming CLI interfaces, ensuring it only triggers on TTY and is cleanly cleared by the first arriving token.
