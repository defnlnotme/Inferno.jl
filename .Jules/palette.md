## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2025-05-16 - [Latency Signaling with Thinking Indicators]
**Learning:** In LLM inference applications where "Time To First Token" (TTFT) can be high, providing a subtle "thinking" indicator (dimmed dots `...`) significantly improves perceived performance. Clearing this indicator exactly when the first token arrives ensures a seamless transition to the generated content. Using ANSI escape sequences `\b\b\b\e[K` allows for precise inline clearing in TTYs.
**Action:** Implement TTY-aware thinking indicators in all streaming response paths to mask inference latency.
