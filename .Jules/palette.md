## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2025-05-22 - [Enhanced CLI Progress & Thinking Feedback]
**Learning:** In LLM CLI interfaces, the 'Assistant:' prompt should immediately follow with a visual "thinking" indicator (e.g., dimmed dots) to signify that generation has started but the first token hasn't arrived yet. Clearing this indicator with backspaces `\b` once the stream begins ensures a seamless transition. For long-running loading, a fractional progress counter (e.g., 'Loading: 5/24') is far superior to simple dots as it provides context on remaining time.
**Action:** Use dimmed dots `\e[2m...\e[0m` and backspaces `\b\b\b` in `bin/chat.jl` and fractional progress counters in `src/Loader.jl`.
