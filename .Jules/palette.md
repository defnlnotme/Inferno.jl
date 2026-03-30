## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2025-05-20 - [Fractional Progress Indicators for Long Operations]
**Learning:** Replacing simple dots with fractional progress indicators (e.g., "5/24") significantly improves user perception of "time-to-completion" and overall app responsiveness. Using ANSI colors (Cyan for info) and carriage returns (\r) keeps the CLI clean and professional.
**Action:** Always prefer explicit "X/Y" progress over dots for discrete, countable operations like layer loading. Ensure a `println()` follows the loop to clear the line.
