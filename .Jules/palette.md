## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2025-05-20 - [High-Density Progress Indicators]
**Learning:** For CLI tools with long-running iterative tasks (like loading transformer layers), a single-line fractional progress indicator (e.g., `5/24`) using `\r` (carriage return) is significantly more informative than a simple dot trail. It provides users with context on the remaining time and process status without cluttering the scrollback buffer.
**Action:** Replace additive indicators (like dots) with overwriting fractional ones for predictable, multi-step operations.
