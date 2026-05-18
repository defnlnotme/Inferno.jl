## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2025-05-20 - [Interactive CLI UX & Terminal Robustness]
**Learning:** In Julia `Base.Terminals`, raw character input via `read(term, Char)` requires careful handling of the escape character (`\e`). Unconditionally discarding it breaks support for ANSI sequences used by arrow keys, Home/End, and other control keys. Additionally, using `\e[2K` for line clearing is significantly more robust than space-padding as it correctly handles various terminal widths without overflow or underflow.
**Action:** Always allow the escape character to be processed as the start of a potential sequence in custom line editors, and prefer standard ANSI erase sequences for terminal state management.
