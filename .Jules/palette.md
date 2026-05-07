## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2026-05-07 - [Terminal Input & Line Clearing]
**Learning:** In custom Julia terminal line editors, an early check to skip the escape character (`\e`) will shadow all multi-byte escape sequences (like arrow keys `\e[A`), breaking keyboard navigation. Additionally, for robust cross-terminal line clearing, the ANSI sequence `\e[2K` is superior to manual space-padding as it handles varying terminal widths without visual artifacts.
**Action:** Always handle the escape character as a potential start of a sequence rather than discarding it, and prefer `\e[2K` for UI refreshes.
