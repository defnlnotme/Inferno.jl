## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2025-05-16 - [Robust Terminal UX and Input Handling]
**Learning:** In custom CLI line editors, blanket-skipping the escape character (`\e`) to "clean" input is a critical anti-pattern that breaks all multi-character sequences including arrow keys, Home/End, and Delete. Furthermore, using space-padding (`" "^80`) for line clearing is fragile and resolution-dependent; the standard ANSI escape sequence `\e[2K` (Clear Line) provides a much more robust and professional result.
**Action:** Always handle escape sequences properly in input loops and prefer standard ANSI codes for terminal manipulation.
