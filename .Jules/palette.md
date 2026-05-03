## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2025-05-16 - [Keyboard Navigation & Robust Line Clearing]
**Learning:** Redundant escape sequence skipping (e.g., `if c == '\e' continue end`) in manual line editors shadows subsequent CSI handling, breaking arrow keys and other terminal controls. Additionally, using space-padding (`" "^80`) for line clearing is fragile and terminal-width dependent; the ANSI sequence `\e[2K` (Clear In Line) is much more robust for modern terminal UX.
**Action:** Avoid manual escape-skipping blocks in line editors and prefer standard ANSI escape sequences like `\e[2K` for UI refresh logic.
