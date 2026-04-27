## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2025-05-20 - [UI State Scoping for Interrupt Handling]
**Learning:** When implementing terminal feedback like "thinking" indicators in Julia CLI tools, define UI state variables (e.g., `is_stdout_tty`, `first_token`) *before* the main `try` block. If they are defined inside the `try` block and an `InterruptException` (Ctrl+C) occurs before they are initialized, the `catch` block will fail with an `UndefVarError` when trying to clean up the UI.
**Action:** Always scope UI state variables at the beginning of the function or interactive loop before entering `try-catch` blocks.
