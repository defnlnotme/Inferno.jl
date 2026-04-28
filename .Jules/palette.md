## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2025-05-20 - [Reliable Terminal UX with Scoped UI State]
**Learning:** When implementing terminal feedback (like thinking indicators) that must be cleared during interrupts, define UI state variables (e.g., `is_stdout_tty`, `first_token`) BEFORE the `try` block. This ensures they are available in the `catch` block for cleanup, preventing `UndefVarError` that would otherwise crash the error handler and leak raw stack traces.
**Action:** Always scope UI-related control flags before `try-catch` blocks in streaming functions.
