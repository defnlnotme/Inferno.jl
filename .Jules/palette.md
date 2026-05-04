## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2025-05-16 - [Custom Line Editor & Escape Sequences]
**Learning:** When implementing a custom line editor in a raw terminal mode, avoid globally skipping the escape character (`\e`). Doing so can shadow and break the processing of multi-byte escape sequences used for arrow keys, Home/End, and other terminal controls. Additionally, using ANSI escape sequences like `\e[2K` for line clearing is more robust across different terminal widths than space-padding.
**Action:** Ensure escape characters are handled as part of sequence detection rather than being dropped, and prefer standard ANSI sequences for terminal manipulation.
