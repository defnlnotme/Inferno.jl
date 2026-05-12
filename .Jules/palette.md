## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2025-05-16 - [CLI Custom Editor & Keyboard Navigation]
**Learning:** In custom Julia line editors using raw mode, escape character handlers must be carefully placed. A generic "skip escape character" check at the start of the input loop will shadow ANSI sequences for arrow keys and home/end keys, breaking standard keyboard navigation. Additionally, consistent prompt styling should be reapplied inside terminal-clearing and line-refreshing functions to maintain the UI's visual identity during redraws.
**Action:** Remove generic escape-skipping blocks in 'read_line_chat' and ensure 'refresh_line' and 'clear_line' use 'printstyled' to maintain consistent prompt colors.
