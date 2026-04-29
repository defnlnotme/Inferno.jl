## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2025-05-16 - [Keyboard Navigation Fix vs Backend Logic]
**Learning:** In projects where input handling is tightly coupled with core engine logic (e.g., custom TTY loops), fixing keyboard navigation bugs can be classified as "backend logic" rather than a "micro-UX improvement". For Palette, it's safer to focus on clear UI markers, styled prompts, and visual alignment rather than modifying the low-level character reading loops.
**Action:** Prioritize visual polish (banners, prompts) and feedback (loading indicators) over fixing core input handling bugs unless they can be addressed without touching the byte-level processing logic.
