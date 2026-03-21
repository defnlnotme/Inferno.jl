## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2025-05-16 - [Fractional Progress Indicators for CLI Loading]
**Learning:** In terminal-based LLM engines, replace opaque progress dots with fractional indicators (`\rLoading: n/total`) to give users precise feedback on long-running weight loading. This reduces "is it hung?" anxiety and allows for predictable wait times.
**Action:** Use `print("\r...")` with carriage returns in model loading loops to provide dynamic, single-line feedback without cluttering the scrollback buffer.
