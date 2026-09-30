# Changelog

Notable changes per version (Keep a Changelog format, SemVer).

## [0.2.0] - 2026-10-01

### Fixed

- **"length cap hit" — compaction no longer dies here.** The running summary
  could outgrow the reply budget, so compaction was guaranteed to fail no
  matter how chunks were split. The summary ceiling now always fits inside the
  reply budget.
- **No more phantom splitting on small sessions.** Prompt sizes were
  underestimated whenever tool calls carried commands, code, or file contents,
  so chunks got split up to three levels deep for no good reason. Sizes are
  now measured correctly.
- **Dropped connections no longer kill compaction.** An "aborted" reply
  without a real cancellation is now retried like any other transient error:
  the same summarizer call is repeated up to 3 times with growing pauses,
  and compaction continues from where it stopped. Only a real Esc/cancel
  stops it.

### Changed

- **Faster compaction on large models.** A summarizer with a big context window
  (e.g. 1M tokens) no longer has every chunk forced into several slow
  sequential calls; chunks are sized to respect the model's real headroom.
- **Retries that actually retry.** Transient provider failures now backtrack
  and retry up to 3 times with growing pauses, like pi does for its own
  requests. Quota and billing errors stop immediately.
- **Live progress.** While compacting, you see in the transcript which model
  runs, how many chunks there are, the current chunk and percent, split levels
  if any, and a final timing summary. None of it reaches the model context.
- **Compact model menu.** `/compact-model` now opens the same kind of list as
  `/model`: type to filter, ↑/↓ to pick, Enter/Esc — and it fits the screen.

### Added

- `DESIGN.md` now ships in the npm package, so the README link no longer 404s.
