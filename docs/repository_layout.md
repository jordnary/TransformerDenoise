# Repository layout

Keep reusable source code, configurations, tests, and general project documentation in Git. Keep assignment materials, reports, personal files, datasets, and generated artifacts local.

| Path | Purpose | Git policy |
| --- | --- | --- |
| `README.md` | Chinese project overview and usage | Tracked; links to the English README |
| `README.en.md` | English project overview and usage | Tracked; links to the Chinese README |
| `denoise/` | Importable Python package and command-line modules | Tracked |
| `configs/` | Reusable training and comparison configurations | Tracked |
| `tests/` | Structural and numerical contract tests | Tracked |
| `docs/` | General project documentation | Tracked; no assignment submissions or personal records |
| `assignment/` | Assignment requirements and supplied materials | Local; only the empty `.gitkeep` is tracked |
| `reports/` | Reports, experiment plans, metrics, and submission drafts | Local; only the empty `.gitkeep` is tracked |
| `personal/` | Personal notes, input images, scratch files, and archives | Local; only the empty `.gitkeep` is tracked |
| `data/` | Downloaded datasets | Ignored |
| `runs/` | Training checkpoints and run state | Ignored |
| `outputs/` | Evaluation exports and generated figures | Ignored |

Use English ASCII names for files and directories, preferably `snake_case`. Use repository-relative paths in documentation and examples. Put personal configuration in `personal/` or use a `*.local.*` filename; do not put credentials or machine-specific information in tracked files.

The `.gitkeep` files preserve the three local-material directories in fresh clones without publishing their contents. Suggested local paths include `assignment/assignment_requirements.png`, `reports/assignment_report.md`, `reports/experiments/`, and `personal/archive/`. Runtime directories are created when training or evaluation needs them. Run the documented commands from the repository root so existing relative configuration and checkpoint paths resolve correctly.

## Before committing

Review the paths and content being staged:

```powershell
git status --short
git diff --cached --name-status
git diff --cached
git ls-files -- assignment reports personal data runs outputs
```

The last command should list only `assignment/.gitkeep`, `reports/.gitkeep`, and `personal/.gitkeep`. Do not use `git add -f` for local materials. Ignore rules do not remove files already tracked by Git: use `git rm --cached -- <relative-path>` to stop tracking a file while retaining the local copy.

Removing a tracked file does not erase its previous versions from Git history. Review existing history before publishing an older repository; cleaning that history is a separate operation. The legacy `archive/` and `submission/` locations remain ignored as a safeguard, but new archives belong in `personal/archive/` and submission drafts in `reports/`.
