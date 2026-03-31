from __future__ import annotations

from pathlib import Path


class StyleCheckError(RuntimeError):
    """Raised when lightweight style checks fail."""


def run_style_checks(paths: list[Path]) -> list[str]:
    issues: list[str] = []
    for path in paths:
        text = path.read_text()
        lines = text.splitlines()
        for idx, line in enumerate(lines, start=1):
            if line.rstrip(" ") != line:
                issues.append(f"{path}:{idx}: trailing whitespace")
            if "\t" in line:
                issues.append(f"{path}:{idx}: tab character found; use spaces")
    return issues


def collect_python_files() -> list[Path]:
    files = sorted(Path("src").rglob("*.py")) + sorted(Path("tests").rglob("*.py"))
    return [path for path in files if path.is_file()]


def main() -> None:
    files = collect_python_files()
    issues = run_style_checks(files)
    if issues:
        raise StyleCheckError("\n".join(issues))


if __name__ == "__main__":
    main()
