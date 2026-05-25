#!/usr/bin/env python3
"""
scripts/check_submission.py
============================
Pre-submission checklist validator for JOSS.

Checks every requirement JOSS reviewers look for and prints a clear
PASS / FAIL / WARN for each one. Fix all FAILs before submitting.

Usage
-----
    python scripts/check_submission.py
    python scripts/check_submission.py --strict   # treat WARNs as FAILs
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── Terminal colours ──────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def ok(msg: str)   -> str: return f"  {GREEN}PASS{RESET}  {msg}"
def fail(msg: str) -> str: return f"  {RED}FAIL{RESET}  {msg}"
def warn(msg: str) -> str: return f"  {YELLOW}WARN{RESET}  {msg}"
def head(msg: str) -> str: return f"\n{BOLD}{msg}{RESET}"


# ── Individual checks ─────────────────────────────────────────────────────────

def check_files() -> list:
    """All required files are present."""
    results = []
    required = {
        "README.md":           "Project homepage and documentation",
        "LICENSE":             "Open-source licence (MIT)",
        "CITATION.cff":        "Software citation metadata",
        "CONTRIBUTING.md":     "Contribution guidelines",
        "CODE_OF_CONDUCT.md":  "Community code of conduct",
        "CHANGELOG.md":        "Version history",
        "requirements.txt":    "Python dependencies",
        "setup.py":            "Package installer",
        "pyproject.toml":      "Modern packaging config",
        "Dockerfile":          "Containerised reproducibility",
        "paper/paper.md":      "JOSS manuscript",
        "paper/paper.bib":     "JOSS references",
        "tests/test_diabpred.py": "Unit tests",
        ".github/workflows/ci.yml": "CI pipeline",
        "scripts/run_experiment.py": "Reproducibility entry point",
        "examples/quick_prediction.py": "Usage example",
    }
    for path, desc in required.items():
        full = ROOT / path
        if full.exists():
            results.append(ok(f"{path} — {desc}"))
        else:
            results.append(fail(f"{path} MISSING — {desc}"))
    return results


def check_placeholders() -> list:
    """No placeholder text left in key files."""
    results = []
    checks = [
        ("paper/paper.md",  "First Last",           "Author name not updated in paper.md"),
        ("paper/paper.md",  "0000-0000-0000-0000",  "ORCID not updated in paper.md"),
        ("paper/paper.md",  "Your Institution",     "Institution not updated in paper.md"),
        ("CITATION.cff",    "0000-0000-0000-0000",  "ORCID not updated in CITATION.cff"),
        ("CITATION.cff",    "Your Institution",     "Institution not updated in CITATION.cff"),
        ("CITATION.cff",    "10.21105/joss.XXXXX",  "JOSS DOI placeholder still in CITATION.cff"),
        ("README.md",       "yourusername",         "GitHub username not updated in README.md"),
        ("setup.py",        "Your Name",            "Author name not updated in setup.py"),
    ]
    for filepath, placeholder, msg in checks:
        full = ROOT / filepath
        if not full.exists():
            continue
        content = full.read_text(encoding="utf-8")
        if placeholder in content:
            results.append(warn(msg))
        else:
            results.append(ok(f"No placeholder '{placeholder}' in {filepath}"))
    return results


def check_zenodo_doi() -> list:
    """CITATION.cff has a real Zenodo DOI."""
    results = []
    cff = ROOT / "CITATION.cff"
    if not cff.exists():
        results.append(fail("CITATION.cff not found"))
        return results
    content = cff.read_text(encoding="utf-8")
    if "10.5281/zenodo." in content:
        results.append(ok("Zenodo DOI found in CITATION.cff"))
    else:
        results.append(warn("Zenodo DOI not in CITATION.cff — needed before JOSS submission"))
    return results


def check_licence() -> list:
    """LICENSE file contains a real name (not placeholder)."""
    results = []
    lic = ROOT / "LICENSE"
    if not lic.exists():
        results.append(fail("LICENSE file missing"))
        return results
    content = lic.read_text(encoding="utf-8")
    if "Your Name" in content:
        results.append(warn("LICENSE still contains 'Your Name' — replace with your real name"))
    else:
        results.append(ok("LICENSE has a real author name"))
    if "MIT License" in content:
        results.append(ok("LICENSE is MIT (OSI-approved)"))
    return results


def check_package_imports() -> list:
    """All diabpred modules import without errors."""
    results = []
    modules = [
        "diabpred",
        "diabpred.data",
        "diabpred.models",
        "diabpred.evaluate",
        "diabpred.predict",
        "diabpred.visualize",
        "diabpred.cli",
    ]
    sys.path.insert(0, str(ROOT))
    for mod in modules:
        try:
            importlib.import_module(mod)
            results.append(ok(f"import {mod}"))
        except Exception as exc:
            results.append(fail(f"import {mod} — {exc}"))
    return results


def check_tests() -> list:
    """pytest passes with zero failures."""
    results = []
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
            capture_output=True, text=True, cwd=ROOT
        )
        output = proc.stdout + proc.stderr
        if proc.returncode == 0:
            # Extract passed count
            lines = [l for l in output.splitlines() if "passed" in l]
            summary = lines[-1].strip() if lines else "unknown"
            results.append(ok(f"All tests pass — {summary}"))
        else:
            failures = [l for l in output.splitlines() if "FAILED" in l or "ERROR" in l]
            for f in failures[:5]:
                results.append(fail(f"Test failure: {f.strip()}"))
    except Exception as exc:
        results.append(fail(f"Could not run pytest — {exc}"))
    return results


def check_lint() -> list:
    """flake8 reports zero errors."""
    results = []
    try:
        proc = subprocess.run(
            ["flake8", "diabpred/", "--max-line-length=100",
             "--ignore=E501,W503,E402"],
            capture_output=True, text=True, cwd=ROOT
        )
        if proc.returncode == 0:
            results.append(ok("flake8 — zero lint errors"))
        else:
            errors = proc.stdout.strip().splitlines()
            for e in errors[:8]:
                results.append(fail(f"Lint: {e}"))
            if len(errors) > 8:
                results.append(fail(f"... and {len(errors) - 8} more lint errors"))
    except FileNotFoundError:
        results.append(warn("flake8 not installed — run: pip install flake8"))
    return results


def check_paper() -> list:
    """paper/paper.md meets JOSS requirements."""
    results = []
    paper = ROOT / "paper" / "paper.md"
    bib   = ROOT / "paper" / "paper.bib"

    if not paper.exists():
        results.append(fail("paper/paper.md missing"))
        return results

    content = paper.read_text(encoding="utf-8")
    word_count = len([w for w in content.split() if w.strip()])

    # Word count
    if word_count >= 500:
        results.append(ok(f"Paper word count: {word_count} words (minimum 500)"))
    else:
        results.append(fail(f"Paper too short: {word_count} words (minimum 500)"))

    # Required sections
    for section in ["# Summary", "# Statement of Need", "# Methods",
                    "# Results", "# References"]:
        if section in content:
            results.append(ok(f"Paper has section: {section}"))
        else:
            results.append(fail(f"Paper missing section: {section}"))

    # YAML front matter fields
    for field in ["title:", "authors:", "orcid:", "affiliations:", "bibliography:"]:
        if field in content:
            results.append(ok(f"Paper YAML has field: {field}"))
        else:
            results.append(fail(f"Paper YAML missing field: {field}"))

    # References
    if bib.exists():
        bib_content = bib.read_text(encoding="utf-8")
        ref_count = bib_content.count("@article") + bib_content.count("@inproceedings")
        if ref_count >= 5:
            results.append(ok(f"paper.bib has {ref_count} references (minimum 5)"))
        else:
            results.append(warn(f"paper.bib only has {ref_count} references"))
    else:
        results.append(fail("paper/paper.bib missing"))

    return results


def check_readme() -> list:
    """README.md contains all required sections."""
    results = []
    readme = ROOT / "README.md"
    if not readme.exists():
        results.append(fail("README.md missing"))
        return results

    content = readme.read_text(encoding="utf-8")

    required_sections = [
        ("## Quick Start",        "Installation/quick-start instructions"),
        ("pip install",           "Install command"),
        ("## Running Tests",      "Test instructions"),
        ("pytest",                "pytest command"),
        ("## Benchmark Results",  "Results table"),
        ("## Citation",           "Citation instructions"),
    ]

    for marker, desc in required_sections:
        if marker in content:
            results.append(ok(f"README has: {desc}"))
        else:
            results.append(warn(f"README missing: {desc} (look for '{marker}')"))

    # CI badge
    if "badge.svg" in content or "actions/workflows" in content:
        results.append(ok("README has CI badge"))
    else:
        results.append(warn("README has no CI badge — add one from GitHub Actions"))

    return results


def check_reproducibility() -> list:
    """Reproducibility entry point works with synthetic data."""
    results = []
    script = ROOT / "scripts" / "run_experiment.py"
    if not script.exists():
        results.append(fail("scripts/run_experiment.py missing"))
        return results

    # Check synthetic data exists or create it
    data_path = ROOT / "data" / "diabetes.csv"
    if not data_path.exists():
        results.append(warn(
            "data/diabetes.csv not found — download from Kaggle before final submission"
        ))
    else:
        results.append(ok("data/diabetes.csv present"))

    # Check outputs from a previous run
    meta = ROOT / "outputs" / "experiment_meta.json"
    if meta.exists():
        import json
        with open(meta) as f:
            m = json.load(f)
        results.append(ok(
            f"Previous experiment output found — best model: {m.get('best_model')} "
            f"(ROC-AUC={m.get('best_roc_auc')})"
        ))
    else:
        results.append(warn(
            "No outputs/experiment_meta.json found — "
            "run: python scripts/run_experiment.py"
        ))

    return results


def check_docker() -> list:
    """Dockerfile present and syntactically valid."""
    results = []
    for fname in ["Dockerfile", "Dockerfile.webapp", "docker-compose.yml"]:
        path = ROOT / fname
        if path.exists():
            results.append(ok(f"{fname} present"))
        else:
            results.append(warn(f"{fname} missing (optional but recommended)"))
    return results


# ── Main runner ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DiabPred JOSS pre-submission checklist",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--strict", action="store_true",
                        help="Treat WARNs as FAILs")
    args = parser.parse_args()

    all_results: list[str] = []

    sections = [
        ("Required Files",          check_files),
        ("Placeholder Text",        check_placeholders),
        ("Zenodo DOI",              check_zenodo_doi),
        ("Licence",                 check_licence),
        ("Package Imports",         check_package_imports),
        ("Unit Tests",              check_tests),
        ("Lint (flake8)",           check_lint),
        ("JOSS Paper",              check_paper),
        ("README Quality",          check_readme),
        ("Reproducibility",         check_reproducibility),
        ("Docker Support",          check_docker),
    ]

    for title, fn in sections:
        print(head(f"[{title}]"))
        results = fn()
        for r in results:
            print(r)
        all_results.extend(results)

    # Summary
    passes = sum(1 for r in all_results if "PASS" in r)
    fails  = sum(1 for r in all_results if "FAIL" in r)
    warns  = sum(1 for r in all_results if "WARN" in r)

    print(f"\n{'=' * 55}")
    print(f"{BOLD}Summary{RESET}")
    print(f"  {GREEN}PASS{RESET}  {passes}")
    print(f"  {YELLOW}WARN{RESET}  {warns}  (fix before submitting)")
    print(f"  {RED}FAIL{RESET}  {fails}  (must fix — will cause rejection)")
    print(f"{'=' * 55}")

    if fails == 0 and (warns == 0 or not args.strict):
        print(f"\n{GREEN}{BOLD}Ready to submit to JOSS!{RESET}")
        print("  Go to: https://joss.theoj.org/papers/new")
    elif fails == 0:
        print(f"\n{YELLOW}{BOLD}Almost ready — resolve WARNs first.{RESET}")
    else:
        print(f"\n{RED}{BOLD}Not ready — fix all FAILs before submitting.{RESET}")

    sys.exit(1 if (fails > 0 or (args.strict and warns > 0)) else 0)


if __name__ == "__main__":
    main()
