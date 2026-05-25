# JOSS Submission Checklist

Use this before submitting to https://joss.theoj.org/papers/new

## Software Requirements

- [x] Open-source license (`LICENSE` – MIT)
- [x] Source code in a public GitHub repository
- [x] Installable package (`pip install -e .` or `pip install -r requirements.txt`)
- [x] Automated tests (`pytest tests/ -v`)
- [x] Tests pass on CI (GitHub Actions `.github/workflows/ci.yml`)
- [x] Significant software (not a trivial wrapper)

## Documentation Requirements

- [x] `README.md` with installation, usage, and API reference
- [x] Inline docstrings on every public function
- [x] Usage examples (`examples/quick_prediction.py`)
- [x] Reproducibility instructions (README + `run_experiment.py`)

## Community Requirements

- [x] `CONTRIBUTING.md` with contribution guidelines
- [x] `CODE_OF_CONDUCT.md` (link to Contributor Covenant in CONTRIBUTING.md)
- [x] `CITATION.cff` for software citation

## Paper Requirements (`paper/paper.md`)

- [x] Title
- [x] Author list with ORCID
- [x] Affiliation(s)
- [x] **Summary** – what does the software do? (≥ 100 words)
- [x] **Statement of Need** – why does this software exist? who needs it?
- [x] **Methods** – how does it work?
- [x] **Results** – what does it produce?
- [x] References (`paper.bib`) – ≥ 5 relevant citations
- [x] Paper renders without error (`docker run --rm -v $PWD:/data -w /data openjournals/paperdraft`)
- [ ] Figures referenced in paper (add after running experiment)

## Pre-submission Steps

1. **Get an ORCID** – https://orcid.org/register (free, takes 2 min)
2. **Push to GitHub** – make the repository public
3. **Tag a release** – `git tag v1.0.0 && git push origin v1.0.0`
4. **Create a Zenodo release** – connect GitHub to https://zenodo.org and archive the tag
5. **Verify CI passes** on the main branch
6. **Run the experiment once** and commit the benchmark CSV to `outputs/tables/`
7. **Check paper builds** with the JOSS Docker image (command above)

## Submission

1. Go to https://joss.theoj.org/papers/new
2. Fill in:
   - Repository URL: `https://github.com/yourusername/diabpred`
   - Branch: `main`
   - Software language: Python
   - Journal: JOSS
3. Select an editor or leave for assignment
4. Respond promptly to reviewer comments (typical turnaround: 2–8 weeks)

## Common Rejection Reasons (avoid these)

- Paper too short (< 250 words in Statement of Need + Summary)
- Tests missing or not running
- No CI badge on README
- ORCID missing for authors
- References lack DOIs
- Software doesn't install cleanly from instructions
