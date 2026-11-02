# GitHub Actions Lab – CI Pipeline for an ML Metrics Library

![CI](https://github.com/GauravChoughule/github-actions-CI-lab/actions/workflows/ci.yml/badge.svg)

This lab sets up **Continuous Integration (CI)** with GitHub Actions: every push and pull request is automatically checked for code style, tested on three Python versions, and measured for test coverage. Broken code can be blocked from being merged.

The code being tested is a small library of **machine learning evaluation metrics written from scratch** (accuracy, precision, recall, F1, confusion matrix, MAE, RMSE), replacing the calculator from the original lab.

---

## 1. What you need to install

| Tool | Version | Why | Download |
|---|---|---|---|
| Git | any recent version | to clone and push the code | https://git-scm.com/downloads |
| Python | 3.10, 3.11 or 3.12 | to run the tests on your own machine | https://www.python.org/downloads/ |
| GitHub account | – | GitHub Actions runs on GitHub | https://github.com |

Windows: when installing Python, tick **"Add python.exe to PATH"**.

Check the installs in a terminal (PowerShell on Windows, Terminal on Mac):

```bash
git --version
python --version     # Mac/Linux may need: python3 --version
```

You do **not** need to install anything for the CI itself; GitHub runs it on its own machines.

---

## 2. Project structure

```
github-actions-ci-lab
├── .github/workflows/ci.yml   # the CI pipeline
├── src/metrics.py             # the code being tested
├── test/test_pytest.py        # pytest tests (20 test cases)
├── test/test_unittest.py      # unittest tests (11 tests)
├── data/sample_predictions.csv  # saved model predictions used by a test
├── pyproject.toml             # pytest and ruff settings
├── requirements.txt           # pinned tool versions
└── .gitignore
```

---

## 3. Run the tests on your own machine

```bash
git clone https://github.com/GauravChoughule/github-actions-ci-lab.git
cd github-actions-ci-lab
```

Create and activate a virtual environment:

```bash
python -m venv lab_env
lab_env\Scripts\activate          # Windows PowerShell
source lab_env/bin/activate       # Mac/Linux
```

Install the tools:

```bash
pip install -r requirements.txt
```

Run each check exactly as the pipeline does:

```bash
ruff check .
pytest -v --cov=src --cov-report=term-missing --cov-fail-under=90
python -m unittest discover -s test -p "test_unittest.py" -v
```

### What success looks like

- `ruff check .` prints `All checks passed!`
- `pytest` ends with `20 passed` and a coverage table showing `src/metrics.py` at 100%, followed by `Required test coverage of 90% reached.`
- `unittest` ends with `Ran 11 tests ... OK`

Windows PowerShell blocking `activate`? Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

---

## 4. See the pipeline run on GitHub

1. Push any change to `main`, or open the **Actions** tab and click **CI → Run workflow**.
2. In the **Actions** tab, click the run.

### What a successful run looks like

- **Lint (ruff)** runs first and gets a green check.
- Then three test jobs run in parallel: **Test (Python 3.10)**, **Test (Python 3.11)**, **Test (Python 3.12)**, all green.
- The run's **Summary** page shows a coverage table for each Python version.
- Under **Artifacts** at the bottom of the Summary page, there's a downloadable test report for each Python version.
- The badge at the top of this README turns green: "CI passing".

---

## 5. See CI catch a bug (pull request check)

This shows the main point of CI: bad code is caught before it reaches `main`.

**One-time setup: block merging when CI fails**
On GitHub, open **Settings → Branches → Add branch protection rule** (it may say "Add classic branch protection rule"). Set the branch name pattern to `main`, tick **Require status checks to pass before merging**, and add `Lint (ruff)` and the three `Test (Python …)` checks. Save. The checks only appear in the search box after the workflow has run at least once.

**Introduce a bug on a new branch:**

```bash
git checkout -b break-accuracy
```

In `src/metrics.py`, inside `accuracy`, change `return correct / len(y_true)` to `return correct / (len(y_true) + 1)`. Then:

```bash
git commit -am "Introduce a bug on purpose"
git push -u origin break-accuracy
```

Open a pull request from `break-accuracy` into `main` on GitHub.

**Expected result:** lint passes, but all three test jobs fail with red crosses, the failing tests are listed in the logs, and the **Merge** button is blocked. Close the PR without merging and delete the branch.

---

## 6. Changes made to the original lab

### The CI pipeline (main change)

| Area | Original lab | This version | What it does |
|---|---|---|---|
| Location | Workflows in `Labs/Github_Labs/Lab1/workflows/` | `.github/workflows/ci.yml` at the repo root | GitHub only runs workflows from this folder, so the originals never actually ran |
| Errors | `run-nam` typo; `branches` and `branches-ignore` used together (not allowed) | Fixed | The original pytest workflow file was invalid |
| Action versions | `checkout@v2`, `setup-python@v2`, `upload-artifact@v2` | `@v4`, `@v5`, `@v4` | v2 versions are outdated; `upload-artifact@v2` no longer works |
| Python | 3.8 (end of life) | **Matrix of 3.10, 3.11, 3.12** run in parallel | Proves the code works on every supported version |
| Workflows | Two separate workflows, one for pytest, one for unittest | One pipeline with two jobs: **lint**, then **test** | `needs: lint` means tests only run if the code passes style checks |
| Triggers | Push to `main` (plus unrelated issue/label events) | Push to `main`, **pull requests** into `main`, and a manual **Run workflow** button | PRs get checked before merging; README-only pushes are skipped |
| Linting | None | **ruff** checks for unused imports, bugs and style, with errors shown inline on the PR | Catches problems before tests even run |
| Coverage | None | **pytest-cov**, and the run **fails if coverage drops below 90%** | Stops untested code from being merged |
| Reporting | `echo "Tests passed"` | Coverage table on the run's **Summary** page, plus test and coverage reports saved as **artifacts** for 7 days | Results are visible without digging through logs |
| Speed | Downloads packages every run | **pip caching** | Faster runs |
| Wasted runs | None handled | **concurrency**: a new push cancels the older run on the same branch | Saves runner minutes |
| Safety | Default permissions, no time limit | `permissions: contents: read` and job **timeouts** | Least privilege; a stuck job can't run for hours |
| Failures | – | `fail-fast: false`, and unittest runs even if pytest fails | You see every failure in one run, not just the first |

### The code and tests

| Area | Original lab | This version |
|---|---|---|
| Code | Calculator (`fun1`–`fun4`) | ML evaluation metrics written from scratch |
| Input checks | Only some functions checked input | Every metric rejects empty lists, different lengths, non-numbers and booleans with a clear error |
| pytest | 4 tests with plain `assert` | 20 test cases using **parametrize**, a **fixture** that loads saved predictions from `data/`, `pytest.approx` for decimals, and `pytest.raises` for errors |
| unittest | 4 tests, plus a `sys.path` workaround | 11 tests in 3 classes, using `subTest`, `assertAlmostEqual` and `assertRaises`; no path workaround needed |
| Config | None | `pyproject.toml` holds pytest and ruff settings in one place |
| Dependencies | `pytest` (unpinned) | Exact versions of pytest, pytest-cov and ruff, so local runs and CI match |

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run commands from the repo's top folder, not from inside `test/`. |
| Workflow doesn't appear in the Actions tab | Check the file is at exactly `.github/workflows/ci.yml` at the repo root. |
| `Required test coverage of 90% not reached` | New code was added without tests. Add tests for it. |
| Status checks don't appear in branch protection | Let the workflow run once first, then search for the check names. |
