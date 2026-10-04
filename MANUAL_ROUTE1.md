# Manual: reproduce the paper on your own computer (Route 1)

This manual takes you from nothing to all figures and tables of the paper, rebuilt and checked on
your own computer. No programming knowledge is needed. Copy each command, paste it into the terminal,
and press Enter.

**Time needed:** about 10 minutes the first time (mostly installation), then about 2 minutes per run.
**Computer needed:** Linux or macOS. On Windows, use Ubuntu through WSL (see Step 0). No GPU is needed.
**Disk space:** about 1.5 GB.

---

## Step 0: open a terminal

| System | How |
|---|---|
| **Linux** | Press `Ctrl + Alt + T`. |
| **macOS** | Press `Cmd + Space`, type `Terminal`, press Enter. |
| **Windows** | Open PowerShell as administrator and run `wsl --install`. Restart, then open "Ubuntu" from the Start menu. |

All following commands are typed in this terminal window.

## Step 1: make sure `git` is installed

```bash
git --version
```

If you see a version number such as `git version 2.43.0`, go to Step 2. Otherwise install git:

| System | Command |
|---|---|
| Ubuntu, Debian or WSL | `sudo apt update && sudo apt install git` |
| macOS | `xcode-select --install` |

## Step 2: download the package

```bash
git clone https://github.com/prashantstar123/NeuronStar.git
cd NeuronStar
```

This downloads about 300 MB: the code, the saved results and all observational data. You do not need
to download anything else.

## Step 3: install the Python packages (one time only)

You need **either** conda **or** Python 3.11. To see what you have, type:

```bash
conda --version
python3.11 --version
```

**If `conda --version` shows a number**, use this:

```bash
conda env create -f environment-route1.yml
conda activate ddb-route1
```

**Otherwise, if `python3.11 --version` shows a number**, use this:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-route1.txt
```

**If you have neither**, install Miniforge, which gives you conda:

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Answer `yes` to the questions. Close the terminal, open a new one, go back into the folder with
`cd NeuronStar`, and use the conda commands above.

**Important:** every time you open a new terminal, activate the environment again:
- `conda activate ddb-route1` (conda), or
- `source .venv/bin/activate` (Python 3.11), run inside the `NeuronStar` folder.

## Step 4: check everything

```bash
./reproduce.sh check
```

You should see:

```text
[check] PASS Python environment (exact package versions)
[check] unpacked data/observations/J0740_NICERXMM_full_mr.txt
[check] unpacked data/observations/J0437_post_equal_weights.dat
[check] unpacked data/observations/J1231_wmrsamples.txt
[check] PASS all 144 data and result files match their recorded checksums
```

The three `unpacked` lines appear only the first time. If you see `FAIL` instead, look up the message
in the table at the end of this manual.

## Step 5: rebuild and check the paper

```bash
./reproduce.sh route1
```

The program works through five steps and prints `PASS` after each figure and table. It finishes with:

```text
ROUTE 1 PASSED in 1.1 min: all 14 data-figure files are pixel-identical to the paper, the workflow diagram matches its SHA-256, and every number in Tables I and III-XII agrees.
```

**That's it: you have reproduced the paper.**

## Step 6: look at the results

| What | Where |
|---|---|
| The rebuilt figures (PDF) | `build/figures/` |
| The recomputed table values | `build/tables/` |
| The complete rebuilt paper | `build/reproduced_manuscript.pdf` (only if LaTeX is installed; see Step 7) |

To open the folder:

| System | Command |
|---|---|
| Linux | `xdg-open build/figures` |
| macOS | `open build/figures` |
| Windows (WSL) | `explorer.exe build/figures` |

## Step 7 (optional): build the paper PDF

The figures and tables are checked without LaTeX. To also build the full paper PDF, install LaTeX once:

| System | How |
|---|---|
| Ubuntu, Debian or WSL | `sudo apt install texlive-latex-extra texlive-publishers texlive-science` |
| macOS | install MacTeX from https://www.tug.org/mactex/ |

Then run `./reproduce.sh route1` again.

---

## What exactly was checked?

| Check | What it means |
|---|---|
| Files | Every saved result and data file is exactly the file used for the paper. Each has a fingerprint (SHA-256), and even a one-byte change would be detected. |
| Figures | Every data figure (Figs. 2–11) is drawn again from the saved results; Fig. 1, the workflow diagram, is checked by its SHA-256. Your new figure and the published figure are turned into pictures and compared dot by dot (pixel by pixel). They must be identical. |
| Tables | Every number in Tables I and III–XII is calculated again from the saved results and compared with the number printed in the paper. They must be identical; for the Appendix C Tables X–XII, the recomputed value rounded to the printed precision must equal the printed number. Table II lists the likelihood inputs, with their sources in `data/SOURCES.md`. |
| Paper | The paper is compiled again using your newly drawn figures. |

## If something goes wrong

| You see | Do this |
|---|---|
| `command not found: conda` or `python3.11` | Do Step 3 again; install Miniforge if needed. |
| `Python 3.11 is required` or `missing Python package` | Activate the environment (see "Important" in Step 3), then run the command again. |
| `numpy 2.4.6 is required, found ...` (or another package) | Your environment has other versions. Create a fresh one with Step 3. |
| `does not match its recorded checksum` or `is damaged` | The download is incomplete. Delete the folder and do Step 2 again. |
| `Permission denied` for `./reproduce.sh` | Type `bash reproduce.sh check` or `bash reproduce.sh route1` instead. |
| `pdflatex was not found` | This is not an error. LaTeX is not installed, so the PDF was not built; figures and tables were still checked. See Step 7. |
| `ROUTE 1 FAILED at step ...` | Copy the messages shown in the terminal and attach the files in `build/logs/` to a GitHub issue. |

For help, open an issue at https://github.com/prashantstar123/NeuronStar/issues, or
email Prashant Thakur at prashant@yonsei.ac.kr or prashantthakur1921@gmail.com. Please attach the
files in `build/logs/`.
