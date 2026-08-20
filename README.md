# Hierarchical Statistical Analyzer

[![DOI](https://zenodo.org/badge/1207166109.svg)](https://doi.org/10.5281/zenodo.19500564)

GUI-based Python framework for hierarchical and replicate-aware statistical analysis of biological datasets.

The software is designed for:

- stratified nonparametric testing
- blocked experimental designs
- hierarchical biological data
- replicate-aware inference
- skewed or non-Gaussian distributions commonly encountered in molecular biology

## Features

### Statistical methods

- Global stratified rank permutation test across selected groups
- Pairwise stratified Wilcoxon (van Elteren-style) permutation tests
- Linear mixed-effects models with biological replicate as a random intercept
- Optional LMM response transforms: none, log2, natural log, or log10
- Holm correction across planned pairwise contrasts
- Block-aware Cliff's delta, group medians, and median differences

Log transforms require strictly positive responses. The analyzer reports an error for nonpositive values rather than silently discarding them or adding a pseudocount.

### Experimental design support

- Biological replicate blocking
- Hierarchical data structures
- Planned pairwise contrasts
- Unequal sample sizes
- Within-replicate permutations that preserve group sizes
- Multiple statistical engines in one analysis

### GUI workflow

- CSV and XLSX input
- Automatic initial column mapping with editable group and replicate assignments
- Contrast selection
- Background analysis with progress and elapsed-time reporting
- On-screen results and CSV export

## Installation

Clone the repository and install its dependencies:

```bash
git clone https://github.com/sebastienleonmichel/hierarchical-statistical-analyzer.git
cd hierarchical-statistical-analyzer
python -m pip install -r requirements.txt
```

Tkinter is included with most standard Python installations. On systems where it is packaged separately, install the appropriate Tk package for your Python distribution.

## Usage

```bash
python hierarchical_statistical_analyzer.py
```

The application accepts wide-format `.csv` and `.xlsx` files. Each input column is mapped to a biological group and replicate ID in the GUI. Select the planned contrasts and statistical engines, then run the analysis. Results are saved beside the input file as `<input-name>_statistical-analysis.csv`.

## Citation

If you use this software, cite the archived release identified in [`CITATION.cff`](CITATION.cff).

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
