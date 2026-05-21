
"""
Hierarchical Statistical Analyzer
=================================

GUI-based framework for:
- stratified nonparametric testing
- linear mixed-effects modeling
- fixed-block linear modeling
- replicate-aware statistical analysis
- skewed biological datasets

Supports:
- blocked experimental designs
- hierarchical inference
- multiple testing correction
- contextual data mapping

Author: Sébastien Terreau
Year: 2026
Version: 2.2.11
"""


import time
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import rankdata, normaltest

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


APP_VERSION = "v2.2.11"


# ============================================================
# RESHAPE
# ============================================================

def reshape_long(df_wide, col_to_group, col_to_repl):

    rows = []

    for col in df_wide.columns:

        grp = col_to_group[col]
        rep = col_to_repl[col]

        vals = pd.to_numeric(
            df_wide[col],
            errors="coerce"
        ).dropna()

        vals = vals[vals > 0]

        for v in vals:

            rows.append({

                "value": float(v),
                "group": grp,
                "replicate_id": rep,
                "column": col

            })

    return pd.DataFrame(rows)


# ============================================================
# REPLICATE-LEVEL SUMMARY UTILITIES
# ============================================================

def add_multiple_testing_corrections(rows, raw_p):
    """
    Add Bonferroni, Holm, and Benjamini-Hochberg FDR-adjusted
    p-values to a list of result rows.
    """

    raw_p_array = np.array(raw_p)

    valid = np.isfinite(raw_p_array)

    holm_adj = np.full(len(raw_p), np.nan)
    bonf_adj = np.full(len(raw_p), np.nan)
    fdr_adj = np.full(len(raw_p), np.nan)

    if np.sum(valid) > 0:

        _, p_holm, _, _ = multipletests(
            raw_p_array[valid],
            method="holm"
        )

        _, p_bonf, _, _ = multipletests(
            raw_p_array[valid],
            method="bonferroni"
        )

        _, p_fdr, _, _ = multipletests(
            raw_p_array[valid],
            method="fdr_bh"
        )

        holm_adj[valid] = p_holm
        bonf_adj[valid] = p_bonf
        fdr_adj[valid] = p_fdr

    for i in range(len(rows)):

        rows[i]["p_bonferroni"] = bonf_adj[i]
        rows[i]["p_holm"] = holm_adj[i]
        rows[i]["p_fdr_bh"] = fdr_adj[i]

    return rows


def summarize_replicate_medians(
    df_long,
    log_transform=False
):
    """
    Collapse cell-level values into one median value per
    group per biological replicate.

    This avoids treating thousands of cells as independent
    biological replicates when running replicate-level tests.
    """

    df = df_long.copy()

    if log_transform:

        df["response"] = np.log(df["value"])

    else:

        df["response"] = df["value"]

    summary = (
        df.groupby(
            ["replicate_id", "group"],
            as_index=False
        )["response"]
        .median()
        .rename(columns={"response": "summary_value"})
    )

    return summary



def simplify_output_columns(results):
    """
    Keep the output table focused on contrast, method, effect size, and p-values.
    """

    columns_to_drop = [
        "Summary",
        "n_replicates",
        "statistic",
        "coefficient_B_minus_A"
    ]

    return results.drop(
        columns=[
            col for col in columns_to_drop
            if col in results.columns
        ]
    )


def run_column_residual_normality(df_wide):
    """
    Run D'Agostino-Pearson omnibus normality tests on residuals
    from an intercept-only model for each original input column.

    For each column:
        residual = value - column mean

    Decision threshold:
        p < 0.05 -> non-normal residuals
    """

    rows = []

    for col in df_wide.columns:

        vals = pd.to_numeric(
            df_wide[col],
            errors="coerce"
        ).dropna()

        # Keep this consistent with the analyzer's main reshape step.
        vals = vals[vals > 0]

        n_values = int(len(vals))

        if n_values > 0:

            mean_value = float(vals.mean())
            median_value = float(vals.median())

        else:

            mean_value = np.nan
            median_value = np.nan

        row = {
            "Analysis": "Residual normality",
            "Column": col,
            "Contrast": "",
            "Method": "D'Agostino-Pearson omnibus normality test",
            "p_raw": np.nan,
            "p_bonferroni": np.nan,
            "p_holm": np.nan,
            "p_fdr_bh": np.nan,
            "Normality_decision": "not tested: n < 8",
            "Median": median_value,
            "Mean": mean_value
        }

        if n_values >= 8:

            residuals = vals.to_numpy(dtype=float) - float(vals.mean())

            try:

                _, pval = normaltest(
                    residuals,
                    nan_policy="omit"
                )

                pval = float(pval)

                row["p_raw"] = pval

                if np.isfinite(pval):

                    if pval < 0.05:

                        row["Normality_decision"] = (
                            "non-normal residuals"
                        )

                    else:

                        row["Normality_decision"] = (
                            "no evidence of non-normal residuals"
                        )

                else:

                    row["Normality_decision"] = "not tested: invalid p-value"

            except Exception as e:

                row["Normality_decision"] = f"normality test failed: {e}"

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# WILCOXON UTILITIES
# ============================================================

def mannwhitney_u(x, gA_mask):

    r = rankdata(x, method="average")

    n1 = int(gA_mask.sum())

    U = r[gA_mask].sum() - n1 * (n1 + 1) / 2

    return float(U)


def cliffs_delta_from_values(values_A, values_B):
    """
    Compute Cliff's delta for two independent sets of values.

    Sign convention:
        positive delta -> group B tends to have higher values than group A
        negative delta -> group B tends to have lower values than group A

    Ties contribute 0 to the effect size.
    """

    values_A = np.asarray(values_A, dtype=float)
    values_B = np.asarray(values_B, dtype=float)

    values_A = values_A[np.isfinite(values_A)]
    values_B = values_B[np.isfinite(values_B)]

    nA = len(values_A)
    nB = len(values_B)

    if nA == 0 or nB == 0:

        return np.nan

    values = np.concatenate([values_A, values_B])

    gA_mask = np.concatenate([
        np.ones(nA, dtype=bool),
        np.zeros(nB, dtype=bool)
    ])

    u_A = mannwhitney_u(
        values,
        gA_mask
    )

    # U_A estimates P(A > B) + 0.5 * P(A = B).
    # Cliff's delta for B minus A is the opposite direction:
    # P(B > A) - P(B < A).
    delta_B_minus_A = 1 - (2 * u_A / (nA * nB))

    return float(delta_B_minus_A)


def stratified_cliffs_delta(
    df_long,
    groupA,
    groupB
):
    """
    Compute a block-aware Cliff's delta for a pairwise contrast.

    The effect size is calculated within each biological replicate/block
    and combined by weighting each block by the number of possible A-B
    pairs. Cross-replicate comparisons are not made.

    Sign convention:
        positive delta -> group B tends to be higher than group A
        negative delta -> group B tends to be lower than group A
    """

    total_u_A = 0.0
    total_pairs = 0

    for _, sub in df_long.groupby("replicate_id"):

        sub = sub[
            sub["group"].isin([groupA, groupB])
        ]

        if len(sub) == 0:
            continue

        values = sub["value"].to_numpy(dtype=float)
        labels = sub["group"].to_numpy()

        gA_mask = labels == groupA

        nA = int(gA_mask.sum())
        nB = int((labels == groupB).sum())

        if nA == 0 or nB == 0:
            continue

        total_u_A += mannwhitney_u(
            values,
            gA_mask
        )

        total_pairs += nA * nB

    if total_pairs == 0:

        return np.nan

    delta_B_minus_A = 1 - (2 * total_u_A / total_pairs)

    return float(delta_B_minus_A)




def median_difference_from_values(values_A, values_B):
    """
    Compute median difference for a pairwise contrast.

    Sign convention:
        positive value -> group B median is higher than group A median
        negative value -> group B median is lower than group A median
    """

    values_A = np.asarray(values_A, dtype=float)
    values_B = np.asarray(values_B, dtype=float)

    values_A = values_A[np.isfinite(values_A)]
    values_B = values_B[np.isfinite(values_B)]

    if len(values_A) == 0 or len(values_B) == 0:

        return np.nan

    return float(
        np.median(values_B) - np.median(values_A)
    )


def stratified_median_difference(
    df_long,
    groupA,
    groupB
):
    """
    Compute a block-aware median difference for a pairwise contrast.

    For each biological replicate/block, the median of group B is compared
    with the median of group A. The reported value is the median of these
    within-replicate differences.

    Sign convention:
        positive value -> group B tends to have a higher median than group A
        negative value -> group B tends to have a lower median than group A
    """

    differences = []

    for _, sub in df_long.groupby("replicate_id"):

        sub = sub[
            sub["group"].isin([groupA, groupB])
        ]

        values_A = sub.loc[
            sub["group"] == groupA,
            "value"
        ].to_numpy(dtype=float)

        values_B = sub.loc[
            sub["group"] == groupB,
            "value"
        ].to_numpy(dtype=float)

        values_A = values_A[np.isfinite(values_A)]
        values_B = values_B[np.isfinite(values_B)]

        if len(values_A) == 0 or len(values_B) == 0:
            continue

        differences.append(
            np.median(values_B) - np.median(values_A)
        )

    if len(differences) == 0:

        return np.nan

    return float(
        np.median(differences)
    )


def van_elteren_test(
    df_long,
    groupA,
    groupB,
    n_perm=10000,
    seed=0,
    progress_callback=None,
    progress_update_interval=None
):
    """
    Stratified Wilcoxon / van Elteren-style permutation test.

    progress_callback, when provided, is called periodically during
    permutations so the GUI can refresh elapsed time/status while the
    analysis is running.
    """

    rng = np.random.default_rng(seed)

    if progress_update_interval is None:

        progress_update_interval = max(
            1,
            n_perm // 100
        )

    observed = 0.0

    replicate_subsets = []

    for _, sub in df_long.groupby("replicate_id"):

        sub = sub[
            sub["group"].isin([groupA, groupB])
        ]

        if len(sub) == 0:
            continue

        x = sub["value"].to_numpy()

        groups = sub["group"].to_numpy()

        gA = groups == groupA

        if gA.sum() == 0:
            continue

        observed += mannwhitney_u(x, gA)

        replicate_subsets.append(
            (x, int(gA.sum()))
        )

    perms = []

    last_progress_report = 0

    for perm_index in range(n_perm):

        total = 0.0

        for x, nA in replicate_subsets:

            idx = np.arange(len(x))

            rng.shuffle(idx)

            mask = np.zeros(
                len(x),
                dtype=bool
            )

            mask[idx[:nA]] = True

            total += mannwhitney_u(x, mask)

        perms.append(total)

        if progress_callback is not None:

            completed = perm_index + 1

            if (
                completed % progress_update_interval == 0
                or completed == n_perm
            ):

                progress_callback(
                    completed - last_progress_report
                )

                last_progress_report = completed

    perms = np.array(perms)

    center = perms.mean()

    p = (
        np.sum(
            np.abs(perms - center)
            >= np.abs(observed - center)
        ) + 1
    ) / (n_perm + 1)

    return observed, p



def kruskal_h_from_ranks(
    ranks,
    labels,
    group_names
):
    """
    Compute a Kruskal-Wallis-style H statistic from precomputed ranks.

    Ranks are computed within one replicate/block. The statistic is used
    as the block-level component of the global stratified rank test.
    """

    n_total = len(ranks)

    if n_total == 0:

        return 0.0

    group_terms = 0.0
    n_groups_present = 0

    for group_name in group_names:

        mask = labels == group_name
        n_group = int(mask.sum())

        if n_group == 0:

            continue

        mean_rank = float(ranks[mask].mean())

        group_terms += n_group * (
            mean_rank - (n_total + 1) / 2
        ) ** 2

        n_groups_present += 1

    if n_groups_present < 2:

        return 0.0

    h_stat = (
        12 / (n_total * (n_total + 1))
    ) * group_terms

    return float(h_stat)


def global_stratified_rank_test(
    df_long,
    groups,
    n_perm=10000,
    seed=0
):
    """
    Omnibus stratified rank permutation test across all selected groups.

    The statistic is the sum of Kruskal-Wallis-style rank statistics
    calculated within each biological replicate/block. During permutation,
    group labels are shuffled within each replicate, preserving the
    original number of observations per group within that replicate.

    This provides a global screen before post hoc pairwise stratified
    Wilcoxon tests.
    """

    rng = np.random.default_rng(seed)

    groups = list(groups)

    replicate_subsets = []
    observed = 0.0

    for _, sub in df_long.groupby("replicate_id"):

        sub = sub[
            sub["group"].isin(groups)
        ].copy()

        if sub["group"].nunique() < 2:

            continue

        values = sub["value"].to_numpy(dtype=float)
        labels = sub["group"].to_numpy()
        ranks = rankdata(values, method="average")

        observed += kruskal_h_from_ranks(
            ranks,
            labels,
            groups
        )

        replicate_subsets.append(
            (ranks, labels.copy())
        )

    if len(replicate_subsets) == 0:

        return np.nan, np.nan

    perms = []

    for _ in range(n_perm):

        total = 0.0

        for ranks, labels in replicate_subsets:

            permuted_labels = labels.copy()

            rng.shuffle(permuted_labels)

            total += kruskal_h_from_ranks(
                ranks,
                permuted_labels,
                groups
            )

        perms.append(total)

    perms = np.array(perms)

    p = (
        np.sum(perms >= observed) + 1
    ) / (n_perm + 1)

    return observed, float(p)


# ============================================================
# LMM
# ============================================================

def run_lmm(
    df_long,
    contrasts,
    log_transform=False
):

    df = df_long.copy()

    if log_transform:

        df["response"] = np.log(df["value"])

    else:

        df["response"] = df["value"]

    rows = []

    raw_p = []

    for A, B in contrasts:

        sub = df[
            df["group"].isin([A, B])
        ].copy()

        delta = stratified_cliffs_delta(
            df,
            A,
            B
        )

        median_diff = stratified_median_difference(
            df,
            A,
            B
        )

        # --------------------------------------------
        # REQUIRE MULTIPLE REPLICATES
        # --------------------------------------------

        if sub["replicate_id"].nunique() < 2:

            rows.append({

                "Contrast":
                    f"{A} vs {B}",

                "Method":
                    "LMM failed: <2 replicate levels",

                "Cliffs_delta_B_minus_A":
                    delta,

                "Median_difference_B_minus_A":
                    median_diff

            })

            raw_p.append(np.nan)

            continue

        sub["group"] = (
            sub["group"]
            .astype("category")
        )

        try:

            model = smf.mixedlm(
                "response ~ group",
                data=sub,
                groups=sub["replicate_id"]
            )

            fit = model.fit(reml=False)

            # ----------------------------------------
            # ROBUST EXTRACTION
            # ----------------------------------------

            beta = float(
                fit.params.iloc[1]
            )

            pval = float(
                fit.pvalues.iloc[1]
            )

            raw_p.append(pval)

            rows.append({

                "Contrast":
                    f"{A} vs {B}",

                "Method":
                    "Linear Mixed Model",

                "Cliffs_delta_B_minus_A":
                    delta,

                "Median_difference_B_minus_A":
                    median_diff,

                "p_raw":
                    pval

            })

        except Exception as e:

            rows.append({

                "Contrast":
                    f"{A} vs {B}",

                "Method":
                    f"LMM failed: {e}",

                "Cliffs_delta_B_minus_A":
                    delta,

                "Median_difference_B_minus_A":
                    median_diff

            })

            raw_p.append(np.nan)

    # ========================================================
    # MULTIPLE TESTING CORRECTIONS
    # ========================================================

    raw_p_array = np.array(raw_p)

    valid = np.isfinite(raw_p_array)

    holm_adj = np.full(len(raw_p), np.nan)
    bonf_adj = np.full(len(raw_p), np.nan)
    fdr_adj = np.full(len(raw_p), np.nan)

    if np.sum(valid) > 0:

        _, p_holm, _, _ = multipletests(
            raw_p_array[valid],
            method="holm"
        )

        _, p_bonf, _, _ = multipletests(
            raw_p_array[valid],
            method="bonferroni"
        )

        _, p_fdr, _, _ = multipletests(
            raw_p_array[valid],
            method="fdr_bh"
        )

        holm_adj[valid] = p_holm
        bonf_adj[valid] = p_bonf
        fdr_adj[valid] = p_fdr

    for i in range(len(rows)):

        rows[i]["p_bonferroni"] = bonf_adj[i]
        rows[i]["p_holm"] = holm_adj[i]
        rows[i]["p_fdr_bh"] = fdr_adj[i]

    return pd.DataFrame(rows)



# ============================================================
# FIXED-BLOCK LINEAR MODEL ON REPLICATE MEDIANS
# ============================================================

def run_fixed_block_lm(
    df_long,
    contrasts,
    log_transform=False
):
    """
    Run a fixed-block linear model on replicate-level medians.

    Model for each pairwise contrast:
        summary_value ~ group_code + C(replicate_id)

    Here replicate_id is treated as a fixed blocking factor
    instead of a random effect.
    """

    summary = summarize_replicate_medians(
        df_long,
        log_transform=log_transform
    )

    rows = []

    raw_p = []

    for A, B in contrasts:

        sub = summary[
            summary["group"].isin([A, B])
        ].copy()

        complete_replicates = (
            sub.groupby("replicate_id")["group"]
            .nunique()
        )

        complete_replicates = complete_replicates[
            complete_replicates == 2
        ].index

        sub = sub[
            sub["replicate_id"].isin(complete_replicates)
        ].copy()

        delta = cliffs_delta_from_values(
            sub.loc[sub["group"] == A, "summary_value"],
            sub.loc[sub["group"] == B, "summary_value"]
        )

        median_diff = median_difference_from_values(
            sub.loc[sub["group"] == A, "summary_value"],
            sub.loc[sub["group"] == B, "summary_value"]
        )

        if sub["replicate_id"].nunique() < 2:

            rows.append({

                "Contrast":
                    f"{A} vs {B}",

                "Method":
                    "Fixed-block LM failed: <2 complete replicate blocks",

                "Cliffs_delta_B_minus_A":
                    delta,

                "Median_difference_B_minus_A":
                    median_diff,

                "Summary":
                    "replicate median",

                "n_replicates":
                    sub["replicate_id"].nunique()

            })

            raw_p.append(np.nan)

            continue

        sub["group_code"] = (
            sub["group"] == B
        ).astype(int)

        try:

            model = smf.ols(
                "summary_value ~ group_code + C(replicate_id)",
                data=sub
            )

            fit = model.fit()

            beta = float(
                fit.params["group_code"]
            )

            pval = float(
                fit.pvalues["group_code"]
            )

            rows.append({

                "Contrast":
                    f"{A} vs {B}",

                "Method":
                    "Fixed-block linear model",

                "Cliffs_delta_B_minus_A":
                    delta,

                "Median_difference_B_minus_A":
                    median_diff,

                "Summary":
                    "replicate median",

                "n_replicates":
                    sub["replicate_id"].nunique(),

                "coefficient_B_minus_A":
                    beta,

                "p_raw":
                    pval

            })

            raw_p.append(pval)

        except Exception as e:

            rows.append({

                "Contrast":
                    f"{A} vs {B}",

                "Method":
                    f"Fixed-block LM failed: {e}",

                "Cliffs_delta_B_minus_A":
                    delta,

                "Median_difference_B_minus_A":
                    median_diff,

                "Summary":
                    "replicate median",

                "n_replicates":
                    sub["replicate_id"].nunique()

            })

            raw_p.append(np.nan)

    rows = add_multiple_testing_corrections(
        rows,
        raw_p
    )

    return pd.DataFrame(rows)


# ============================================================
# SCROLLABLE GUI FRAME
# ============================================================

class ScrollableFrame(ttk.Frame):
    """
    A reusable scrollable frame for GUI tabs containing many rows.

    This is used for the Mapping and Contrasts tabs so that datasets
    with many conditions or many pairwise contrasts remain accessible.
    """

    def __init__(self, parent):

        super().__init__(parent)

        self.canvas = tk.Canvas(
            self,
            borderwidth=0,
            highlightthickness=0
        )

        self.v_scrollbar = ttk.Scrollbar(
            self,
            orient="vertical",
            command=self.canvas.yview
        )

        self.scrollable_frame = ttk.Frame(
            self.canvas
        )

        self.window_id = self.canvas.create_window(
            (0, 0),
            window=self.scrollable_frame,
            anchor="nw"
        )

        self.canvas.configure(
            yscrollcommand=self.v_scrollbar.set
        )

        self.canvas.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        self.v_scrollbar.grid(
            row=0,
            column=1,
            sticky="ns"
        )

        self.rowconfigure(
            0,
            weight=1
        )

        self.columnconfigure(
            0,
            weight=1
        )

        self.scrollable_frame.bind(
            "<Configure>",
            self._update_scroll_region
        )

        self.canvas.bind(
            "<Configure>",
            self._resize_inner_frame
        )

        self.canvas.bind(
            "<Enter>",
            self._bind_mousewheel
        )

        self.canvas.bind(
            "<Leave>",
            self._unbind_mousewheel
        )

    def _update_scroll_region(self, event=None):

        self.canvas.configure(
            scrollregion=self.canvas.bbox("all")
        )

    def _resize_inner_frame(self, event):

        self.canvas.itemconfigure(
            self.window_id,
            width=event.width
        )

    def _bind_mousewheel(self, event=None):

        self.canvas.bind_all(
            "<MouseWheel>",
            self._on_mousewheel
        )

        self.canvas.bind_all(
            "<Button-4>",
            self._on_mousewheel
        )

        self.canvas.bind_all(
            "<Button-5>",
            self._on_mousewheel
        )

    def _unbind_mousewheel(self, event=None):

        self.canvas.unbind_all(
            "<MouseWheel>"
        )

        self.canvas.unbind_all(
            "<Button-4>"
        )

        self.canvas.unbind_all(
            "<Button-5>"
        )

    def _on_mousewheel(self, event):

        if getattr(event, "num", None) == 4:

            delta = -1

        elif getattr(event, "num", None) == 5:

            delta = 1

        else:

            delta = -int(event.delta / 120)

            if delta == 0:

                delta = -1 if event.delta > 0 else 1

        self.canvas.yview_scroll(
            delta,
            "units"
        )


# ============================================================
# TIMER UTILITIES
# ============================================================

def format_elapsed_time(seconds):
    """
    Format elapsed time as HH:MM:SS for the GUI timer.
    """

    seconds = int(max(0, seconds))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:

        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


# ============================================================
# GUI
# ============================================================

class HierarchicalStatisticalAnalyzerGUI(tk.Tk):

    def __init__(self):

        super().__init__()

        self.title(
            f"Hierarchical Statistical Analyzer {APP_VERSION}"
        )

        self.geometry("1300x900")

        self.df = None
        self.input_path = None

        self.group_vars = {}
        self.repl_vars = {}
        self.contrast_vars = {}

        self.engine_vars = {
            "Stratified Wilcoxon": tk.BooleanVar(value=True),
            "Linear Mixed Model": tk.BooleanVar(value=True),
            "Fixed-block linear model": tk.BooleanVar(value=False)
        }

        self.log_var = tk.BooleanVar(
            value=False
        )

        self.status_var = tk.StringVar(
            value="Ready"
        )

        self.elapsed_var = tk.StringVar(
            value="Elapsed time: 00:00"
        )

        self.analysis_start_time = None
        self.last_timer_update = 0
        self.timer_job = None
        self.worker_queue = queue.Queue()
        self.analysis_thread = None

        self._build_gui()

    def _build_gui(self):

        nb = ttk.Notebook(self)

        nb.pack(fill=tk.BOTH, expand=True)

        self.tab_input = ttk.Frame(nb)
        self.tab_mapping = ttk.Frame(nb)
        self.tab_contrast = ttk.Frame(nb)
        self.tab_options = ttk.Frame(nb)
        self.tab_run = ttk.Frame(nb)

        nb.add(self.tab_input, text="1) Input")
        nb.add(self.tab_mapping, text="2) Mapping")
        nb.add(self.tab_contrast, text="3) Contrasts")
        nb.add(self.tab_options, text="4) Options")
        nb.add(self.tab_run, text="5) Run")

        self.build_input()
        self.build_mapping()
        self.build_contrasts()
        self.build_options()
        self.build_run()

    def build_input(self):

        ttk.Button(
            self.tab_input,
            text="Open data file (.csv or .xlsx)",
            command=self.open_data_file
        ).pack(padx=20, pady=(20, 5))

        ttk.Label(
            self.tab_input,
            text=f"Supported input formats: .csv and .xlsx | {APP_VERSION}"
        ).pack(padx=20, pady=(0, 20))

        self.columns_box = tk.Listbox(
            self.tab_input,
            width=100,
            height=25
        )

        self.columns_box.pack(
            padx=20,
            pady=20
        )

    def open_data_file(self):

        path = filedialog.askopenfilename(
            title="Open CSV or XLSX file",
            filetypes=[
                ("CSV or Excel files", "*.csv *.xlsx"),
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        file_path = Path(path)
        suffix = file_path.suffix.lower()

        try:

            if suffix == ".csv":

                self.df = pd.read_csv(file_path)

            elif suffix == ".xlsx":

                self.df = pd.read_excel(
                    file_path,
                    engine="openpyxl"
                )

            else:

                messagebox.showerror(
                    "Unsupported file",
                    "Please select a .csv or .xlsx file."
                )

                return

        except ImportError:

            messagebox.showerror(
                "Missing dependency",
                "Reading .xlsx files requires openpyxl. Install it with: pip install openpyxl"
            )

            return

        except Exception as e:

            messagebox.showerror(
                "File loading error",
                f"Could not load the selected file:\n{e}"
            )

            return

        self.input_path = file_path

        self.columns_box.delete(0, tk.END)

        for c in self.df.columns:

            self.columns_box.insert(tk.END, c)

        self.refresh_mapping()
        self.refresh_contrasts()

    def open_csv(self):
        # Backwards-compatible alias for older callback names.
        self.open_data_file()

    def build_mapping(self):

        self.mapping_scroll = ScrollableFrame(
            self.tab_mapping
        )

        self.mapping_scroll.pack(
            fill=tk.BOTH,
            expand=True,
            padx=20,
            pady=20
        )

        self.mapping_frame = self.mapping_scroll.scrollable_frame

    def refresh_mapping(self):

        for w in self.mapping_frame.winfo_children():
            w.destroy()

        self.group_vars = {}
        self.repl_vars = {}

        if self.df is None:
            return

        ttk.Label(
            self.mapping_frame,
            text="Input column",
            width=40
        ).grid(row=0, column=0, sticky="w", padx=5, pady=(0, 8))

        ttk.Label(
            self.mapping_frame,
            text="Group / condition",
            width=20
        ).grid(row=0, column=1, sticky="w", padx=5, pady=(0, 8))

        ttk.Label(
            self.mapping_frame,
            text="Replicate ID",
            width=20
        ).grid(row=0, column=2, sticky="w", padx=5, pady=(0, 8))

        for i, col in enumerate(self.df.columns, start=1):

            ttk.Label(
                self.mapping_frame,
                text=col,
                width=40
            ).grid(row=i, column=0, sticky="w", padx=5, pady=2)

            g = tk.StringVar(
                value=col.split("_")[0]
            )

            r = tk.StringVar(
                value=col.split("_")[-1]
            )

            self.group_vars[col] = g
            self.repl_vars[col] = r

            ttk.Entry(
                self.mapping_frame,
                textvariable=g,
                width=20
            ).grid(row=i, column=1, sticky="w", padx=5, pady=2)

            ttk.Entry(
                self.mapping_frame,
                textvariable=r,
                width=20
            ).grid(row=i, column=2, sticky="w", padx=5, pady=2)

    def build_contrasts(self):

        ttk.Button(
            self.tab_contrast,
            text="Update contrasts from mapping",
            command=self.refresh_contrasts
        ).pack(anchor="w", padx=20, pady=(20, 5))

        self.contrast_scroll = ScrollableFrame(
            self.tab_contrast
        )

        self.contrast_scroll.pack(
            fill=tk.BOTH,
            expand=True,
            padx=20,
            pady=20
        )

        self.contrast_frame = self.contrast_scroll.scrollable_frame

    def refresh_contrasts(self):

        for w in self.contrast_frame.winfo_children():
            w.destroy()

        self.contrast_vars = {}

        if self.df is None:
            return

        groups = []

        for col in self.df.columns:

            group_name = self.group_vars[col].get().strip()

            if group_name and group_name not in groups:

                groups.append(group_name)

        groups = sorted(groups)

        row = 0

        for i in range(len(groups)):

            for j in range(i + 1, len(groups)):

                A = groups[i]
                B = groups[j]

                var = tk.BooleanVar(
                    value=True
                )

                self.contrast_vars[(A, B)] = var

                ttk.Checkbutton(
                    self.contrast_frame,
                    text=f"{A} vs {B}",
                    variable=var
                ).grid(row=row, column=0, sticky="w")

                row += 1

    def build_options(self):

        frm = self.tab_options

        ttk.Label(
            frm,
            text="Statistical engines"
        ).pack(anchor="w", padx=20, pady=(20, 5))

        ttk.Label(
            frm,
            text="Select one or multiple tests to run at the same time."
        ).pack(anchor="w", padx=20, pady=(0, 10))

        for engine_name, engine_var in self.engine_vars.items():

            ttk.Checkbutton(
                frm,
                text=engine_name,
                variable=engine_var
            ).pack(anchor="w", padx=40, pady=2)

        ttk.Separator(
            frm,
            orient="horizontal"
        ).pack(fill="x", padx=20, pady=20)

        ttk.Checkbutton(
            frm,
            text="Log-transform values",
            variable=self.log_var
        ).pack(anchor="w", padx=20, pady=5)


    def build_run(self):

        control_frame = ttk.Frame(
            self.tab_run
        )

        control_frame.pack(
            fill="x",
            padx=20,
            pady=(20, 10)
        )

        self.run_button = ttk.Button(
            control_frame,
            text="Run Analysis",
            command=self.run_analysis
        )

        self.run_button.pack(
            anchor="w",
            pady=(0, 10)
        )

        ttk.Label(
            control_frame,
            textvariable=self.status_var
        ).pack(
            anchor="w",
            pady=(0, 5)
        )

        ttk.Label(
            control_frame,
            textvariable=self.elapsed_var
        ).pack(
            anchor="w",
            pady=(0, 5)
        )

        self.progress_bar = ttk.Progressbar(
            control_frame,
            mode="indeterminate",
            length=500
        )

        self.progress_bar.pack(
            anchor="w",
            fill="x"
        )

        output_frame = ttk.Frame(
            self.tab_run
        )

        output_frame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=20,
            pady=20
        )

        self.output = tk.Text(
            output_frame,
            width=180,
            height=40,
            wrap="none"
        )

        output_y_scroll = ttk.Scrollbar(
            output_frame,
            orient="vertical",
            command=self.output.yview
        )

        output_x_scroll = ttk.Scrollbar(
            output_frame,
            orient="horizontal",
            command=self.output.xview
        )

        self.output.configure(
            yscrollcommand=output_y_scroll.set,
            xscrollcommand=output_x_scroll.set
        )

        self.output.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        output_y_scroll.grid(
            row=0,
            column=1,
            sticky="ns"
        )

        output_x_scroll.grid(
            row=1,
            column=0,
            sticky="ew"
        )

        output_frame.rowconfigure(
            0,
            weight=1
        )

        output_frame.columnconfigure(
            0,
            weight=1
        )

    def run_analysis(self):
        """
        Validate GUI input and start the statistical analysis in a
        background thread.

        Tkinter can only refresh labels, progress bars, and timers while
        its main event loop is free. Long permutation tests therefore need
        to run outside the main GUI thread.
        """

        if self.df is None:

            messagebox.showerror(
                "No input file",
                "Please open a CSV or Excel file first."
            )

            return

        col_to_group = {
            c: self.group_vars[c].get().strip()
            for c in self.df.columns
        }

        col_to_repl = {
            c: self.repl_vars[c].get().strip()
            for c in self.df.columns
        }

        if any(not group for group in col_to_group.values()):

            messagebox.showerror(
                "Missing group name",
                "All columns must have a group name in the Mapping tab."
            )

            return

        if any(not repl for repl in col_to_repl.values()):

            messagebox.showerror(
                "Missing replicate ID",
                "All columns must have a replicate ID in the Mapping tab."
            )

            return

        contrasts = []

        current_groups = set(col_to_group.values())

        for (A, B), var in self.contrast_vars.items():

            if var.get():

                if A not in current_groups or B not in current_groups:

                    messagebox.showerror(
                        "Outdated contrasts",
                        "The contrast list does not match the Mapping tab. Click 'Update contrasts from mapping' and run the analysis again."
                    )

                    return

                contrasts.append((A, B))

        if not contrasts:

            messagebox.showerror(
                "No contrast selected",
                "Please select at least one contrast before running the analysis."
            )

            return

        selected_engines = [
            engine_name
            for engine_name, engine_var in self.engine_vars.items()
            if engine_var.get()
        ]

        if not selected_engines:

            messagebox.showerror(
                "No statistical engine selected",
                "Please select at least one statistical engine before running the analysis."
            )

            return

        # Clear any old messages left in the queue.
        while not self.worker_queue.empty():

            try:

                self.worker_queue.get_nowait()

            except queue.Empty:

                break

        self.analysis_start_time = time.monotonic()
        self.last_timer_update = 0

        self.status_var.set(
            "Preparing analysis..."
        )

        self.elapsed_var.set(
            "Elapsed time: 00:00"
        )

        self.output.delete("1.0", tk.END)

        self.run_button.configure(
            state="disabled"
        )

        self.progress_bar.start(15)
        self._schedule_elapsed_timer()

        # Copy the dataframe before moving into the worker thread.
        # This keeps the background analysis isolated from GUI state.
        df_copy = self.df.copy()
        input_path = self.input_path
        log_transform = self.log_var.get()

        self.analysis_thread = threading.Thread(
            target=self._analysis_worker,
            kwargs={
                "df_wide": df_copy,
                "col_to_group": col_to_group,
                "col_to_repl": col_to_repl,
                "contrasts": contrasts,
                "selected_engines": selected_engines,
                "log_transform": log_transform,
                "input_path": input_path,
            },
            daemon=True
        )

        self.analysis_thread.start()
        self.after(100, self._poll_worker_queue)

    def _schedule_elapsed_timer(self):
        """
        Update the elapsed-time label from the Tkinter main thread.
        This keeps moving while the analysis runs in a worker thread.
        """

        if self.analysis_start_time is None:

            return

        elapsed = time.monotonic() - self.analysis_start_time

        self.elapsed_var.set(
            f"Elapsed time: {format_elapsed_time(elapsed)}"
        )

        self.timer_job = self.after(
            500,
            self._schedule_elapsed_timer
        )

    def _stop_elapsed_timer(self):
        """
        Stop the scheduled timer callback safely.
        """

        if self.timer_job is not None:

            self.after_cancel(
                self.timer_job
            )

            self.timer_job = None

        if self.analysis_start_time is not None:

            elapsed = time.monotonic() - self.analysis_start_time

            self.elapsed_var.set(
                f"Elapsed time: {format_elapsed_time(elapsed)}"
            )

    def _analysis_worker(
        self,
        df_wide,
        col_to_group,
        col_to_repl,
        contrasts,
        selected_engines,
        log_transform,
        input_path
    ):
        """
        Run the statistics outside the Tkinter main thread.

        This method must not directly modify Tkinter widgets. It sends
        status/results/errors back to the main thread through worker_queue.
        """

        try:

            n_perm = 10000

            self.worker_queue.put((
                "status",
                "Reshaping data..."
            ))

            df_long = reshape_long(
                df_wide,
                col_to_group,
                col_to_repl
            )

            all_results = []

            # ====================================================
            # STRATIFIED WILCOXON
            # ====================================================

            if "Stratified Wilcoxon" in selected_engines:

                rows = []

                raw_p = []

                selected_groups = sorted({
                    group
                    for contrast in contrasts
                    for group in contrast
                })

                self.worker_queue.put((
                    "status",
                    "Running global stratified rank test..."
                ))

                global_stat, global_p = global_stratified_rank_test(
                    df_long,
                    selected_groups,
                    n_perm=n_perm
                )

                rows.append({

                    "Contrast":
                        "Global",

                    "Method":
                        "Global stratified rank test",

                    "Cliffs_delta_B_minus_A":
                        np.nan,

                    "Median_difference_B_minus_A":
                        np.nan,

                    "p_raw":
                        global_p,

                    "p_bonferroni":
                        np.nan,

                    "p_holm":
                        np.nan,

                    "p_fdr_bh":
                        np.nan

                })

                for contrast_index, (A, B) in enumerate(
                    contrasts,
                    start=1
                ):

                    self.worker_queue.put((
                        "status",
                        f"Running Stratified Wilcoxon: {A} vs {B} "
                        f"({contrast_index}/{len(contrasts)})"
                    ))

                    U, p = van_elteren_test(
                        df_long,
                        A,
                        B,
                        n_perm=n_perm
                    )

                    delta = stratified_cliffs_delta(
                        df_long,
                        A,
                        B
                    )

                    median_diff = stratified_median_difference(
                        df_long,
                        A,
                        B
                    )

                    raw_p.append(p)

                    rows.append({

                        "Contrast":
                            f"{A} vs {B}",

                        "Method":
                            "Stratified Wilcoxon",

                        "Cliffs_delta_B_minus_A":
                            delta,

                        "Median_difference_B_minus_A":
                            median_diff,

                        "p_raw":
                            p

                    })

                posthoc_rows = rows[1:]

                posthoc_rows = add_multiple_testing_corrections(
                    posthoc_rows,
                    raw_p
                )

                rows = [
                    rows[0]
                ] + posthoc_rows

                wilcox_df = pd.DataFrame(rows)

                all_results.append(wilcox_df)

            # ====================================================
            # LMM
            # ====================================================

            if "Linear Mixed Model" in selected_engines:

                self.worker_queue.put((
                    "status",
                    "Running Linear Mixed Model..."
                ))

                lmm_df = run_lmm(
                    df_long,
                    contrasts,
                    log_transform=log_transform
                )

                all_results.append(lmm_df)

            # ====================================================
            # FIXED-BLOCK LINEAR MODEL
            # ====================================================

            if "Fixed-block linear model" in selected_engines:

                self.worker_queue.put((
                    "status",
                    "Running Fixed-block linear model..."
                ))

                fixed_block_df = run_fixed_block_lm(
                    df_long,
                    contrasts,
                    log_transform=log_transform
                )

                all_results.append(fixed_block_df)

            results = pd.concat(
                all_results,
                ignore_index=True
            )

            results = simplify_output_columns(results)

            results.insert(
                0,
                "Analysis",
                "Contrast test"
            )

            if "Column" not in results.columns:

                results.insert(
                    1,
                    "Column",
                    ""
                )

            if "Normality_decision" not in results.columns:

                results["Normality_decision"] = ""

            self.worker_queue.put((
                "status",
                "Running residual normality tests..."
            ))

            normality_df = run_column_residual_normality(
                df_wide
            )

            results = pd.concat(
                [
                    results,
                    normality_df
                ],
                ignore_index=True
            )

            output_columns = [
                "Analysis",
                "Column",
                "Contrast",
                "Method",
                "p_raw",
                "p_bonferroni",
                "p_holm",
                "p_fdr_bh",
                "Normality_decision",
                "Cliffs_delta_B_minus_A",
                "Median_difference_B_minus_A",
                "Median",
                "Mean"
            ]

            for col in output_columns:

                if col not in results.columns:

                    results[col] = np.nan

            results = results[output_columns]

            if input_path is not None:

                output_path = (
                    input_path.parent
                    / f"{input_path.stem}_statistical-analysis.csv"
                )

            else:

                output_path = Path("statistical-analysis.csv")

            results.to_csv(
                output_path,
                index=False
            )

            self.worker_queue.put((
                "done",
                results,
                output_path
            ))

        except Exception as e:

            self.worker_queue.put((
                "error",
                str(e)
            ))

    def _poll_worker_queue(self):
        """
        Receive worker-thread status/results/errors and update the GUI
        from the main Tkinter thread.
        """

        try:

            while True:

                message = self.worker_queue.get_nowait()

                message_type = message[0]

                if message_type == "status":

                    self.status_var.set(
                        message[1]
                    )

                elif message_type == "done":

                    _, results, output_path = message

                    self._finish_successful_analysis(
                        results,
                        output_path
                    )

                    return

                elif message_type == "error":

                    self._finish_failed_analysis(
                        message[1]
                    )

                    return

        except queue.Empty:

            pass

        self.after(100, self._poll_worker_queue)

    def _finish_successful_analysis(
        self,
        results,
        output_path
    ):
        """
        Final GUI update after successful analysis.
        """

        self._stop_elapsed_timer()
        self.progress_bar.stop()

        self.output.delete("1.0", tk.END)

        self.output.insert(
            tk.END,
            results.to_string(index=False)
        )

        self.status_var.set(
            "Analysis completed."
        )

        self.run_button.configure(
            state="normal"
        )

        messagebox.showinfo(
            "Done",
            f"Analysis completed. Results saved as:\n{output_path}"
        )

        self.analysis_start_time = None

    def _finish_failed_analysis(
        self,
        error_message
    ):
        """
        Final GUI update after failed analysis.
        """

        self._stop_elapsed_timer()
        self.progress_bar.stop()

        self.status_var.set(
            "Analysis failed."
        )

        self.run_button.configure(
            state="normal"
        )

        messagebox.showerror(
            "Analysis error",
            f"The analysis failed:\n{error_message}"
        )

        self.analysis_start_time = None


if __name__ == "__main__":

    app = HierarchicalStatisticalAnalyzerGUI()

    app.mainloop()
