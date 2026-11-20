"""Load KTAS, describe patients with triage-time fields only, and sample decision sets.

Labels (KTAS_expert) are used here ONLY to keep sets with at least two distinct levels (same rule as the
2026-10-22 pilot). They never reach a prompt: `describe` reads PROMPT_FIELDS only.
"""
import random

import pandas as pd

# Code meanings from the Kaggle data description (verify against the paper's S1 data dictionary)
SEX = {1: "female", 2: "male"}
ARRIVAL = {1: "walk-in", 2: "public ambulance", 3: "private vehicle", 4: "private ambulance"}
MENTAL = {1: "alert", 2: "responds to voice", 3: "responds to pain", 4: "unresponsive"}
INJURY = {1: "no", 2: "yes"}

# Only these may reach a prompt. Never add: Diagnosis in ED, Disposition, Error_group,
# Length of stay_min, mistriage, KTAS_RN, KTAS duration_min, KTAS_expert.
PROMPT_FIELDS = ["Sex", "Age", "Arrival mode", "Injury", "Chief_complain", "Mental", "Pain", "NRS_pain",
                 "SBP", "DBP", "HR", "RR", "BT", "Saturation"]


def _val(x):
    s = str(x).strip()
    return "not recorded" if s in ("", "nan", "??", "#BOÞ!") else s


def load(path):
    df = pd.read_csv(path, sep=";", encoding="latin-1", dtype=str)
    df["pid"] = range(len(df))
    for c in ["KTAS_expert", "KTAS_RN", "Sex", "Arrival mode", "Mental", "Injury", "Pain", "Age"]:
        df[c] = df[c].astype(int)
    return df


def describe(r):
    r = {k: r[k] for k in PROMPT_FIELDS}
    pain = "no pain reported" if r["Pain"] == 0 else f"pain present, score {_val(r['NRS_pain'])}/10"
    return (f"{r['Age']}-year-old {SEX.get(r['Sex'], 'unknown sex')}, arrived by "
            f"{ARRIVAL.get(r['Arrival mode'], 'other means')}, injury: {INJURY.get(r['Injury'])}. "
            f"Chief complaint: {r['Chief_complain'].strip()}. Mental state: {MENTAL.get(r['Mental'])}. "
            f"{pain}. Vitals: BP {_val(r['SBP'])}/{_val(r['DBP'])} mmHg, HR {_val(r['HR'])}/min, "
            f"RR {_val(r['RR'])}/min, temp {_val(r['BT']).replace(',', '.')} C, SpO2 {_val(r['Saturation'])}%.")


def sample_sets(df, n_sets, seed=42, size=5):
    """The pilot sampler, unchanged, so cached pilot answers are reused for the first sets."""
    rng = random.Random(seed)
    pool = df.pid.tolist()
    rng.shuffle(pool)
    sets = []
    while len(sets) < n_sets and len(pool) >= size:
        s = [pool.pop() for _ in range(size)]
        if df.loc[s, "KTAS_expert"].nunique() >= 2:
            sets.append(s)
    return sets
