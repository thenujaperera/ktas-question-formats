"""Offline checks (no API): aggregation maths, label isolation, and a full mock run. Run: python -m pytest -q"""
import numpy as np

from ktriage import data, prompts
from ktriage.aggregate import ahp_eigen, ahp_from_logodds, bradley_terry, consistency_ratio
from ktriage.experiment import run


def test_consistent_matrix_has_zero_cr_and_right_order():
    w = np.array([0.5, 0.3, 0.2])
    A = w[:, None] / w[None, :]
    est, _ = ahp_eigen(A)
    assert np.allclose(est, w) and abs(consistency_ratio(A)) < 1e-9
    assert list(np.argsort(-ahp_from_logodds(np.log(A)))) == [0, 1, 2]


def test_bradley_terry_orders_by_wins():
    W = np.array([[0, 2, 2], [0, 0, 2], [0, 0, 0]], float)
    assert list(np.argsort(-bradley_terry(W))) == [0, 1, 2]


def test_no_label_reaches_a_prompt():
    df = data.load("data/ktas_data.csv")
    for p in range(0, len(df), 97):
        d = data.describe(df.loc[p])
        text = prompts.direct_one(d) + prompts.pair(d, d)
        assert "KTAS_expert" not in text and str(df.loc[p, "Diagnosis in ED"]) not in text


def test_pilot_prompts_unchanged():
    # the cached pilot answers are keyed on these exact strings
    assert prompts.SYSTEM == ("You are an emergency department triage nurse. "
                              "Judge only from the triage information given.")


def test_mock_run_end_to_end(tmp_path):
    per_set, stop = run("mock", "mock", n_sets=3, arms=None, out_root=str(tmp_path), rpm=0, max_calls=1000)
    assert stop is None
    assert {"direct_set", "direct_each", "pair_bt", "pair_ahp", "pairprob_ahp", "nurse"} <= set(per_set.arm)
    assert per_set.tau.between(-1, 1).all()


def test_decision_backend_offline(tmp_path):
    arms = ["direct_set", "direct_each", "optionwise", "pairprob_bt", "pairprob_ahp"]
    per_set, stop = run("fake-decision", "decision-fake", n_sets=3, arms=arms, out_root=str(tmp_path), rpm=0)
    assert stop is None and set(arms) <= set(per_set.arm)
    assert (per_set[per_set.arm.isin(["direct_each", "pairprob_ahp", "optionwise"])].source == "native").all()
    assert per_set[per_set.arm == "direct_set"].calls.iloc[0] == 5
