"""Table III: compares our tau-b (tiecredit_methods.csv) with the Kendall values printed in the four papers.
Usage: python scripts/published_rescore/check_published.py [tiecredit_methods.csv]
Published values were typed from the ACL Anthology PDFs (sources next to each block). A value 'matches' if it is
within 0.004 of ours (the published values are rounded to 3 decimals and the authors' code differs in small ways)."""
import os, sys
import pandas as pd

fp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiecredit_methods.csv")
M = pd.read_csv(fp)
TOL = 0.004 + 1e-9

# Wang et al. 2023 (newsum-1.1), Table 1, sample-level Kendall; columns coherence, relevance, consistency, fluency.
WANG_ASP = ["coherence", "relevance", "consistency", "fluency"]
WANG = {"rouge1_f": (0.126, 0.252, 0.130, 0.094), "rouge2_f": (0.139, 0.219, 0.155, 0.128), "rougel_f": (0.099, 0.237, 0.092, 0.084),
        "bert_score_f": (0.211, 0.243, 0.090, 0.158), "mover_score": (0.118, 0.244, 0.127, 0.105), "prism_src_hypo": (0.196, 0.163, 0.285, 0.205),
        "bart_score_cnn_src_hypo": (0.342, 0.273, 0.315, 0.292), "bart_score_para_src_hypo": (0.325, 0.241, 0.332, 0.311),
        "ChatGPT stars": (0.403, 0.374, 0.389, 0.329)}          # 'Stars w/o ref'
WANG_EXTRA = {"bart_score_src_hypo": (0.250, 0.197, 0.256, 0.203)}  # plain BARTScore, not counted in Table III

# Chiang & Lee 2023 (findings-emnlp.599), Table 1 (SummEval) and Table 2 (Topical-Chat), Kendall tau.
CH_S_ASP = ["coherence", "consistency", "fluency", "relevance"]
CH_S = {"geval": (0.359, 0.286, 0.203, 0.327), "geval_no_auto_cot": (0.248, 0.185, 0.177, 0.248),
        "free_text_no_auto_cot": (0.342, 0.334, 0.273, 0.228), "rate_explain": (0.44, 0.337, 0.306, 0.348),
        "analyze_rate": (0.476, 0.34, 0.302, 0.305)}
CH_T_ASP = ["naturalness", "coherence", "engagingness", "groundedness"]
CH_T = {"geval": (0.358, 0.391, 0.513, 0.566), "geval_no_auto_cot": (0.331, 0.404, 0.535, 0.582)}
# (Their Table 2 prints identical rows for rate-explain and analyze-rate; their Table 4 gives rate-explain
#  0.512/0.425/0.509/0.663, which matches ours. Those rows are not counted.)

# Shen et al. 2023 (findings-emnlp.278), Table 4 (RTS, MCQ) and Table 18 (alternative prompts), Kendall, 1,200 summaries.
SH_ASP = ["coherence", "consistency", "fluency", "relevance"]
SHEN = {"ChatGPT rts": (0.312, 0.378, 0.240, 0.357), "ChatGPT mcq": (0.350, 0.320, 0.305, 0.329), "GPT-4 rts": (0.361, 0.522, 0.452, 0.373),
        "ChatGPT altrts": (0.285, 0.350, 0.252, 0.351), "ChatGPT altmcq": (0.355, 0.306, 0.229, 0.206), "ChatGPT stareval": (0.341, 0.264, 0.217, 0.323)}

# G-Eval (emnlp-main.153), Table 1: G-Eval-4 average Kendall tau 0.418; we compare our mean-of-20 average.
GEVAL_AVG = 0.418

def check(study, pub, asps, label, count=True):
    S = M[M.study == study]; n = ok = 0; bad = []
    for m, vals in pub.items():
        for a, p in zip(asps, vals):
            r = S[(S.method == m) & (S.aspect == a)]
            if r.empty: bad.append(f"{m}/{a}: missing"); n += 1; continue
            o = r.tau.iloc[0]; n += 1
            if abs(o - p) <= TOL: ok += 1
            else: bad.append(f"{m}/{a}: ours {o:.3f}, published {p:.3f}")
    print(f"{label}: {ok} of {n} within 0.004" + ("" if count else "  (not counted)"))
    for b in bad: print("    ", b)
    return (n, ok) if count else (0, 0)

tot = [check("Wang et al.", WANG, WANG_ASP, "Wang et al."),
       check("Wang et al.", WANG_EXTRA, WANG_ASP, "Wang et al., plain BARTScore", count=False),
       check("Chiang & Lee SummEval", CH_S, CH_S_ASP, "Chiang & Lee, SummEval"),
       check("Chiang & Lee Topical-Chat", CH_T, CH_T_ASP, "Chiang & Lee, Topical-Chat"),
       check("Shen et al.", SHEN, SH_ASP, "Shen et al.")]
n = sum(t[0] for t in tot); ok = sum(t[1] for t in tot)
g = M[(M.study == "G-Eval (proxy)") & (M.method == "mean")].tau.mean()
print(f"G-Eval: our mean tau-b over 4 aspects {g:.3f}, published {GEVAL_AVG:.3f}")
print(f"TOTAL (three studies): {ok} of {n} published values matched within 0.004")
