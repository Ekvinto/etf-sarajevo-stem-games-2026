"""Feature extractors. Each module exposes an extract(conv) -> dict function.

Module             | Features              | Math content
-------------------+-----------------------+-------------------------------------
perplexity.py      | phi_1, phi_2          | Cross-entropy, burstiness
detect_gpt.py      | phi_3                 | Hessian-trace curvature
token_rank.py      | phi_4, phi_5          | Rank histogram (GLTR)
stylometry.py      | phi_6, phi_7, phi_8   | TTR, Yule K, Zipf fit
asymmetry.py       | phi_9, phi_10         | Length ratio, bimodality
sentiment.py       | phi_11, phi_12        | CUSUM, OLS trajectory
topic_shift.py     | phi_13, phi_14        | KL divergence
hmm_stages.py      | phi_15, phi_16        | Forward + Viterbi
semantic.py        | phi_17, phi_18, phi_19| Cosine similarity
"""
