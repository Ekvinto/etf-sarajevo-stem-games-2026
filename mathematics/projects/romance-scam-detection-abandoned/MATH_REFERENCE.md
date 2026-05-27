# Math reference — one page per feature

Every feature definition in one place, with the exact form used by the code.
This is a quick-grab reference; the LaTeX report has the derivations.

Notation: a conversation is `C = ((u_i, m_i, t_i))_{i=1..T}` with
`u_i ∈ {V, S}`. Let `C_S = (m_i)_{u_i = S}` and `w(m) = (w_1, ..., w_{|m|})`.

---

## φ₁ — Mean perplexity

```
PP(w) = exp( -1/N · Σᵢ log p_θ(wᵢ | w_<ᵢ) )
φ₁    = mean over m ∈ C_S of PP(w(m))
```
Reference model `p_θ`: GPT-2 medium. Lower for AI text.

## φ₂ — Burstiness of log-perplexity

```
πⱼ = log PP(sⱼ)              for each sentence sⱼ
B  = (σ_π − μ_π) / (σ_π + μ_π)  ∈ [-1, 1]
φ₂ = B
```
(Goh & Barabási, 2008). Lower for AI text.

## φ₃ — DetectGPT-style curvature

```
log p_θ(x) ≈ -N · log PP(x)
d(x) = log p_θ(x) − E_{x̃∼q} [log p_θ(x̃)]
     ≈ -½ tr( Σ_q · ∇² log p_θ(x) )
```
With K = 10 T5-small perturbations per message,
```
d̂(x) = ( log p_θ(x) − mean_k log p_θ(x̃_k) ) / std_k log p_θ(x̃_k)
φ₃   = mean of d̂ over messages with |w(m)| ≥ 20
```
Higher for AI text. (Mitchell et al., 2023.)

## φ₄, φ₅ — Token-rank fractions (GLTR)

```
rᵢ = | { v ∈ V : p_θ(v | w_<ᵢ) ≥ p_θ(wᵢ | w_<ᵢ) } |
φ₄ = #{i : rᵢ ≤ 10}  / N
φ₅ = #{i : rᵢ ≤ 100} / N
```
Higher for AI text. (Gehrmann, Strobelt, Rush; ACL 2019.)

## φ₆ — Moving-average type-token ratio (MATTR)

```
MATTR_W = 1 / (N − W + 1) · Σᵢ Vᵢ(W) / W
```
with W = 100. Length-robust. Higher = more diverse vocabulary.

## φ₇ — Yule's K

```
K = 10⁴ · ( Σᵢ i² V(i, N) − N ) / N²
```
where V(i, N) = number of types occurring exactly i times. Length-independent
diversity statistic. Lower K = more diverse. (Yule, 1944; Tweedie & Baayen, 1998.)

## φ₈ — Zipf-exponent deviation

```
log f(r) = −s log r + c       (fit by OLS on ranks r ∈ [10, 1000])
φ₈ = | ŝ − 1 |
```
Natural English has s ≈ 1. AI text often has flatter distributions.

## φ₉ — Length asymmetry (log ratio)

```
φ₉ = log( (mean_S |w(m)| + 1) / (mean_V |w(m)| + 1) )
```
Scammers send longer, more polished messages.

## φ₁₀ — Timing bimodality coefficient

```
Δᵢ = (t_{S, i} − t_{S, i-1}) in seconds, for consecutive S messages
BC = ( γ₁² + 1 ) / ( γ₂ + 3(n−1)² / ((n−2)(n−3)) )
φ₁₀ = BC
```
γ₁ = skewness, γ₂ = excess kurtosis. Bimodal delays = operator running
multiple chats.

## φ₁₁ — CUSUM peak

Per-message sentiment `s_t ∈ [-1, 1]` (XLM-RoBERTa multilingual). With
baseline `μ₀ = median(s_{1:T/2})`, reference value `κ = 0.25`:
```
G₀ = 0
G_t = max( 0, G_{t-1} + (μ₀ − s_t − κ) )
φ₁₁ = max_t G_t
```
(Page, 1954.) Detects the sentiment dip at the financial ask.

## φ₁₂ — Sentiment OLS slope

```
φ₁₂ = slope of OLS regression of s_t on t
```

## φ₁₃ — Topic-shift KL divergence

Split scammer messages into early half W_E and late half W_L. With Laplace
smoothing (α = 0.5) over vocabulary V':
```
p_E(v) = (count_E(v) + α) / (Σ count_E + α|V'|)
p_L(v) similarly
φ₁₃ = D_KL(p_L ‖ p_E) = Σ_v p_L(v) log( p_L(v) / p_E(v) )
```

## φ₁₄ — Late-conversation finance mass

```
φ₁₄ = Σ_{v ∈ Finance} p_L(v)
```
where `Finance` is the curated lexicon in `data/scam_lexicon.txt`.

## φ₁₅ — HMM log-likelihood ratio

Two CategoricalHMMs over 5 hidden states {Open, Rapport, Isolate, Hook,
Urgency} with upper-triangular transitions, fit by Baum–Welch:
- λ_S on scam conversations
- λ_N on benign conversations

Forward algorithm:
```
α_t(j) = [ Σᵢ α_{t-1}(i) · A_{ij} ] · b_j(m_t)
P(C_S | λ) = Σⱼ α_T(j)
φ₁₅ = log P(C_S | λ_S) − log P(C_S | λ_N)
```

## φ₁₆ — Urgency-state indicator

```
path* = Viterbi(C_S, λ_S)
φ₁₆ = 1 if Urgency ∈ path* else 0
```

## φ₁₇, φ₁₈, φ₁₉ — Template similarity

Sentence-transformer φ : text → ℝ³⁸⁴ (`paraphrase-multilingual-MiniLM-L12-v2`).
Template corpus T ⊂ {confirmed scam messages}, all embeddings unit-normalized.
For each scammer message m:
```
ρ(m) = max_{t ∈ T}  ⟨φ(m), φ(t)⟩  / (‖φ(m)‖ · ‖φ(t)‖)
φ₁₇ = max_m ρ(m)
φ₁₈ = mean_m ρ(m)
φ₁₉ = | { m : ρ(m) > 0.75 } | / |C_S|
```

---

## Classifier

```
φ(C) = standardize( φ₁, …, φ₁₉ )                    using training-set mean/std
p(scam | C) = σ( wᵀ φ + b )                          σ(z) = 1 / (1 + e^{-z})
```

Loss with L2 regularization, λ chosen by 5-fold CV:
```
J(w, b) = -1/n Σ_k [ y_k log σ(wᵀφ_k + b) + (1 − y_k) log(1 − σ(wᵀφ_k + b)) ] + (λ/2) ‖w‖²
```

Median imputation for NaN features (training set medians, persisted in the
sklearn pipeline).

---

## Evaluation metrics

- **ROC AUC** — primary discrimination metric
- **Average precision** — AP from precision–recall curve
- **Precision at 90% recall** — operational metric for high-recall warning
- **Expected calibration error (ECE)** over 10 quantile bins:
  ```
  ECE = Σ_b (n_b / n) · | acc(b) − conf(b) |
  ```
- **Family ablation** — ΔAUC when each feature family is removed (5 folds).
