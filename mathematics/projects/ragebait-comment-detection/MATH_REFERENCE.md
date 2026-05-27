# Math reference

One-page summary of all 18 features, for cross-reference with the LaTeX
report. Notation:

- $x$ — a comment (token sequence $w_1, \dots, w_N$).
- $\theta$ — the reference LM (default `gpt2-medium`).
- $\phi$ — sentence embedder (`paraphrase-multilingual-MiniLM-L12-v2`, 384-D).
- $\text{PP}_\theta(x) = \exp\!\big(-\tfrac{1}{N} \sum_i \log p_\theta(w_i \mid w_{<i})\big)$.
- $\mathcal{H}(N)$ — benign comments of token-length $\sim N$ (training set).

---

## Stage 1 — AI likelihood features (8)

### ψ₁ — length-normalized perplexity z-score
$$
\psi_1(x) = \frac{\log \text{PP}_\theta(x) - \mu_{\mathcal{H}(N)}}{\sigma_{\mathcal{H}(N)}}
$$
$\mu_{\mathcal{H}(N)}, \sigma_{\mathcal{H}(N)}$ are length-bucketed mean / std on the
benign training subset (10-token buckets). Direction: **lower for AI**.
File: `src/features/ai_likelihood.py`.

### ψ₂ — sentence-burstiness coefficient
Let $\pi_j = \log \text{PP}_\theta(s_j)$ for sentence $s_j$.
$$
\psi_2(x) = \frac{\sigma_\pi - \mu_\pi}{\sigma_\pi + \mu_\pi} \in [-1, 1]
$$
(Goh & Barabási 2008). NaN when $x$ has fewer than 2 sentences. Direction:
**lower for AI** (more uniform per-sentence difficulty).

### ψ₃ — DetectGPT curvature z-score
$$
\psi_3(x) = \frac{\log p_\theta(x) - \tfrac{1}{K}\sum_k \log p_\theta(\tilde x_k)}{\hat \sigma_k}
$$
with $\tilde x_k \sim q(\cdot \mid x)$ a T5-small mask-and-fill perturbation
($K = 20$, mask ratio $= 15\%$). LLM text sits at local maxima of
$\log p_\theta$, so $\psi_3 > 0$ for AI. Mitchell et al. (ICML 2023).
We approximate $\log p_\theta(x) \approx -N \log \text{PP}_\theta(x)$.

### ψ₄, ψ₅ — GLTR token-rank features
For each non-initial token $w_t$, its conditional rank is
$r_t = |\{ v : p_\theta(v \mid w_{<t}) \geq p_\theta(w_t \mid w_{<t}) \}|$.
$$
\psi_4(x) = \frac{|\{t : r_t \leq 10\}|}{N-1}, \qquad
\psi_5(x) = \frac{|\{t : r_t \leq 100\}|}{N-1}
$$
Gehrmann, Strobelt & Rush (ACL 2019). Direction: **higher for AI**.

### ψ₆ — LLM lexical fingerprint
$$
\psi_6(x) = \log\!\Big(1 + \tfrac{20 \cdot |\text{AI-phrase hits in } x|}{N}\Big)
$$
A curated lexicon of LLM-tic phrases (`delve into`, `tapestry of`,
`navigate the complexities`, em-dash-no-space, etc.) lives in
`src/features/ai_lexical.py` and is extended by `data/ai_lexicon.txt`.

### ψ₇ — punctuation regularity
Let $f_k(x)$ be the per-sentence rate of punctuation pattern $k$ (em-dash,
semicolon, ellipsis, Oxford comma, multi-`!`, multi-`?`, ALL-CAPS run,
double-space, smart quote). With training-time benign baselines
$(\mu_k^H, \sigma_k^H)$:
$$
\psi_7(x) = \frac{1}{|K|} \sum_k \frac{|f_k(x) - \mu_k^H|}{\sigma_k^H}
$$
Direction: **higher for AI** (profile drifts away from human baseline).

### ψ₈ — hedge density
$$
\psi_8(x) = \frac{100 \cdot |\text{hedge-phrase hits}|}{N}
$$
"arguably", "perhaps", "to some extent", "it may be argued", etc. Hedges
per 100 tokens.

---

## Stage 2 — Ragebait features (10)

Notation: $V(w), A(w), D(w)$ are NRC-VAD valence ([-1, 1]), arousal ([0, 1]),
dominance ([0, 1]) for word $w$. $\overline{V}(x), \overline{A}(x)$ are
means over VAD-covered tokens in $x$.

### χ₁ — arousal × |valence|
$$
\chi_1(x) = \overline{A}(x) \cdot |\overline{V}(x)|
$$
High only when arousal *and* polarization are high together.

### χ₂ — strongly-negative-word fraction
$$
\chi_2(x) = \Pr_{w \sim x}\big(V(w) < -0.5\big)
$$
Denominator is total token count (not VAD-covered token count), so
$\chi_2$ is also a proxy for affect-vocabulary density.

### χ₃ — MFD vice density
$$
\chi_3(x) = \frac{|\{w \in x : w \in \mathcal{M}_{\text{vice}}\}|}{N}
$$
$\mathcal{M}_{\text{vice}}$ is the union of the five MFD2 vice categories
(Harm, Cheating, Betrayal, Subversion, Degradation). Matching is by
stem-prefix to handle the MFD `*` suffix convention.

### χ₄ — windowed outgroup-NEG co-occurrence
For each outgroup-marker token at position $i$ (window radius $W = 5$):
$$
\text{rate}_i = \frac{|\{j : |j-i| \leq W,\ V(w_j) < -0.3\}|}{2W+1}
$$
$$
\chi_4(x) = \text{mean}_i\, \text{rate}_i \quad\text{(NaN if no outgroup tokens)}.
$$
Approximation to the PMI formulation in the report; the windowed form is
more sample-efficient on short comments.

### χ₅ — rhetorical-pattern score
$$
\chi_5(x) = \text{rq}(x) + \text{hyper}(x) + \text{caps}(x) + \text{excl}(x)
$$
where each sub-feature is a per-sentence (or per-token, for `caps`) rate of
rhetorical questions, hyperbole / absolute quantifiers, ALL-CAPS runs of
$\geq 3$ letters, and exclamation marks. Equal weights at extraction; the
logistic regression reweights at training.

### χ₆ — information-to-affect log-ratio
$$
I(x) = \frac{|\text{NE proxy}| + |\text{numerals}| + |\text{quoted spans}|}{N},
\qquad
A(x) = \overline{A}(x) \cdot \big(|\overline{V}(x)| + \rho_{\text{excl}}(x)\big)
$$
$$
\chi_6(x) = \log\!\frac{A(x) + \varepsilon}{I(x) + \varepsilon}
$$
Higher when affect dominates information. NE proxy: capitalized non-initial
words + consecutive Title-Case pairs.

### χ₇ — counterfactual neutralization gap (novel)
$$
R(x) = \min\!\Big(1, \tfrac{|\nu(x)|}{|x|}\Big), \qquad
S(x) = \max\!\big(0, \cos(\phi(x), \phi(\nu(x)))\big)
$$
$$
\chi_7(x) = (1 - R(x)) \cdot (1 - S(x))
$$
$\nu(x)$ is the neutralized rewrite (rule-based default or Flan-T5-base).
Ragebait collapses: $R \to 0, S \to 0$, so $\chi_7 \to 1$. Substantive
content survives: $R, S \to 1$, so $\chi_7 \to 0$. Mirror of the DetectGPT
curvature argument in the **affective** dimension.

### χ₈ — topic-conditional emotion residual
On the benign training subset, fit ridge regression
$$
\overline{A}(x) \approx \beta_0 + \beta^\top \tau(x), \qquad \tau(x) = \phi(\text{parent topic, fallback }x)
$$
with $\lambda = 1.0$. Then at inference:
$$
\chi_8(x) = \overline{A}(x) - (\beta_0 + \beta^\top \tau(x))
$$
Positive residual = more aroused than topic baseline. Eliminates the
confound where a comment about a war casualty is *expected* to be more
aroused than one about gardening.

### χ₉, χ₁₀ — template similarity
Let $\mathcal{R}$ be the ragebait template corpus.
$$
\rho(x) = \max_{r \in \mathcal{R}} \cos(\phi(x), \phi(r))
$$
$$
\chi_9(x) = \rho(x), \qquad
\chi_{10}(x) = \frac{|\{r \in \mathcal{R} : \cos(\phi(x), \phi(r)) > 0.5\}|}{|\mathcal{R}|}
$$
$\chi_9$ is worst-case match; $\chi_{10}$ is template-coverage breadth. The
0.5 threshold is below the 0.75 used as the `explain_red_flags` display
threshold for $\chi_9$: coverage breadth needs a looser bar than worst-case
match.

---

## Classifier fusion

Two independent calibrated logistic regressions:
$$
\hat p_{AI}(x) = \sigma\!\big(a_0 + a^\top \psi(x)\big), \qquad
\hat p_{RB}(x) = \sigma\!\big(b_0 + b^\top \chi(x)\big)
$$
fit with `LogisticRegressionCV` (`Cs=10`, 5-fold internal CV, ROC-AUC scoring),
wrapped in `CalibratedClassifierCV` with `method='sigmoid'` (Platt scaling).
Pipeline: median imputation $\to$ standardization $\to$ regression.

Joint risk: $\hat p_{AR}(x) = \hat p_{AI}(x) \cdot \hat p_{RB}(x)$.

---

## Account-level aggregation

Given per-comment scores $s^{(1)}, \dots, s^{(n)}$ for user $u$, with platform
bot prior $\pi_0$, bot per-comment rate $\eta_+$ and human per-comment rate
$\eta_-$:
$$
\text{logit}\,\Pr(Z_u = 1 \mid s^{(1:n)}) =
    \text{logit}\, \pi_0 +
    \Big(\sum_{i=1}^n s^{(i)}\Big) \cdot
    \log\!\frac{\eta_+(1 - \eta_-)}{\eta_-(1 - \eta_+)}
$$
Defaults: $\pi_0 = 0.05$, $\eta_+ = 0.80$, $\eta_- = 0.05$. Implemented in
`src/aggregate.py`.

---

## References

### Detection methods
1. Mitchell, E., Lee, Y., Khazatsky, A., Manning, C. D., Finn, C. *DetectGPT:
   Zero-Shot Machine-Generated Text Detection using Probability Curvature.*
   ICML 2023. https://arxiv.org/abs/2301.11305
2. Gehrmann, S., Strobelt, H., Rush, A. M. *GLTR: Statistical Detection and
   Visualization of Generated Text.* ACL 2019, System Demonstrations.
   https://aclanthology.org/P19-3019/
3. Goh, K.-I., Barabási, A.-L. *Burstiness and Memory in Complex Systems.*
   Europhysics Letters 81, 48002, 2008. https://arxiv.org/abs/physics/0610233

### Affect, moral, and social-spread theory
4. Mohammad, S. *Obtaining Reliable Human Ratings of Valence, Arousal, and
   Dominance for 20,000 English Words.* ACL 2018.
   https://aclanthology.org/P18-1017/
5. Graham, J., Haidt, J., Nosek, B. A. *Liberals and Conservatives Rely on
   Different Sets of Moral Foundations.* Journal of Personality and Social
   Psychology, 96(5), 1029–1046, 2009.
6. Hopp, F. R., Fisher, J. T., Cornell, D., Huskey, R., Weber, R. *The
   extended Moral Foundations Dictionary (eMFD): Development and Applications
   of a Crowd-Sourced Approach to Extracting Moral Intuitions from Text.*
   Behavior Research Methods 53, 232–246, 2021.
7. Brady, W. J., Wills, J. A., Jost, J. T., Tucker, J. A., Van Bavel, J. J.
   *Emotion shapes the diffusion of moralized content in social networks.*
   PNAS 114(28), 7313–7318, 2017.
8. Crockett, M. J. *Moral outrage in the digital age.* Nature Human Behaviour
   1(11), 769–771, 2017.
9. Vosoughi, S., Roy, D., Aral, S. *The spread of true and false news online.*
   Science 359(6380), 1146–1151, 2018.

### Datasets
10. Guo, B., Zhang, X., Wang, Z., et al. *How Close is ChatGPT to Human
    Experts? Comparison Corpus, Evaluation, and Detection.* arXiv:2301.07597,
    2023. (HC3 dataset.)
11. Borkan, D., Dixon, L., Sorensen, J., Thain, N., Vasserman, L. *Nuanced
    Metrics for Measuring Unintended Bias with Real Data for Text
    Classification.* WWW 2019. (civil_comments dataset.)
12. cjadams, Sorensen, J., Elliott, J., Dixon, L., McDonald, M., et al.
    *Jigsaw Toxic Comment Classification Challenge.* Kaggle, 2017.
    (wiki_toxic subset.)

### Models and machinery
13. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I.
    *Language Models are Unsupervised Multitask Learners.* 2019. (GPT-2.)
14. Raffel, C., Shazeer, N., Roberts, A., et al. *Exploring the Limits of
    Transfer Learning with a Unified Text-to-Text Transformer.* JMLR 21, 2020.
    (T5.)
15. Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., Zhou, M. *MiniLM: Deep
    Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained
    Transformers.* NeurIPS 2020. (paraphrase-multilingual-MiniLM-L12-v2.)
16. Reimers, N., Gurevych, I. *Sentence-BERT: Sentence Embeddings using
    Siamese BERT-Networks.* EMNLP 2019. (sentence-transformers library.)
17. Chung, H. W., Hou, L., Longpre, S., et al. *Scaling Instruction-Finetuned
    Language Models.* arXiv:2210.11416, 2022. (Flan-T5.)
18. Platt, J. *Probabilistic Outputs for Support Vector Machines and
    Comparisons to Regularized Likelihood Methods.* In *Advances in Large
    Margin Classifiers*, 1999. (Platt scaling, used by sklearn's
    `CalibratedClassifierCV`.)
19. Pedregosa, F., Varoquaux, G., Gramfort, A., et al. *Scikit-learn: Machine
    Learning in Python.* JMLR 12, 2825–2830, 2011.

### Self-exciting processes (for §Further Development)
20. Hawkes, A. G. *Spectra of some self-exciting and mutually exciting point
    processes.* Biometrika 58(1), 83–90, 1971.
