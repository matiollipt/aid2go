# Evaluation Metrics & Statistical Foundations

The current `aid2go evaluate` command scores balanced positive and sampled
negative protein-GO links from a random edge split. Proteins can occur in both
training and evaluation. This is a useful internal link-prediction check, but
it is not a time-delayed, protein-held-out CAFA benchmark. Reported AUPR and
Fmax must be interpreted under that sampled-pair protocol.

Evaluating protein function prediction models requires specialized statistical metrics. Standard classification metrics (such as flat accuracy or binary cross-entropy) are inadequate due to:
1. **Extreme Class Imbalance**: Most specific GO terms are annotated to fewer than 0.01% of proteins in a proteome.
2. **Open-World Assumption**: The absence of an experimental annotation does not imply that the protein does not perform that function (it may simply be uncharacterized).
3. **Ontological Hierarchy & Semantic Depth**: Predicting a generic root term (e.g., `protein binding`) provides virtually zero biological insight compared to predicting a specific enzymatic term (e.g., `acetylgalactosaminyl-transferase activity`).

---

## 1. Protein-Centric Precision, Recall & $F_{\max}$

CAFA employs a **protein-centric** evaluation scheme across a sweep of decision thresholds $\tau \in [0, 1]$:

### Mathematical Definitions

For a given decision threshold $\tau$, let:
- $T_i$ be the set of true propagated experimental annotations for protein $i$.
- $\hat{T}_i(\tau) = \{t \in \mathcal{V}_{GO} \mid s_i(t) \ge \tau\}$ be the set of predicted terms for protein $i$ with score $s_i(t) \ge \tau$.

For each protein $i$ where $|\hat{T}_i(\tau)| > 0$:
$$Pr_i(\tau) = \frac{|\hat{T}_i(\tau) \cap T_i|}{|\hat{T}_i(\tau)|}$$
$$Rc_i(\tau) = \frac{|\hat{T}_i(\tau) \cap T_i|}{|T_i|}$$

The dataset-level Precision and Recall at threshold $\tau$ are computed by averaging over the test proteins:
$$Pr(\tau) = \frac{1}{m(\tau)} \sum_{i: |\hat{T}_i(\tau)| > 0} Pr_i(\tau)$$
$$Rc(\tau) = \frac{1}{N} \sum_{i=1}^N Rc_i(\tau)$$
where $N$ is the total number of test proteins in the target aspect, and $m(\tau) = |\{i \mid |\hat{T}_i(\tau)| > 0\}|$ is the number of proteins with at least one prediction at threshold $\tau$.

The harmonic mean $F_1$-score is defined as:
$$F_1(\tau) = \frac{2 \cdot Pr(\tau) \cdot Rc(\tau)}{Pr(\tau) + Rc(\tau)}$$

### Maximum F-measure ($F_{\max}$)
$$F_{\max} = \max_{\tau \in [0, 1]} F_1(\tau)$$

$F_{\max}$ identifies the optimal trade-off between precision and recall across all operating points of the model.

---

## 2. Information Accretion ($IA$) & Semantic Distance $S_{\max}$

Because $F_{\max}$ treats all terms equally regardless of specificity, the CAFA consortium introduced **Information Accretion ($IA$)** and the semantic distance metric **$S_{\max}$** (*Clark & Radivojac, 2013*; *Radivojac et al., 2013*).

### Information Accretion ($IA$)
The Information Accretion of a term $t$ quantifies the information gain (in bits) obtained by observing term $t$ given that all of its parent terms $\mathcal{P}(t)$ are already observed:

$$IA(t) = -\log_2 \left( \frac{P(t \text{ is annotated})}{P(\text{all } p \in \mathcal{P}(t) \text{ are annotated})} \right)$$

- If $t$ is a root term with no parents ($\mathcal{P}(t) = \emptyset$), $P(\text{all } p \in \mathcal{P}(t)) = 1$, so $IA(t) = -\log_2 P(t)$.
- If $t$ is always co-annotated with its parents, $IA(t) \to 0$ (no new information is provided).
- Highly specific, rare leaf terms have large $IA(t)$ values ($IA(t) > 5.0$).

### Remaining Uncertainty ($RU$) & Misinformation ($MI$)
For a prediction set $\hat{T}_i(\tau)$ and ground truth $T_i$ on protein $i$:

- **Remaining Uncertainty ($RU_i(\tau)$)**: The information content of true terms that the model **failed** to predict (false negatives):
  $$RU_i(\tau) = \sum_{t \in T_i \setminus \hat{T}_i(\tau)} IA(t)$$

- **Misinformation ($MI_i(\tau)$)**: The information content of predicted terms that are **incorrect** (false positives):
  $$MI_i(\tau) = \sum_{t \in \hat{T}_i(\tau) \setminus T_i} IA(t)$$

The dataset-level averages are:
$$RU(\tau) = \frac{1}{N} \sum_{i=1}^N RU_i(\tau), \quad MI(\tau) = \frac{1}{N} \sum_{i=1}^N MI_i(\tau)$$

### Semantic Distance $S_{\max}$
The semantic distance $S(\tau)$ combines $RU(\tau)$ and $MI(\tau)$ in Euclidean space:
$$S(\tau) = \sqrt{RU(\tau)^2 + MI(\tau)^2}$$
$$S_{\max} = \min_{\tau \in [0, 1]} S(\tau)$$

> **Interpretation**: A lower $S_{\max}$ score indicates a superior predictor that minimizes both missing information and false biological claims.

---

## 3. Area Under the Precision-Recall Curve (AUPR)

Aid2Go computes both Micro-AUPR and Macro-AUPR:
- **Micro-AUPR**: Calculated by flattening all prediction pairs $(i, t)$ across the test set and computing the area under the global precision-recall curve via trapezoidal interpolation (`sklearn.metrics.auc`).
- **Macro-AUPR**: Calculated by computing AUPR independently per GO term across all test proteins, and averaging across terms.

---

## 4. Implementation in Aid2Go

In Aid2Go, these calculations are implemented in:
- `aid2go.eval.cafa_metrics.calculate_fmax`: Multi-threshold protein-centric $F_{\max}$ sweep.
- `aid2go.eval.cafa_metrics.calculate_smax`: Information-accretion weighted semantic distance minimization.
- `aid2go.eval.cafa_metrics.calculate_aupr`: Global precision-recall curve integration.
- `aid2go.eval.cafa_evaluator.CAFAEvaluator`: Subontology partition engine computing independent metrics across **BPO**, **CCO**, and **MFO**.
