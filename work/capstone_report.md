# Capstone Report — Structured Content Archetype Clustering

- **Author:** Qasim
- **Lane:** Lane 3 — Structured Content Archetype Clustering
- **Repo:** [CodeByQasim/flyrank-ml-internship](https://github.com/CodeByQasim/flyrank-ml-internship)
- **Date:** September 2026

---

## 0. Abstract

Content marketing teams frequently struggle to prioritize editorial resources across thousands of published pages, often relying on simplistic one-dimensional metrics like raw pageviews or static decay rules. In this research, we analyze 90-day search visibility and user engagement metrics across 30,000+ pseudonymized content items from the FlyRank warehouse release. We engineer multi-dimensional performance features (visibility, efficiency, user engagement, and query concentration) and implement an unsupervised K-Means clustering architecture with PCA dimensionality reduction to discover natural performance archetypes. Our machine learning approach identifies six distinct content archetypes—*Evergreen Champions*, *Decaying Visible Pages*, *Low-CTR Opportunities*, *Hidden Gems*, *High-Intent Niche*, and *Thin/Zombie Content*—achieving a Silhouette Score of **0.384** and high stability across client holdouts, significantly outperforming a rigid rule-based heuristic matrix (Silhouette: **0.182**). Finally, we translate these discovered archetypes into an automated, ranked editorial action engine that routes each content item to a high-confidence operational decision (Protect, Refresh, Rewrite Snippet, Expand, Prune).

---

## 1. Problem framing

Editorial teams maintaining enterprise content repositories face a fundamental operational challenge: **which published URLs require immediate attention, and what specific action should be taken?**

Traditional content audits rely on brittle, one-dimensional threshold rules—such as flagging all pages older than 180 days for a refresh, or flagging pages with declining clicks for deletion. This approach causes substantial resource misallocation:
1. **False Refreshes:** Editorial time is wasted rewriting high-ranking *Evergreen Champions* that merely experienced natural seasonal fluctuations.
2. **Missed Low-Hanging Fruit:** High-impression pages with sub-optimal snippets (*Low-CTR Opportunities*) undergo expensive 2,000-word rewrites when a 5-minute title and meta tag optimization was the appropriate lever.
3. **Premature Pruning:** *Hidden Gems* (pages with high user engagement stuck on Google Page 2) are deleted rather than promoted through internal linking.

Unsupervised machine learning provides an objective, multi-dimensional lens to segment inventory into distinct behavioral archetypes based on the interaction between visibility, ranking efficiency, user engagement, and keyword concentration.

- **Unit of Analysis:** Single pseudonymized content item (`content_hash_id`) aggregated over a 90-day trailing performance window.
- **Output:** Clustered archetype profile assignments and an impact-ranked operational action queue (`outputs/archetype_action_queue.csv`).
- **Target Actions:** `PROTECT & MONITOR`, `REFRESH & EXPAND DEPTH`, `REWRITE SNIPPET & META`, `BOOST INTERNAL LINKS`, `EXPAND KEYWORD CLUSTERS`, `PRUNE OR 301 MERGE`.

---

## 2. Data safety & Data Contract

### Dataset Source & Cohort Definition
- **Data Source:** FlyRank Pseudonymized Warehouse Release (`FlyRank/internship-warehouse` build `v20260703`).
- **Primary Tables:** `dim_content`, `fact_content_daily_performance` (partitioned daily time series), and `fact_content_query_90d` (query distribution context).
- **Date Window:** 90-day observation window (April 1, 2026 – June 30, 2026).
- **Inclusion Filters:** Content items with `impressions_90d >= 100` and `content_age_days >= 30` to exclude unindexed, sparse noise.

### Data Privacy & Leakage Safeguards
- **Cryptographic Hashing:** All domain names, client names, raw URLs, content titles, and query strings are pseudonymized (`client_hash_id`, `content_hash_id`, `keyword_hash_id`).
- **Feature Exclusion:** Identifiers are used solely for grouping and holdout validation—**never as model features**.
- **Label Discipline:** No target-derived fields (e.g. `trend_direction` or `trend_pct`) are included in clustering feature matrices.
- **Honest Scope:** This is **Structured Content Archetype Clustering** based on telemetry and metadata—it does not claim to perform semantic text processing.

---

## 3. Baseline

To establish a fair performance benchmark, we constructed a **Rule-Based Heuristic Archetype Matrix** reflecting standard industry decision thresholds:
- **Rule 1 (Champions):** `impressions_90d >= 1000` ∧ `ctr >= 2.0%` ∧ `avg_position <= 5.0`
- **Rule 2 (Stale Decay):** `content_age_days >= 180` ∧ `impressions_90d >= 500` ∧ `avg_position > 10.0`
- **Rule 3 (Low CTR):** `impressions_90d >= 500` ∧ `avg_position <= 10.0` ∧ `ctr < 1.0%`
- **Rule 4 (Hidden Gems):** `ctr >= 2.5%` ∧ `avg_position > 10.0` ∧ `impressions_90d < 500`
- **Rule 5 (Thin / Zombie):** `word_count < 800` ∧ `impressions_90d < 250`
- **Rule 6 (Standard Backlog):** All remaining inventory.

### Baseline Evaluation Numbers
- **Baseline Silhouette Score:** `0.182`
- **Baseline Calinski-Harabasz Index:** `1,420.5`
- **Baseline Davies-Bouldin Index:** `2.841`

The heuristic baseline suffers from high intra-cluster variance and arbitrary boundary cutoffs, leaving a large portion of inventory in an uninformative "general backlog" bucket.

---

## 4. Model / Analysis

### Feature Engineering
To handle extreme power-law distributions and normalize varied scales, we engineered 9 continuous features:
1. `log_impressions = ln(1 + impressions_90d)` (Search demand scale)
2. `log_clicks = ln(1 + clicks_90d)` (Captured traffic scale)
3. `ctr = (clicks_90d / impressions_90d) * 100` (Snippet conversion efficiency)
4. `position_opportunity = max(0, 20 - avg_position)` (Ranking strength)
5. `log_word_count = ln(1 + word_count)` (Content depth)
6. `log_age = ln(1 + content_age_days)` (Lifecycle maturity)
7. `scroll_rate = (scroll_events / sessions) * 100` (On-page user engagement)
8. `log_visible_queries = ln(1 + visible_queries)` (Topical footprint breadth)
9. `top_query_share = max_query_imp / total_query_imp` (Keyword reliance vs diversification)

All features are normalized using `StandardScaler`.

### Machine Learning Model Formulation
We selected **K-Means Clustering** ($K=6$) initialized with `k-means++` across 15 random initializations. Optimal $K$ was determined via Elbow Inertia and Silhouette optimization across $K \in [3, 8]$.

---

## 5. Evaluation & Validation

### Model vs. Baseline Comparison on Identical Feature Space

| Evaluation Metric | Rule-Based Baseline | K-Means ML Model ($K=6$) | Relative Lift |
| :--- | :---: | :---: | :---: |
| **Silhouette Score (↑ higher is better)** | `0.182` | **`0.384`** | **+111.0%** |
| **Calinski-Harabasz Index (↑ higher is better)** | `1,420.5` | **`3,892.4`** | **+174.0%** |
| **Davies-Bouldin Index (↓ lower is better)** | `2.841` | **`1.124`** | **-60.4%** |

### Client-Grouped Holdout Validation
To ensure that discovered clusters generalize across independent web domains rather than overfitting to specific client architectures, we evaluated the model using `GroupShuffleSplit` across client domains (75% Train Clients, 25% Unseen Holdout Clients):
- **Train Client Silhouette Score:** `0.388`
- **Unseen Holdout Client Silhouette Score:** `0.372`
- **Generalization Retention Rate:** **95.8%**
- **Seed Stability (Adjusted Rand Index across seeds):** **0.942**

The model demonstrates excellent cross-domain generalizability and cluster partition stability.

---

## 6. Interpretation & Discovered Archetypes

The 6 discovered archetypes present clear operational signatures:

| Archetype Name | Inventory Share | Med Imp (90d) | Mean CTR | Avg Pos | Med Words | Signature & Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Evergreen Champions** | 14.2% | 3,420 | 3.4% | 3.2 | 1,850 | High volume, dominant Page 1 rankings, exceptional CTR. Core revenue drivers. |
| **2. Low-CTR Opportunities** | 16.5% | 2,180 | 0.6% | 4.8 | 1,420 | Strong Page 1 rankings with severe click under-capture. Snippet mismatch. |
| **3. Decaying Visible Pages** | 19.1% | 1,450 | 1.1% | 9.4 | 1,200 | Older articles (>240d) losing rank momentum and slipping to bottom of Page 1. |
| **4. Hidden Gems / High Potential**| 12.8% | 340 | 3.1% | 14.6 | 1,600 | High user engagement and CTR stuck on Google Page 2. High upside for link equity. |
| **5. High-Intent Niche** | 15.3% | 620 | 2.2% | 7.1 | 1,100 | High query concentration on specialized topics with strong scroll depth. |
| **6. Thin / Zombie Content** | 22.1% | 120 | 0.4% | 28.5 | 450 | Low word count, negligible traffic, high age. Dragging down site quality score. |

---

## 7. Recommendations & Action Playbook

We deployed an automated scoring engine that maps every page to an actionable operational queue:

1. **`REWRITE SNIPPET & META` (Low-CTR Opportunities):**
   - *Action:* Retitle `<title>` tags and meta descriptions to improve SERP click-through rates.
   - *Priority:* Ranked by raw impression volume (largest immediate traffic gain).
2. **`REFRESH & EXPAND DEPTH` (Decaying Visible Pages):**
   - *Action:* Update statistics, refresh timestamps, and expand core sections with fresh research.
   - *Priority:* Ranked by age and historical impression volume.
3. **`BOOST INTERNAL LINKS` (Hidden Gems):**
   - *Action:* Add contextual in-body links from *Evergreen Champions* to push rankings from Page 2 to Page 1.
   - *Priority:* Ranked by CTR efficiency and closeness to position 10.
4. **`PROTECT & MONITOR` (Evergreen Champions):**
   - *Action:* Lock content from unnecessary edits; establish automated position volatility alerts.
5. **`PRUNE OR 301 MERGE` (Thin / Zombie Content):**
   - *Action:* 301-redirect low-value URLs to parent category hubs or remove to consolidate crawl budget.

---

## 8. Reproducibility

- **Repository:** `https://github.com/CodeByQasim/flyrank-ml-internship`
- **Primary Notebook:** `work/notebooks/capstone.ipynb`
- **Random Seed:** `RANDOM_SEED = 42`
- **Environment:** Python 3.11+, `duckdb>=1.0`, `scikit-learn>=1.4`, `pandas>=2.2`, `numpy>=1.26`, `matplotlib>=3.8`
- **Re-run Instructions:**
  ```bash
  git clone https://github.com/CodeByQasim/flyrank-ml-internship.git
  cd flyrank-ml-internship
  pip install -r requirements.txt
  # Run capstone notebook or local scripts
  ```

---

## 9. Acknowledgments & Data Credit

Built on the **[FlyRank ML Internship Dataset](https://flyrank.ai)**.  
We express our gratitude to the FlyRank ML Internship team for providing open research access to the pseudonymized search intelligence warehouse.
