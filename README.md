# 🧠 Neural Trend Detection for the USA Beauty Market

A segment-aware trend intelligence system that predicts which beauty trends are accelerating — and for whom — by combining unsupervised consumer segmentation with a supervised neural forecaster.

---

## 🌟 Project Overview

This project started as a transformer-based trend detector for social media captions and hashtags. The available data turned out to be something else entirely — structured behavioral survey data with no text and no timestamps. Rather than force a mismatch, the project pivoted to a different, more answerable question: **can consumer behavioral archetypes improve trend momentum prediction beyond search data alone?**

The answer is yes. A segment-aware neural network that combines Google Trends signal with consumer behavioral profiles outperforms a trends-only model by 5x in explained variance (R² 0.056 → 0.290).

## 🎯 Objectives

1. Identify behavioral consumer archetypes from shopping preference data using unsupervised learning
2. Compress noisy, high-dimensional behavioral features into a latent representation via an autoencoder
3. Cluster the latent representations into distinct consumer segments
4. Train a feedforward neural network (FFNN) to predict segment-specific trend momentum
5. Evaluate whether segment awareness adds genuine predictive value over a trends-only baseline

## 🧠 Methodology

**Pipeline:** Behavioral data → Autoencoder → K-Means clustering → Segment-aware FFNN → Trend momentum predictions

| Stage | What it does |
|---|---|
| **Autoencoder** | Compresses 25 behavioral features into an 8-dimensional latent representation (68% compression) |
| **K-Means** | Clusters the latent representations into consumer archetypes |
| **Segment-Aware FFNN** | Takes an 18-dimensional input (10 trend features + 8 segment behavioral dimensions) and predicts trend momentum per segment |

Two consumer archetypes emerged from clustering:
- **Digital Native** (13% of consumers) — online-first discovery, ingredient-led content, creator-driven
- **Traditional Shopper** (87% of consumers) — in-store experience, trusted brand positioning, service-driven

## 📊 Model Performance

| Metric | Trend-Only FFNN (10-dim) | Segment-Aware FFNN (18-dim) |
|---|---|---|
| R² | 0.056 | **0.290** |
| MAE | 0.334 | **0.268** |
| vs. naive baseline (MAE 0.331) | −5.6% (worse) | **+19.0% (better)** |
| Training samples | 168 | 336 |

Adding the 8-dimensional segment behavioral vector — derived from the autoencoder and K-Means clustering — improved R² 5x and turned a model that underperformed the naive baseline into one that meaningfully beat it.

**Component-level validation:**
- Autoencoder: validation loss 0.589 on 68% compression, with train/val curves converging (no overfitting)
- Clustering: silhouette score 0.2475 (clears the 0.2 meaningful-structure threshold); 88% average cluster purity against unseen shopping-preference labels; 6 of 10 key features statistically significant between clusters (Mann-Whitney U, p < 0.0001)
- FFNN: test loss decreased consistently from 0.950 (epoch 1) to 0.630 (epoch 150) with no train/test divergence

## 💡 Key Findings

**SPF makeup is the standout insight.** It has the highest composite trend score (92.3) of any product tracked, but the FFNN is the only signal predicting it's *decelerating* for both segments (−0.241 Digital Native, −0.253 Traditional Shopper). Historical volume is high, but the most recent 8-month window shows the trend cooling — a divergence a brand manager relying on composite score alone would miss, leading to over-investment.

**Digital Natives lead adoption on 8 of 12 tracked trends** (glass skin, tinted moisturizer, liquid blush, retinol serum, peptide serum, lip liner, skincare makeup, and others). The 4 Traditional Shopper-led exceptions are interpretable: solid perfume is a tactile, in-store discovery product; skin barrier repair is education- and trust-driven; retinol alternatives skew toward consumers seeking gentler formulations; clean girl makeup is a declining trend sustained by existing shoppers rather than new discovery.

**Channel strategy by segment:**

| Segment | Size | Top Trends | Recommended Channel |
|---|---|---|---|
| Digital Native | 13% | glass skin, liquid blush, peptide serum | Online DTC, ingredient-led content, creator partnerships |
| Traditional Shopper | 87% | solid perfume, skin barrier repair | In-store experience, trusted brand positioning, service |

## ⚠️ Documented Limitations

Honest evaluation of where this model's predictions should and shouldn't be trusted:

- **Behaviorally-weighted targets, not observed adoption data** — training targets are trend momentum scores weighted by behavioral alignment, not measured adoption speed. Validating against real sales or loyalty-card data would be needed for production use.
- **Small training dataset** — 336 samples from 16 trends × 2 segments via sliding-window augmentation. A production system tracking 100+ trends would yield far more training diversity.
- **Small momentum deltas between segments** — most predicted differences are below 0.10, reflecting genuine behavioral similarity found in clustering (the segments differ on channel preference, not psychographic profile).
- **No ground truth for k=2 clusters** — statistically optimal by silhouette score, but sub-segments likely exist within the 87% Traditional Shopper cluster that the current bottleneck collapses together.
- **Dataset geography unclear** — the behavioral dataset's structure suggests non-US origin; US Google Trends data is used as a market-context proxy, not a confirmed geographic match.
- **Static behavioral snapshot** — no timestamps in the consumer dataset, so segment profiles may not reflect behavior 12–18 months later.

## 📖 The Story Behind This Project

The first unexpected finding came from debugging, not modeling. The original keyword list — "viral skincare TikTok," "virtual try-on makeup," "K-beauty routine" — returned near-zero US search volume (0.5–1.7 out of 100). The fix wasn't a data correction; it was a reframe: Digital Natives don't Google what they see on TikTok, they Google the specific product it leads them to (retinol serum, snail mucin moisturizer, peptide serum). Discovery and search are different moments in the purchase journey.

The clustering result was cleaner than expected, and more limited than hoped. The two archetypes are nearly identical on psychographic dimensions — they differ on channel preference and spending behavior, not personality. No scoring formula could manufacture differentiation that wasn't in the data, so the project leaned into that finding rather than against it.

The hardest architectural decision was integrating the two parallel tracks — trend signal and consumer profile — that the original pipeline kept separate. Concatenating the 8-dimensional segment vector with the 10-dimensional trend window features into a single 18-dimensional FFNN input is what took R² from 0.056 to 0.290.

## 🛠 Built With

**ML:** PyTorch, scikit-learn — Autoencoder, K-Means on neural embeddings, Segment-Aware FFNN, UMAP, representation learning
**Data processing:** pandas, UMAP, matplotlib
**Data sources:** Kaggle Consumer Shopping Trends dataset; Google Trends US (April 2024 – April 2026)

## 📂 Repository Contents

- Data:

| Source | Description | Shape |
|---|---|---|
| [Consumer Behavioral Dataset (Kaggle)](https://www.kaggle.com/datasets/minahilfatima12328/consumer-shopping-trends-analysis/data)	| 25 behavioral + demographic features per consumer	| (11,789 × 25) |
| Google Trends Time Series (US) |	Monthly search interest, 15 beauty trends, 24 months | (25 × 16) |
| Google Trends — Top Queries (US) |	Highest-volume related searches per trend	| (598 × 4) |
| Google Trends — Rising Queries (US) | Breakout and high-growth related searches per trend |	(598 × 4) |

- AML-Beauty Trend Intelligence

---

*Part of a broader portfolio applying machine learning and consumer behavioral analysis to the beauty industry. See also: [Glossier Digital Marketing Analytics Case Study](https://github.com/carolina-galindo-m/Consumer-Insights-Digital-Analytics-Case-Study-Glossier) and [ULTA Beauty Financial Forecasting](https://github.com/carolina-galindo-m/ULTA-Beauty-Financial-Forecasting-Strategic-Insights-Case-Study).*
