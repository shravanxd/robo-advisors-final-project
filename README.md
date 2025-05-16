# Can Computer Vision Models Generate Better Trading Signals than Time Series Models?

**Shravan Khunti** (NetID: ssk10036)  
**Yash Jadhav** (NetID: yj3076)  
MS in Data Science, NYU Center for Data Science  
Course: Robo Advisors & Systematic Trading  
Professor: Vasant Dhar  
[GitHub Repository](https://github.com/shravanxd/robo-advisors-final-project)

---

## Motivation

This project was inspired by a lecture from Professor Vasant Dhar on transforming time series data into images for model input—an idea that resonated with prior research conducted at the National University of Singapore. While earlier work focused on 1D convolutions over raw time series, this capstone explores a novel computer vision approach using **Gramian Angular Field (GAF)** transformations.

We hypothesize that visual encoding may reveal structural patterns missed by traditional numerical methods, enabling more effective trading signal generation. As a future extension, a **Reinforcement Learning (RL)** layer will be added to evaluate execution behavior using signals from both Time Series (TS) and Computer Vision (CV) pipelines.

---

## Project Overview

This capstone implements a multi-phase modeling pipeline for algorithmic trading, structured as:

Time Series → Computer Vision (GAF) → Reinforcement Learning (planned)


- **Time Series Models** use engineered features (RSI, MACD, etc.) for classification.
- **Computer Vision Models** use CNNs on GAF images of market windows.
- **Goal:** Evaluate which approach delivers better signals on metrics like Sharpe ratio, return, drawdown, and trading frequency.

---

## Recommended Reading

**Start Here → [rasi_final_project_report.pdf](./rasi_final_project_report.pdf)**

For a comprehensive overview of our capstone—including background, modeling logic, results, and final insights—we recommend reading the **final project report** before diving into the notebook or code.

---

## Implementation Breakdown

### Cell 1: Initialization & Library Imports
- Imports NumPy, Pandas, TensorFlow, Scikit-learn, `pyts.image.GramianAngularField`, etc.
- Sets global seeds, style settings, and confirms package versions.

### Cell 2: Data Loading, Cleaning & EDA
- Stocks: `AAPL`, `MSFT`, `GOOGL`, `SPY` from 2010–2025.
- Cleaned and normalized price/volume.
- Explored correlations, drawdowns, volatility, and returns.
- Key Insight: Distinct asset risk profiles justify model variety.

### Cell 3: Feature Engineering & Labeling
- Engineered 40+ technical indicators (MA, RSI, MACD, OBV, etc.).
- Labeled future 5-day returns into **Buy**, **Sell**, **Hold**.
- Bias-free historical usage ensured (no leakage).
- Analyzed label distributions across tickers.

### Cell 4: Data Preparation for TS & CV Models
- Split data chronologically: 70% Train, 10% Val, 20% Test.
- **Time Series**: 42-feature numeric matrix.
- **GAF Images**: 4 channels (Price, Volume, Volatility, Momentum), 20×20×4 shape.
- Preserved label balance and look-ahead prevention.

### Cell 5: Time Series Modeling
- Models: Logistic Regression, Random Forest, XGBoost, LightGBM.
- Tuned class weights (Sell emphasis), calibrated thresholds.
- Evaluated via Sharpe, strategy return, win rate, drawdown.

| Ticker | Best Model         | Sharpe | Strategy Return | B&H Return |
|--------|--------------------|--------|------------------|------------|
| AAPL   | XGBoost + Sell2.0x | 1.69   | 51.96%          | 16.58%     |
| MSFT   | LogReg + Sell0.5x  | 1.10   | 31.23%          | 25.75%     |
| GOOGL  | LogReg + SH0.5x    | 0.83   | 47.52%          | 30.87%     |
| SPY    | LightGBM + SH0.5x  | 0.87   | 23.90%          | 12.32%     |

### Cell 6: Computer Vision (GAF) Modeling
- CNNs trained on GAF images.
- Tuned: learning rate, regularization, dropout, batch size.
- Metrics: EvalScore, Sharpe, Return, Win Rate.

| Ticker | CNN Config        | Sharpe | Strategy Return | B&H Return |
|--------|-------------------|--------|------------------|------------|
| AAPL   | CNN Cfg35         | 0.88   | 43.04%          | 11.38%     |
| MSFT   | CNN Cfg16         | 1.14   | -22.22%         | 23.65%     |
| GOOGL  | CNN Cfg67         | 0.91   | 50.63%          | 22.68%     |
| SPY    | CNN Cfg16         | 0.51   | 16.48%          | 6.57%      |

- GAF provided rich visual features; however, drawdowns were higher.
- Models responded well to sell-weighted training.

### Cell 7: Final Comparative Analysis

| Approach        | Ticker | Sharpe | Return | Drawdown | Trades |
|----------------|--------|--------|--------|----------|--------|
| Time Series     | SPY    | 1.66   | 37.23% | -9.31%   | 26     |
| Computer Vision | GOOGL  | 0.91   | 50.63% | -30.46%  | 84     |

- **Time Series** models were more conservative, with lower drawdowns and better precision.
- **Computer Vision** models achieved higher raw returns but were more volatile.
- GOOGL and AAPL stood out for both pipelines.

### Cell 8: Statistical Analysis
| Metric          | CV Mean | TS Mean | Diff (%) | Better Model | Significant | Effect Size |
| --------------- | ------- | ------- | -------- | ------------ | ----------- | ----------- |
| Sharpe Ratio    | 0.86    | 1.33    | -35.6%   | TS           | No          | Large       |
| Total Return    | 0.22    | 0.30    | -26.8%   | TS           | No          | Small       |
| Win Rate        | 0.57    | 0.72    | -20.6%   | TS           | No          | Large       |
| Max Drawdown    | -27.4%  | -7.8%   | -252.3%  | TS           | **Yes**     | Large       |
| Composite Score | 0.53    | 0.80    | -33.7%   | TS           | **Yes**     | Very Large  |

- Time Series models consistently outperformed Computer Vision models across all major trading metrics.
= Max Drawdown and Composite Score differences were statistically significant.
= CV models executed over 300% more trades, indicating higher churn but lower precision.

TS models were more stable and effective at capital preservation, with fewer trades and higher quality signals.
---

## Conclusion & Future Work

- **Conclusion:** Time Series models are more robust and risk-aware. However, Computer Vision models—especially GAF-based CNNs—unlock high-return opportunities, especially during volatile regimes.
- **Next Steps:**
  - Build a **Hybrid Ensemble** combining TS + CV strengths.
  - Add a **Reinforcement Learning (RL) Layer** to act on hybrid signals.
  - Explore dynamic regime switching based on confidence scores and market volatility.
## Final Takeaway
- Time Series models outperformed across all core metrics and showed stronger risk control.

- Computer Vision models (e.g., GAF-based CNNs) showed promise in capturing high-return patterns but suffered from higher volatility and overtrading.

- Results are based on a small sample (4 tickers) and should be treated as a baseline benchmark, not a definitive conclusion.

- Statistical tests (t-test, Wilcoxon, Cohen’s d) validate the direction of performance differences.

- Composite score design is subjective and may vary across portfolio objectives.

- Hybrid models that ensemble CV and TS predictions offer a promising path forward, combining precision and pattern recognition.

- Future enhancements:

  - Stronger compute and training pipelines

  - Better handling of overfitting and data leakage

  - Reinforcement Learning layers to act on hybrid signals

  - Regime switching logic to dynamically adapt to market conditions



---

## Contact

For questions, collaborations, or feedback:

- Shravan Khunti: [shravan.khunti@nyu.edu](mailto:shravan.khunti@nyu.edu)
- Yash Jadhav: [yj3076@nyu.edu](mailto:yj3076@nyu.edu)

---

> *This project is part of the NYU MSDS Capstone for "Robo Advisors & Systematic Trading" (Spring 2025). All results are for academic purposes and not intended as financial advice.*



