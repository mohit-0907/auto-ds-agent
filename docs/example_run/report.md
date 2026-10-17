# Run report: mpg.csv

_Generated 2026-10-17 06:19 by auto-ds-agent. Narrative by openai:gpt-6.1-sol; every number in it was checked against the run's computed results._

## Summary

The run predicted car fuel efficiency in miles per gallon and recommends Hist Gradient Boosting. Its cross-validation RMSE was 2.811 mpg, 64.0% lower than the mean-prediction baseline's RMSE of 7.805.

## Dataset

398 rows × 9 columns; 0 duplicate rows.

## Task

**regression** · target `mpg` · router confidence 0.99

Regression was selected with `mpg` as the target, matching the request to predict fuel efficiency. The target is numeric with 129 distinct values ranging from 9.0 to 46.6, supporting treatment as a continuous quantity rather than a classification task.

## Feature engineering

The generated feature-engineering code passed validation on the first attempt and was used for the final features. It dropped the high-cardinality car-name column, retained numeric vehicle measurements and model year, imputed missing or non-finite numeric values with column medians, and one-hot encoded origin. It also added horsepower per unit weight and displacement per cylinder without using the target. Generic handling was included for any additional identifier-like, text, date, or categorical columns.

## Leaderboard

| Rank | Model | RMSE | MAE | R² | Train (s) | Predict (ms/fold) |
|---|---|---|---|---|---|---|
| 1 | Random Forest | 2.772 ± 0.445 | 1.943 ± 0.200 | 0.872 ± 0.029 | 1.92 | 22.3 |
| 2 | LightGBM | 2.793 ± 0.388 | 1.973 ± 0.169 | 0.871 ± 0.023 | 0.15 | 5.3 |
| 3 | **Hist Gradient Boosting** ✅ | 2.811 ± 0.435 | 1.970 ± 0.210 | 0.869 ± 0.030 | 0.48 | 3.7 |
| 4 | XGBoost | 3.029 ± 0.335 | 2.116 ± 0.144 | 0.848 ± 0.025 | 2.21 | 2.0 |
| 5 | Ridge Regression | 3.136 ± 0.366 | 2.329 ± 0.208 | 0.837 ± 0.023 | 0.02 | 0.8 |
| 6 | Dummy (mean) | 7.805 ± 0.463 | 6.538 ± 0.373 | -0.007 ± 0.007 | 0.03 | 1.0 |

## Recommendation

Use Hist Gradient Boosting. Random Forest had the lowest mean RMSE at 2.772, but beat Hist Gradient Boosting on only 3 of 5 shared folds. Choosing the simpler Hist Gradient Boosting was deliberate: the selection rule requires a higher-scoring model to beat every simpler model on at least 4 of 5 folds. Ridge Regression is an alternative if prediction cost or interpretability is the priority: it had RMSE 3.136 and took 0.8 ms per fold, versus 3.7 ms for Hist Gradient Boosting.

## Caveats and next steps

- The dataset contains 398 rows; check performance on additional cars before relying on the model more broadly.
- Results come from shared 5-fold cross-validation. Hist Gradient Boosting's RMSE was 2.811 ± 0.435, so consider fold-to-fold variation alongside the mean.
- Random Forest and LightGBM had slightly lower mean errors than the selected model. Keep them as comparison candidates if evaluating on new data.
- Prediction timing was measured per fold. Check latency in the intended deployment setting before choosing between Hist Gradient Boosting and Ridge Regression.

## Warnings recorded during the run

- Chose Hist Gradient Boosting over Random Forest: Random Forest's better mean rmse (2.7720 vs 2.8109, lower is better) held on only 3 of 5 folds, so the simpler model wins the tie.
