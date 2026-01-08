# Flight Delay Prediction – Random Forest vs XGBoost

This is my personal machine learning project to predict whether a flight will arrive late using only information available **before** departure (schedule, route, carrier, etc.). The goal is to build realistic preprocessing pipelines, compare tree‑based models, and tune the decision threshold so the model fits a real business goal: it is worse to miss a real delay than to send an extra warning.

## Repository structure

All notebooks are in the `Notebooks/` folder:

- `data_exploration.ipynb`  
  Explore the sampled dataset, check class balance, and look at how delays relate to departure time, routes, and carriers.

- `preprocessing.ipynb`  
  Apply standard preprocessing: cleaning, basic categorical encodings, and scaling for numeric features.

- `advanced_preprocessing.ipynb`  
  Build richer features such as time‑of‑day bins, carrier–airport combinations, and other advanced encodings.

- `automated_preprocessing_pipelines.ipynb`  
  Wrap both standard and advanced preprocessing into reusable pipelines that can be plugged directly into models.

- `baseline_model_training.ipynb`  
  Train baseline Random Forest and XGBoost models on different preprocessed datasets (16 experiments) to get reference metrics.

- `model_optimization.ipynb`  
  Use the advanced preprocessing pipeline to tune Random Forest and XGBoost, analyse feature importance, and tune the decision threshold (0.4, 0.5, 0.6) for the final XGBoost model.

## Data

- Historical flight records with scheduled times, origin, destination, carrier, distance, and related columns.  
- For faster experiments, I randomly sample **10,000 rows** from the full dataset while keeping the delay proportion and feature distributions similar.  
- Target: binary label indicating if the flight arrived delayed (1) or not delayed (0).

## Preprocessing

### Standard preprocessing (for baselines)

Implemented in `preprocessing.ipynb`:

- Clean types and handle obvious issues.  
- Apply simple encodings to categorical features.  
- Standard‑scale numeric columns.  
- Use this pipeline to train quick baseline models and understand the raw signal.

### Advanced preprocessing (industry‑style)

Implemented in `advanced_preprocessing.ipynb`:

- Time features:
  - `dep_time_bin` for broad time‑of‑day periods (morning, afternoon, evening, night).
  - `crs_dep_time` for the exact scheduled departure hour.
- Route and airport features:
  - `origin`, `dest`, and `carrier_origin` to capture airport and carrier–airport effects.
- Other features:
  - `distance`, `month`, and `day_of_week` to model route length and seasonal/weekly patterns.
- High‑cardinality categoricals:
  - Target encoding and similar techniques so I can keep the feature space compact while still capturing how categories relate to delay.

### Automated pipelines

Implemented in `automated_preprocessing_pipelines.ipynb`:

- Combine preprocessing and the estimator into single pipeline objects.  
- Make experiments easier to rerun, avoid data leakage, and ensure fair model comparisons.

After comparing results, I use the **advanced preprocessing pipeline** for all optimized models because it improves performance on delayed flights and looks closer to a production setup.

## Modeling approach

### Baseline models

Trained in `baseline_model_training.ipynb` using the advanced preprocessed data.

- **Baseline Random Forest**
  - Accuracy: **0.8155**
  - F1 (delay): **0.3399**
  - ROC‑AUC: **0.7158**

- **Baseline XGBoost**
  - Accuracy: **0.82505**
  - F1 (delay): **0.3165**
  - ROC‑AUC: **0.7388**

These baselines achieve good accuracy but relatively low F1 on the delay class, which means they still miss many delayed flights.

### Model optimization

Performed in `model_optimization.ipynb` on the advanced feature set.

- **Tuned Random Forest**
  - Best parameters (RandomizedSearchCV):
    - `bootstrap=False`
    - `max_depth=32`
    - `max_features=None`
    - `min_samples_leaf=9`
    - `min_samples_split=9`
    - `n_estimators=367`
  - Accuracy: **0.7877**
  - F1 (delay): **0.3527**
  - ROC‑AUC: **0.6590**

  Tuning RF slightly improves F1 compared to the baseline RF, but accuracy and ROC‑AUC drop, and the model still misses many delayed flights.

- **Tuned XGBoost**
  - Best parameters (RandomizedSearchCV):
    - `colsample_bytree ≈ 0.963`
    - `gamma ≈ 1.246`
    - `learning_rate ≈ 0.092`
    - `max_depth = 5`
    - `min_child_weight = 4`
    - `n_estimators = 112`
    - `scale_pos_weight ≈ 4.043`
    - `subsample ≈ 0.840`
  - Accuracy: **0.70725**
  - F1 (delay): **0.4599**
  - ROC‑AUC: **0.74896**

  Compared to baselines and tuned RF, tuned XGBoost trades some accuracy for a much higher F1 and ROC‑AUC on the delay class. It catches more delayed flights and separates risky vs safe flights more reliably.

## Confusion matrices and decision thresholds

On the test set:

- **Tuned Random Forest**
  - Predicts many on‑time flights correctly.
  - Misses a large number of delayed flights (high false‑negative count).

- **Tuned XGBoost**
  - Correctly identifies many more delayed flights (higher true positives).
  - Generates more false alarms (higher false positives).

Because missing a real delay is more costly than an extra warning, I tune the XGBoost decision threshold:

- Threshold **0.6** → stricter: higher precision, lower recall (fewer warnings, more missed delays).
- Threshold **0.5** → balanced behaviour (default).
- Threshold **0.4** → more aggressive: lower precision, higher recall (more warnings, fewer missed delays).

The final operating threshold is chosen on the lower side (0.5 or 0.4) to prioritise recall and F1 for delayed flights rather than raw accuracy.

## Feature importance – what matters most?

I analyse feature importance for both tuned models.

- **Tuned XGBoost**
  - Most important features: `dep_time_bin`, `carrier_origin`, `crs_dep_time`, followed by `month`, `dest`, `day_of_week`, `op_unique_carrier`, `origin`, and `distance`.
  - Interpretation:
    - Departure time (both binned and exact) is crucial because congestion peaks and late‑night operations drive many delays.
    - Route and carrier–airport combinations matter: some airports and carriers consistently run later than others.
    - Calendar and distance features capture seasonal and route‑length patterns.

- **Tuned Random Forest**
  - Top features: `crs_dep_time`, `dep_time_bin`, `carrier_origin`, `distance`, `dest`, `month`, `origin`, `day_of_week`, `op_unique_carrier`.
  - This largely agrees with XGBoost, giving confidence that both models are learning sensible, pre‑flight signals instead of relying on any leaked, post‑fact information.

## Final model and takeaway

- Baseline models have strong accuracy but weak performance on the delay class.  
- Tuning Random Forest helps a bit but still misses many delays.  
- Tuning XGBoost, using the advanced preprocessing pipeline, and lowering the decision threshold produces a model that catches far more delayed flights and ranks risk much better, even though its accuracy is lower.

For an operations team, that trade‑off is acceptable: it is better to get a few extra “this flight might be delayed” warnings than to be surprised by delays that the model could have flagged.

## Environment

To reproduce the results:

```bash
pip install -r requirements.txt


## Reflection

- What worked:
  - Advanced preprocessing (time bins, route/carrier features, target encoding) plus XGBoost improved F1 and ROC‑AUC on delayed flights.
  - Decision‑threshold tuning let me align the model with a clear business goal (better to over‑warn than miss delays).

- What didn’t help as much:
  - Tuning Random Forest gave only a small F1 gain and actually reduced ROC‑AUC.
  - Focusing only on accuracy hid how many delays were being missed.

- What I’d do next:
  - Add weather and airport‑congestion features to see how much they boost performance.
  - Deploy the final pipeline as a small API or Streamlit app so users can score flights interactively.
