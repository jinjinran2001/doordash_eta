# DoorDash ETA Prediction

This project explores machine learning and optimization techniques to predict delivery time (ETA) for DoorDash food orders. We benchmark multiple models and explore computational optimizations including parallel processing and GPU acceleration.

## Dataset

The dataset used is publicly available on Kaggle:  
[DoorDash ETA Prediction Dataset](https://www.kaggle.com/datasets/dharun4772/doordash-eta-prediction/data)
We also include the dataset as `historical_data.csv` in the repository

It contains ~197,000 historical delivery records with metadata such as:
- Timestamps (created_at, actual_delivery_time)
- Order attributes (subtotal, total_items, etc.)
- Operational features (total_onshift_dashers, estimated driving time)

## Models Used

- **Linear Regression / Ridge / Lasso**
- **Random Forest**
- **Gradient Boosting (XGBoost, LightGBM, CatBoost, HistGradientBoosting)**
- **Neural Network (PyTorch)**

## Optimization Techniques

- Internal parallelism (e.g. `n_jobs=-1` in Random Forest)
- External parallelism with `joblib.Parallel`
- GPU acceleration (CatBoost + PyTorch)
- Randomized feature selection with multiprocessing
- Profiling with `line_profiler`

## Results

| Model            | MAE (minutes) | Notes                        |
|------------------|---------------|------------------------------|
| Ridge            | ~10.0         | Good baseline                |
| Random Forest    | 8.03          | Decent accuracy, slower train|
| LightGBM         | **7.82**      | Most accurate                |
| CatBoost (GPU)   | 0.94s train   | Fastest training             |
| Neural Net       | ~8.65         | Captures non-linear patterns |


