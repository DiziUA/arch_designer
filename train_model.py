# train_model.py

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# 1) Пути
DATA_PATH  = "data/dataset.csv"
MODEL_PATH = "models/efficiency_model.pkl"

# 2) Проверяем и создаём папку моделей
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# 3) Загружаем датасет
df = pd.read_csv(DATA_PATH)

# 4) Разбиваем на X и y
X = df.drop(columns=["perf_per_watt"])
y = df["perf_per_watt"]

# 5) Обучаем модель
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 6) Сохраняем
joblib.dump(model, MODEL_PATH)
print(f"Model trained on {len(df)} samples and saved to {MODEL_PATH}")
