import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

# 1. Cargamos los datos y creamos las primeras variables
df = pd.read_csv("coolers_features_ext.csv", parse_dates=["calday"])
df["compressor_per_door"] = df["compressor"] / (df["door_opens"] + 1e-6)
df["compressor_per_on_time"] = df["compressor"] / (df["on_time"] + 1e-6)
df["power_per_on_time"] = df["power"] / (df["on_time"] + 1e-6)

# 2. Creamos variables temporales a partir de la fecha
df["month"] = df["calday"].dt.month
df["dayofweek"] = df["calday"].dt.dayofweek
df["week_of_year"] = df["calday"].dt.isocalendar().week
df["day_of_year"] = df["calday"].dt.dayofyear

# 3. Agregamos features basadas en ventas, compresor y temperatura
df["sales_roll_7"] = df.groupby("cooler_id")["amount"].transform(lambda x: x.rolling(7, min_periods=1).mean())
df["sales_per_time"] = df["amount"] / (df["on_time"] + 1e-6)
df["comp_sales_ratio"] = df["compressor"] / (df["amount"] + 1e-6)
df["comp_mean_7"] = df.groupby("cooler_id")["compressor"].transform(lambda x: x.rolling(7, min_periods=1).mean())
df["comp_std_7"] = df.groupby("cooler_id")["compressor"].transform(lambda x: x.rolling(7, min_periods=1).std().fillna(0))

df["temp_diff_7"] = df.groupby("cooler_id")["temperature"].transform(lambda x: x.diff(7).abs())
df["temp_pct_change_7"] = (df["temperature"] - df.groupby("cooler_id")["temperature"].shift(7)) \
                          / (df.groupby("cooler_id")["temperature"].shift(7) + 1e-6)

# 4. Creamos indicadores binarios basados en los umbrales resultantes del análisis
q75_temp = df["temp_diff_7"].quantile(0.75)
q25_temp = df["temp_diff_7"].quantile(0.25)
q75_comp = df["compressor_per_on_time"].quantile(0.75)

df["high_temp_jump_7"] = (df["temp_diff_7"] > q75_temp).astype(int)
df["low_on_high_comp_ratio"] = ((df["on_time"] < 20) & (df["compressor_per_on_time"] > q75_comp)).astype(int)
df["flat_temp_consistent_use"] = ((df["on_time"] > 23) & (df["temp_diff_7"] < q25_temp)).astype(int)
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
df["is_month_start"] = (df["day_of_year"] % 30 == 1).astype(int)

# 5. Definimos qué columnas usaremos como features para el modelo
features = [
    "day_of_year", "on_time", "amount", "month",
    "compressor_per_door", "sales_per_time", "sales_roll_7",
    "week_of_year", "power_per_on_time", "comp_mean_7",
    "comp_sales_ratio", "door_opens", "temperature", "power"
]
X = df[features]
y = df["warning"]

# 6. Dividimos los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 7. Preparamos datos para XGBoost y entrenamos el modelo
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "seed": 42
}
model = xgb.train(params, dtrain, num_boost_round=120, evals=[(dtrain, "train")], verbose_eval=30)

# 8. Evaluamos el modelo en el set de prueba
y_pred = model.predict(dtest)
pr_auc = average_precision_score(y_test, y_pred)
precision, recall, _ = precision_recall_curve(y_test, y_pred)
pr_curve_auc = auc(recall, precision)

print(f"PR AUC en test: {pr_auc:.3f}")
print(f"Área bajo Curva PR: {pr_curve_auc:.3f}")

# 9. Mostramos visualmente la curva Precision‑Recall
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision‑Recall")
plt.legend()
plt.tight_layout()
plt.savefig("pr_curve_test.png")
print("Curva guardada en 'pr_curve_test.png'")

# 10. Revisamos la importancia de cada feature
importance = model.get_score(importance_type="gain")
imp_df = pd.DataFrame(
    [{"feature": f, "importance": importance.get(f, 0)} for f in X_train.columns]
).sort_values("importance", ascending=False)

print("\nImportancia de las variables:\n", imp_df)
