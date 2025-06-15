import pandas as pd

df = pd.read_csv("coolers_completo_sin_nulos.csv", parse_dates=["calday"])

# 1. Eficiencia energética
df["power_per_on_time"] = df["power"] / (df["on_time"] + 1e-6)

# 2. Tendencia de consumo: media móvil (7 días) por cooler_id
df["power_roll_7"] = df.groupby("cooler_id")["power"].transform(lambda x: x.rolling(7, min_periods=1).mean())

# 3. Cambio de temperatura a 7 días
df["temp_prev_7"] = df.groupby("cooler_id")["temperature"].shift(7)
df["temp_diff_7"] = (df["temperature"] - df["temp_prev_7"]).abs()

# 4. Estacionalidad extendida
df["week_of_year"] = df["calday"].dt.isocalendar().week
df["day_of_year"] = df["calday"].dt.dayofyear

# Guardamos resultados
df.to_csv("coolers_features_ext.csv", index=False)
print("✅ Se agregaron nuevas features al dataset.")
