import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------
# 1. UČITAVANJE PODATAKA
# ---------------------------------------------------------

file_name = 'Google-Playstore.csv' 

try:
    df = pd.read_csv(file_name)
    print(f"Uspešno učitan fajl. Dimenzije: {df.shape}")
except FileNotFoundError:
    print("GREŠKA: Fajl nije pronađen. Proveri putanju ili ime fajla.")

print("\nPrvih 5 redova sirovih podataka:")
print(df.head())

# ---------------------------------------------------------
# 2. ČIŠĆENJE I OBRADA PODATAKA (DATA WRANGLING)
# ---------------------------------------------------------

print(f"\nBroj redova pre čišćenja nula ocena: {len(df)}")
df = df[df['Rating Count'] > 0].copy()
df.dropna(subset=['Rating'], inplace=True)
print(f"Broj redova nakon izbacivanja neocenjenih aplikacija: {len(df)}")

def clean_size(size_val):
    size_str = str(size_val).replace(',', '') # Ukloni zareze
    if 'M' in size_str:
        return float(size_str.replace('M', ''))
    elif 'k' in size_str:
        return float(size_str.replace('k', '')) / 1024  # k u M
    elif 'G' in size_str:
        return float(size_str.replace('G', '')) * 1024  # G u M
    elif 'Varies with device' in size_str:
        return np.nan
    else:
        try:
            return float(size_str)
        except:
            return np.nan

df['Size_MB'] = df['Size'].apply(clean_size)

mean_size = df['Size_MB'].mean()
df['Size_MB'].fillna(mean_size, inplace=True)

features = ['Category', 'Size_MB', 'Minimum Installs', 'Price', 'Content Rating', 'Ad Supported', 'In App Purchases']
target = 'Rating'

df_model = df[features + [target]].copy()

df_model.dropna(inplace=True)

# ---------------------------------------------------------
# 3. EKSPLORATIVNA ANALIZA (EDA) - GRAFICI
# ---------------------------------------------------------
sns.set(style="whitegrid")

plt.figure(figsize=(10, 5))
sns.histplot(df_model['Rating'], bins=20, kde=True, color='skyblue')
plt.title('Distribucija ocena aplikacija (Target)')
plt.xlabel('Rating')
plt.ylabel('Broj Aplikacija')
plt.show()

plt.figure(figsize=(10, 5))
sns.scatterplot(data=df_model[df_model['Price'] < 50], x='Price', y='Rating', alpha=0.3)
plt.title('Odnos Cene i Ocene (za aplikacije < $50)')
plt.show()

# ---------------------------------------------------------
# 4. PRIPREMA ZA MODEL (ENCODING)
# ---------------------------------------------------------

le_cat = LabelEncoder()
df_model['Category'] = le_cat.fit_transform(df_model['Category'].astype(str))

le_content = LabelEncoder()
df_model['Content Rating'] = le_content.fit_transform(df_model['Content Rating'].astype(str))

df_model['Ad Supported'] = df_model['Ad Supported'].astype(int)
df_model['In App Purchases'] = df_model['In App Purchases'].astype(int)

X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nDimenzije trening skupa: {X_train.shape}")
print(f"Dimenzije test skupa: {X_test.shape}")

# ---------------------------------------------------------
# 5. TRENIRANJE BASELINE MODELA (Random Forest)
# ---------------------------------------------------------
print("\nTreniranje Random Forest modela... (ovo može potrajati par sekundi)")

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

# ---------------------------------------------------------
# 6. EVALUACIJA I REZULTATI
# ---------------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "="*40)
print("REZULTATI PRVE KONTROLNE TAČKE")
print("="*40)
print(f"MAE (Srednja apsolutna greška): {mae:.4f}")
print(f"RMSE (Koren srednje kvadratne greške): {rmse:.4f}")
print(f"R2 Score (Koeficijent determinacije): {r2:.4f}")
print("="*40)

feature_imp = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_imp, y=feature_imp.index, palette='viridis')
plt.title('Koji atributi najviše utiču na ocenu?')
plt.xlabel('Score važnosti')
plt.show()

results = pd.DataFrame({'Stvarna Ocena': y_test.values, 'Predviđena': y_pred})
print("\nPrimer predikcija:")
print(results.head(10))