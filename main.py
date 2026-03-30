import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


from ann.dense import Dense
from ann.train import train, predict
from ann.helper import standardize
from ann.batch_norm import BatchNorm
from ann.dropout import Dropout


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
    size_str = str(size_val).replace(',', '')
    if 'M' in size_str:
        return float(size_str.replace('M', ''))
    elif 'k' in size_str:
        return float(size_str.replace('k', '')) / 1024
    elif 'G' in size_str:
        return float(size_str.replace('G', '')) * 1024
    elif 'Varies with device' in size_str:
        return np.nan
    else:
        try:
            return float(size_str)
        except:
            return np.nan

df['Size_MB'] = df['Size'].apply(clean_size)
mean_size = df['Size_MB'].mean()
df['Size_MB'] = df['Size_MB'].fillna(mean_size)

features = [
    'Category', 'Size_MB', 'Minimum Installs', 'Price', 
    'Content Rating', 'Ad Supported', 'In App Purchases',
    'Editors Choice', 'Rating Count', 'Last Updated', 'Scraped Time'
]
target = 'Rating'

df_model = df[features + [target]].copy()
df_model.dropna(inplace=True)

# ---------------------------------------------------------
# 3. PRIPREMA ZA MODEL (ENCODING I TRANSFORMACIJE)
# ---------------------------------------------------------

# A) Rad sa datumima: Računanje starosti poslednjeg ažuriranja
df_model['Scraped Time'] = pd.to_datetime(df_model['Scraped Time'], format='mixed')
df_model['Last Updated'] = pd.to_datetime(df_model['Last Updated'], format='mixed')
df_model['Days_Since_Last_Update'] = (df_model['Scraped Time'] - df_model['Last Updated']).dt.days

df_model.drop(['Scraped Time', 'Last Updated'], axis=1, inplace=True)

# B) Logaritamska transformacija (Sada uključuje i Rating Count!)
df_model['Minimum Installs'] = np.log1p(df_model['Minimum Installs'])
df_model['Price'] = np.log1p(df_model['Price'])
df_model['Rating Count'] = np.log1p(df_model['Rating Count'])

# C) Binovanje veličine aplikacije (Podela u 5 grupa)
df_model['Size_Category'] = pd.qcut(
    df_model['Size_MB'], 
    q=5, 
    labels=['Very_Small', 'Small', 'Medium', 'Large', 'Very_Large']
)
df_model.drop('Size_MB', axis=1, inplace=True)

# D) Konverzija boolean vrednosti u int (0 i 1) - Uključen Editors Choice
df_model['Ad Supported'] = df_model['Ad Supported'].astype(int)
df_model['In App Purchases'] = df_model['In App Purchases'].astype(int)
df_model['Editors Choice'] = df_model['Editors Choice'].astype(int)

# E) One-Hot Encoding za sve kategorijske promenljive
df_model = pd.get_dummies(
    df_model, 
    columns=['Category', 'Content Rating', 'Size_Category'], 
    drop_first=True, 
    dtype=int
)

# F) Definisanje X i y matrica
target = 'Rating'
final_features = [col for col in df_model.columns if col != target]

X = df_model[final_features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nDimenzije trening skupa: {X_train.shape}")
print(f"Dimenzije test skupa: {X_test.shape}")

# ---------------------------------------------------------
# 4. TRENIRANJE CUSTOM ANN MODELA
# ---------------------------------------------------------
print("\nPriprema podataka za Veštačku Neuronsku Mrežu (ANN)...")

# 1. Konverzija podataka u NumPy nizove i usklađivanje dimenzija
X_ann_train = X_train.values
X_ann_test = X_test.values
y_ann_train = y_train.values.reshape(-1, 1)
y_ann_test = y_test.values.reshape(-1, 1)

# 2. Standardizacija (Neophodna za rad neuronskih mreža)
X_ann_train_scaled, mu, std = standardize(X_ann_train)
X_ann_test_scaled, _, _ = standardize(X_ann_test, mu, std)

# 3. Definisanje arhitekture mreže
input_dim = X_ann_train_scaled.shape[1]

layers = [
    Dense(input_dim, 128, activation='parametric_relu', optimizer_type='adam'), 
    BatchNorm(),
    Dense(128, 64, activation='parametric_relu', optimizer_type='adam'),
    Dropout(p=0.1),
    Dense(64, 16, activation='parametric_relu', optimizer_type='adam'),
    Dense(16, 1, activation='linear', optimizer_type='adam')
]

print("Treniranje ANN modela...")
# 4. Treniranje modela koristeći MSE (Mean Squared Error) za regresiju
loss_history = train(
    X_ann_train_scaled, 
    y_ann_train, 
    layers, 
    epochs=60,
    learning_rate=0.001, 
    cost_type='mse',
    lr_decay='step_decay',
    batch_size=2048,
    D=15,   
    F=0.8
)

# 5. Predikcija i evaluacija
y_ann_pred = predict(X_ann_test_scaled, layers)

mae_ann = mean_absolute_error(y_ann_test, y_ann_pred)
rmse_ann = np.sqrt(mean_squared_error(y_ann_test, y_ann_pred))
r2_ann = r2_score(y_ann_test, y_ann_pred)

print("\n" + "="*40)
print("REZULTATI CUSTOM ANN MODELA")
print("="*40)
print(f"MAE:  {mae_ann:.4f}")
print(f"RMSE: {rmse_ann:.4f}")
print(f"R2:   {r2_ann:.4f}")
print("="*40)

# 6. Prikaz funkcije gubitka (Loss Curve)
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.title('Kriva učenja ANN modela (MSE Loss)')
plt.xlabel('Epoha')
plt.ylabel('Loss')
plt.show()

# ---------------------------------------------------------
# 8. PRIKAZ KONKRETNIH PREDIKCIJA
# ---------------------------------------------------------
print("\n" + "="*50)
print("PRIMER PREDIKCIJA (STVARNO vs PREDVIĐENO)")
print("="*50)

num_samples = 15
sample_indices = X_test.index[:num_samples]

sample_names = df.loc[sample_indices, 'App Name'].values
sample_actuals = y_ann_test[:num_samples].flatten()
sample_preds = y_ann_pred[:num_samples].flatten()

results_df = pd.DataFrame({
    'Aplikacija': sample_names,
    'Stvarna Ocena': sample_actuals,
    'ANN Predikcija': np.round(sample_preds, 2)
})

results_df['Aplikacija'] = results_df['Aplikacija'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)

print(results_df.to_string(index=False))
print("="*50 + "\n")