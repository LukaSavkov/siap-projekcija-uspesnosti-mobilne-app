import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- FUNKCIJE ZA ČIŠĆENJE ---
def clean_size(size_val):
    """Parsira veličinu aplikacije u megabajte."""
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

def load_and_process_data(filepath):
    """Učitava, čisti i priprema podatke."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        return None
    
    df = df[df['Rating Count'] > 0].copy()
    df = df.dropna(subset=['Rating'])
    
    # Sređivanje veličine
    df['Size_MB'] = df['Size'].apply(clean_size)
    mean_size = df['Size_MB'].mean()
    df['Size_MB'] = df['Size_MB'].fillna(mean_size)

    # Sređivanje instalacija
    if df['Installs'].dtype == 'O': 
        df['Installs'] = df['Installs'].astype(str).str.replace('+', '', regex=False)
        df['Installs'] = df['Installs'].str.replace(',', '', regex=False)
        df = df[df['Installs'] != 'Free']
        df['Installs'] = pd.to_numeric(df['Installs'])

    # Sređivanje cene
    if df['Price'].dtype == 'O':
        df['Price'] = df['Price'].astype(str).str.replace('$', '', regex=False)
        df['Price'] = pd.to_numeric(df['Price'])

    return df

# --- FUNKCIJE ZA VIZUELIZACIJU (NOVO) ---

def plot_rating_distribution(df):
    """Crta histogram ocena."""
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Rating'], bins=20, kde=True, color='skyblue')
    plt.title('Distribucija ocena aplikacija')
    plt.xlabel('Rating')
    plt.ylabel('Minimum Installs')
    plt.show()

def plot_correlations(df):
    """Crta heatmap korelaciju numeričkih atributa."""
    numeric_cols = ['Rating', 'Size_MB', 'Minimum Installs', 'Price']
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Korelaciona matrica')
    plt.show()

def plot_category_impact(df):
    """Crta boxplot ocena po kategorijama."""
    plt.figure(figsize=(14, 6))
    sorted_cats = df.groupby('Category')['Rating'].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x='Category', y='Rating', hue='Category', order=sorted_cats, palette='viridis', legend=False)
    plt.xticks(rotation=90)
    plt.title('Distribucija ocena po kategorijama')
    plt.show()

def plot_price_impact(df):
    """Crta odnos plaćenih i besplatnih aplikacija."""
    df_temp = df.copy()
    df_temp['Type'] = df_temp['Price'].apply(lambda x: 'Paid' if x > 0 else 'Free')
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_temp, x='Type', y='Rating', hue='Type', palette='Set2', legend=False)
    plt.title('Poređenje ocena: besplatne vs plaćene')
    plt.show()

def plot_size_distribution(df):
    """Prikazuje distribuciju veličine aplikacija (logaritamska skala)."""
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Size_MB'], bins=30, kde=True, log_scale=True, color='green')
    plt.title('Distribucija veličine aplikacija')
    plt.xlabel('Size_MB')
    plt.ylabel('Minimum Installs')
    plt.show()

def plot_installs_vs_rating(df):
    """Prikazuje odnos popularnosti (broja instalacija) i ocene."""
    plt.figure(figsize=(10, 6))
    
    sns.scatterplot(data=df, x='Minimum Installs', y='Rating', alpha=0.1, color='purple')
    
    plt.xscale('log')
    plt.title('Da li su popularnije aplikacije bolje ocenjene?')
    plt.xlabel('Minimum Installs')
    plt.ylabel('Rating')
    plt.show()

def plot_content_rating_impact(df):
    """Prikazuje distribuciju ocena prema uzrasnom ograničenju (Content Rating)."""
    plt.figure(figsize=(10, 6))
    
    sorted_content = df.groupby('Content Rating')['Rating'].median().sort_values(ascending=False).index
    
    sns.boxplot(data=df, x='Content Rating', y='Rating', order=sorted_content, palette='coolwarm', hue='Content Rating', legend=False)
    plt.title('Ocene prema uzrasnom ograničenju')
    plt.show()

def plot_feature_importance(model, feature_names):
    """Crta važnost atributa iz modela."""
    feature_imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=feature_imp, y=feature_imp.index, hue=feature_imp.index, palette='viridis', legend=False)
    plt.title('Važnost atributa')
    plt.xlabel('Značaj')
    plt.show()

def print_dataset_stats(df):
    """Ispisuje osnovnu statistiku."""
    print(f"Ukupan broj aplikacija: {df.shape[0]}")
    print(f"Broj atributa: {df.shape[1]}")
    free = len(df[df['Price'] == 0])
    total = len(df)
    print(f"Besplatne: {free:,} ({free/total:.2%})")
    print(f"Plaćene:   {total-free:,} ({(total-free)/total:.2%})")