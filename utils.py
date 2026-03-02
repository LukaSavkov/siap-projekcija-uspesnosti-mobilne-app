import pandas as pd
import numpy as np

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

def load_and_process_data(filepath):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"GREŠKA: Fajl '{filepath}' nije pronađen.")
        return None
    
    df = df[df['Rating Count'] > 0].copy()
    
    df = df.dropna(subset=['Rating'])
    
    df['Size_MB'] = df['Size'].apply(clean_size)
    
    mean_size = df['Size_MB'].mean()
    df['Size_MB'] = df['Size_MB'].fillna(mean_size)

    if df['Installs'].dtype == 'O': 
        df['Installs'] = df['Installs'].astype(str).str.replace('+', '', regex=False)
        df['Installs'] = df['Installs'].str.replace(',', '', regex=False)
        df = df[df['Installs'] != 'Free']
        df['Installs'] = pd.to_numeric(df['Installs'])

    if df['Price'].dtype == 'O':
        df['Price'] = df['Price'].astype(str).str.replace('$', '', regex=False)
        df['Price'] = pd.to_numeric(df['Price'])

    return df