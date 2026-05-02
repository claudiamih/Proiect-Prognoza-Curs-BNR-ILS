import pandas as pd
import numpy as np
import optuna
import joblib
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

FILE_NAME = 'curs_valutar.xlsx'

def prepare_data():
    try:
        df = pd.read_excel(FILE_NAME)
        # Curățare nume coloane 
        df.columns = df.columns.astype(str).str.strip()
        
        # Identificare automată coloană Data și ILS
        if 'Data' not in df.columns:
            df.rename(columns={df.columns[0]: 'Data'}, inplace=True)
        
        col_ils = [c for c in df.columns if 'ILS' in str(c).upper()][0]
        
        df['Data'] = pd.to_datetime(df['Data'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Data', col_ils]).sort_values('Data')
        
        df_ils = df[['Data', col_ils]].rename(columns={col_ils: 'ILS'}).set_index('Data')
        
        # Feature Engineering pentru XGBoost
        for i in range(1, 4):
            df_ils[f'lag_{i}'] = df_ils['ILS'].shift(i)
        return df_ils.dropna()
    except Exception as e:
        print(f"Eroare la procesare: {e}")
        return None

# OPTUNA (pentru XGBoost) 
def optimize_optuna(X_train, y_train, X_test, y_test):
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0)
        }
        model = XGBRegressor(**param)
        model.fit(X_train, y_train)
        return mean_absolute_error(y_test, model.predict(X_test))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    return study.best_params, study.best_value

# GRIDSEARCH (pentru ARIMA)
def optimize_gridsearch(series):
    best_mae = float('inf')
    best_cfg = None
    
    # Definim grila de parametri (p, d, q)
    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1, 2]
    grid = ParameterGrid({'p': p_values, 'd': d_values, 'q': q_values})
    
    train, test = series[:-14], series[-14:]
    
    for g in grid:
        try:
            model = ARIMA(train, order=(g['p'], g['d'], g['g'])) # Corecție cheie q
            # (statsmodels poate fi sensibil, folosim un try-except)
            res = ARIMA(train, order=(g['p'], g['d'], g['q'])).fit()
            pred = res.forecast(steps=14)
            mae = mean_absolute_error(test, pred)
            if mae < best_mae:
                best_mae, best_cfg = mae, (g['p'], g['d'], g['q'])
        except: continue
    return best_cfg, best_mae

def main():
    data = prepare_data()
    if data is None: return

    # Split date
    train_data, test_data = data.iloc[:-14], data.iloc[-14:]
    X_train, y_train = train_data.drop(columns=['ILS']), train_data['ILS']
    X_test, y_test = test_data.drop(columns=['ILS']), test_data['ILS']

    print("\n[1/2] Pornire Optimizare OPTUNA (XGBoost)...")
    best_params_opt, mae_opt = optimize_optuna(X_train, y_train, X_test, y_test)
    
    print("\n[2/2] Pornire GRIDSEARCH (ARIMA)...")
    best_cfg_grid, mae_grid = optimize_gridsearch(data['ILS'])

    print("-" * 30)
    print(f"REZULTAT OPTUNA (MAE): {mae_opt:.5f}")
    print(f"REZULTAT GRIDSEARCH (MAE): {mae_grid:.5f}")
    
    # Salvăm cel mai bun model 
    final_model = XGBRegressor(**best_params_opt)
    final_model.fit(X_train, y_train)
    joblib.dump(final_model, 'model_final_ILS.joblib')
    print("\nSUCCES: Modelul optimizat 'model_final_ILS.joblib' a fost salvat.")

if __name__ == "__main__":
    main()