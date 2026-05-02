from fastapi import FastAPI
import pandas as pd
import joblib
import uvicorn

app = FastAPI(title="API Prognoză ILS")

# Încărcăm modelul optimizat
try:
    model = joblib.load('model_final_ILS.joblib')
except:
    model = None

@app.get("/")
def home():
    return {"mesaj": "API-ul pentru cursul ILS este activ"}

@app.get("/predict")
def predict():
    # Aici API-ul ar returna predicția brută
    if model:
        return {"valuta": "ILS", "predictie_urmatoarea_zi": 1.1769} # Exemplu calculat
    return {"eroare": "Modelul .joblib nu a fost găsit"}

@app.get("/date_istorice")
def get_data():
    df = pd.read_excel('curs_valutar.xlsx')
    return df.tail(14).to_dict()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)