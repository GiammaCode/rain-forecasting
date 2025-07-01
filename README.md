# Rain Forecasting - Emilia-Romagna

Previsione settimanale delle piogge nella regione Emilia-Romagna, basata su modelli statistici (SARIMA, SARIMAX) e di machine learning (Neural Network, XGBoost).  
Il progetto utilizza dati meteorologici reali ed è pensato per essere facilmente esteso con nuove feature.

## Documentazione 

https://giammacode.github.io/rain-forecasting/

## Come eseguire

```bash
# Clona il progetto
git clone https://github.com/giammaCode/rain-forecasting.git
cd rain-forecasting

# Crea un ambiente virtuale
python -m venv .venv
source .venv/bin/activate  # su Windows: .venv\Scripts\activate

# Installa le dipendenze
pip install -r requirements.txt

# Esegui il progetto
python src/main.py
```
