# Preprocessing - Previsione Pluviometrica Emilia-Romagna (2014–2024)

## Obiettivo

Preparare i dati settimanali dell'altezza pluviometrica (in mm) in Emilia-Romagna per effettuare previsioni con modelli
statistici e di machine learning. I dati coprono il periodo dal **2014 al 2024** (11 anni × 52 settimane).

---

## Struttura del Dataset

- Ogni **colonna** rappresenta un anno, da `2014` a `2024`.
- Ogni **riga** rappresenta una settimana (1–52).
- Le celle contengono la quantità di pioggia caduta in mm.

Esempio:

| Settimana | 2014 | 2015 | ... | 2023 | 2024 |
|-----------|------|------|-----|------|------|
| 1         | 17.3 | 26.1 | ... | 22.4 | 19.7 |
| 2         | 21.5 | 33.0 | ... | 16.8 | 25.9 |
| ...       | ...  | ...  | ... | ...  | ...  |
| 52        | 24.9 | 27.7 | ... | 28.3 | 30.1 |

---

## Fasi del Preprocessing

### 1. Caricamento del Dataset

```python
df = pd.read_csv('Pioggia_Settimanale_Emilia-Romagna.csv')
```

### 2. Creazione della Serie Temporale Unificata

Concateniamo i dati di tutte le stagioni in un’unica lista data (usata per analisi generali).

```python
data = []
for i in range(len(df.columns)):
    data.extend(df.iloc[:, i].to_numpy())
```

### 3. Divisione Train/Test

train: Dati dal 2014 al 2023 (10 anni)
test: Dati del 2024

```python
dftrain = df.iloc[:, :-1]
train = []
for i in range(dftrain.shape[1]):
    train.extend(dftrain.iloc[:, i].to_numpy())

test = df.iloc[:, -1].to_numpy()
```

### 4. Statistiche Descrittive

Visualizzazione e analisi di base:

```python
print(f'Min: {min(data)} mm')
print(f'Max: {max(data)} mm')
print(f'Mean: {np.mean(data):.2f} mm')
print(f'Median: {np.median(data):.2f} mm')
print(f'Standard Deviation: {np.std(data):.2f} mm')
```

### 5. Grafico Temporale
Visualizzazione della serie pluviometrica complessiva:

```python
plt.plot(data)
plt.title('Altezza Pluviometrica Settimanale - Emilia-Romagna 2014-2024')
plt.grid()
plt.show()
```

### 6. Autocorrelazione (ACF)
Verifica delle correlazioni tra settimane distanti (necessaria per ARIMA/SARIMA):

```python
plot_acf(np.array(data))
plt.title("Autocorrelazione - Pioggia Settimanale")
plt.grid()
plt.show()
```

### 7.Test di Stazionarietà (ADF)
Si applica l'Augmented Dickey-Fuller test per verificare se la serie è stazionaria (necessario per ARIMA/SARIMA):

```python
adf_result = adfuller(data)
print(f'ADF stats: {adf_result[0]}, P-value: {adf_result[1]}')
```

Se p-value < 0.05, la serie è stazionaria e non serve differenziazione.
Risultato del nostro dataset:
ADF Statistic ≈ -8.59
p-value ≈ 7.16 × 10⁻¹⁴ → Serie stazionaria