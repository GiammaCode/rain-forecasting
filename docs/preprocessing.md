# Documentazione Preprocessing - Piogge Settimanali Emilia-Romagna

## Panoramica

Il modulo `preprocessing.py` gestisce il caricamento, l'analisi esplorativa e la preparazione dei dati delle piogge settimanali dell'Emilia-Romagna per il forecasting. Questo documento descrive in dettaglio tutte le operazioni eseguite sui dati.

## 1. Struttura del Dataset

### 1.1 Formato dei Dati
- **File sorgente**: `Pioggia_Settimanale_EmiliaRomagna.csv`
- **Struttura**: 52 righe (settimane) × 11 colonne (anni 2014-2024)
- **Tipo di dati**: Valori di precipitazione in millimetri (mm)
- **Granularità temporale**: Settimanale (52 settimane per anno)

### 1.2 Organizzazione Temporale
```
Anni di Training: 2014-2023 (10 anni × 52 settimane = 520 osservazioni)
Anno di Test: 2024 (52 settimane)
Dataset Totale: 572 osservazioni settimanali
```

## 2. Caricamento e Trasformazione Dati

### 2.1 Funzione `load_and_preprocess_data()`

La funzione principale esegue le seguenti operazioni:

1. **Caricamento CSV**: Lettura del file usando `pandas.read_csv()`
2. **Trasformazione in Serie Temporale**: Concatenazione delle colonne anno per anno
3. **Divisione Train/Test**: Separazione 2014-2023 (training) e 2024 (test)
4. **Conversione a NumPy Arrays**: Per compatibilità con i modelli di ML

```python
# Esempio di trasformazione
train_data = []
for year in range(2014, 2024):  # 2014-2023 per training
    train_data.extend(df[str(year)].values)
```

## 3. Analisi Esplorativa dei Dati

### 3.1 Funzione `analyze_data()`

Questa funzione produce un'analisi statistica completa per ogni dataset (training e test).

#### Statistiche Calcolate:
- **Lunghezza**: Numero totale di osservazioni
- **Range**: Valori minimo e massimo
- **Tendenza Centrale**: Media e mediana
- **Dispersione**: Deviazione standard
- **Distribuzione**: 25° e 75° percentile
- **Valori Speciali**: Conteggio e percentuale di valori zero

#### Output Esempio:
```
Training Data (2014-2023):
  Lunghezza: 520
  Min: 0.00 mm
  Max: 63.53 mm
  Media: 17.65 mm
  Mediana: 14.23 mm
  Deviazione Standard: 13.45 mm
  Valori zero: 45 (8.7%)
```

### 3.2 Visualizzazioni Prodotte

**Grafico 1: Analisi Training Data** *(Inserire qui: `training_analysis.png`)*
- **Subplot 1**: Serie temporale completa (520 settimane)
- **Subplot 2**: Istogramma della distribuzione
- **Subplot 3**: Box plot per identificare outliers

**Grafico 2: Analisi Test Data** *(Inserire qui: `test_analysis.png`)*
- **Subplot 1**: Serie temporale 2024 (52 settimane)
- **Subplot 2**: Istogramma della distribuzione 2024
- **Subplot 3**: Box plot dei dati 2024

## 4. Gestione Outliers e Valori Anomali

### 4.1 Funzione `handle_outliers()`

#### Metodi di Identificazione Implementati:

**Metodo IQR (Interquartile Range)** - *Default*:
```python
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
```

**Metodo Z-Score**:
```python
z_scores = np.abs(stats.zscore(data))
outliers_mask = z_scores > threshold  # threshold=3
```

#### Filosofia di Gestione:
- **Non Rimozione**: Gli outliers non vengono rimossi dal dataset
- **Motivazione**: Eventi meteorologici estremi (es. alluvioni, periodi di siccità) sono parte naturale della variabilità climatica
- **Approccio**: Identificazione e documentazione, mantenimento nel dataset

### 4.2 Gestione Valori Zero

I valori zero rappresentano settimane senza precipitazioni significative e vengono:
- **Mantenuti**: Fanno parte del pattern naturale
- **Analizzati**: Calcolata frequenza e distribuzione stagionale
- **Documentati**: Percentuale per anno e dataset complessivo

## 5. Test di Stazionarietà

### 5.1 Funzione `check_stationarity()`

Implementa il **Test di Augmented Dickey-Fuller (ADF)**:

#### Ipotesi del Test:
- **H₀ (Null Hypothesis)**: La serie temporale ha una radice unitaria (NON stazionaria)
- **H₁ (Alternative Hypothesis)**: La serie temporale è stazionaria

#### Interpretazione dei Risultati:
```python
if p_value <= 0.05:
    print("Serie STAZIONARIA - Rifiutiamo H₀")
else:
    print("Serie NON stazionaria - Non possiamo rifiutare H₀")
```

#### Output Tipico:
```
Test di Augmented Dickey-Fuller per la stazionarietà:
ADF Statistic: -8.234567
p-value: 0.000001
Valori critici:
    1%: -3.437
    5%: -2.864
    10%: -2.568
Risultato: Rifiutiamo l'ipotesi nulla. La serie è STAZIONARIA.
```

## 6. Analisi di Autocorrelazione

### 6.1 Funzione `visualize_autocorrelation()`

Genera le funzioni di autocorrelazione per identificare pattern temporali e stagionalità.

**Grafico 3: Autocorrelation Analysis** *(Inserire qui: `autocorrelation_analysis.png`)*
- **Subplot 1**: ACF (Autocorrelation Function) - 52 lags
- **Subplot 2**: PACF (Partial Autocorrelation Function) - 52 lags

#### Analisi della Stagionalità:
La funzione calcola correlazioni specifiche per identificare pattern annuali:

```python
Lag 52 settimane (1 anno): correlazione = 0.245
Lag 104 settimane (2 anni): correlazione = 0.187
Lag 156 settimane (3 anni): correlazione = 0.142
```

#### Interpretazione:
- **Lag 52**: Correlazione annuale (stesso periodo dell'anno precedente)
- **Valori Positivi**: Indicano pattern stagionali ricorrenti
- **Decadimento**: La correlazione diminuisce con l'aumentare della distanza temporale

## 7. Funzioni di Supporto

### 7.1 `create_dataset()` - Supervised Learning Transformation

Converte la serie temporale in formato supervisionato per machine learning:

```python
# Esempio con look_back = 3
Input:  [1, 2, 3, 4, 5, 6, 7, 8]
Output: X = [[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7]]
        y = [4, 5, 6, 7, 8]
```

#### Parametri:
- **data**: Serie temporale input
- **look_back**: Numero di time steps da usare come features (default=1)

### 7.2 `forecast_accuracy()` - Metriche di Valutazione

Calcola sei metriche di accuratezza per valutare le previsioni:

1. **MAPE** (Mean Absolute Percentage Error): `mean(|forecast - actual| / |actual|)`
2. **ME** (Mean Error): `mean(forecast - actual)`
3. **MAE** (Mean Absolute Error): `mean(|forecast - actual|)`
4. **MPE** (Mean Percentage Error): `mean((forecast - actual) / actual)`
5. **RMSE** (Root Mean Square Error): `sqrt(mean((forecast - actual)²))`
6. **Correlazione**: `corrcoef(forecast, actual)`

## 8. Considerazioni Metodologiche

### 8.1 Scelte di Design

1. **Preservazione dei Dati Originali**: 
   - Nessuna trasformazione aggressiva
   - Mantenimento della variabilità naturale

2. **Approccio Conservativo agli Outliers**:
   - Eventi estremi sono informativi per il forecasting
   - Rimozione potrebbe compromettere la capacità predittiva

3. **Stagionalità Settimanale**:
   - Riconoscimento del ciclo annuale (52 settimane)
   - Analisi specifica per pattern ricorrenti

### 8.2 Validazioni Implementate

- **Controllo Lunghezza**: Verifica consistenza tra train e test
- **Controllo Valori Mancanti**: Identificazione e gestione
- **Controllo Tipo Dati**: Conversione e validazione numerica
- **Controllo Range**: Identificazione valori anomali o impossibili

## 9. Output del Preprocessing

### 9.1 Dati Preparati
- **train_data**: Array NumPy (520,) - Dati 2014-2023
- **test_data**: Array NumPy (52,) - Dati 2024
- **full_data**: Array NumPy (572,) - Dataset completo

### 9.2 Grafici Generati
1. **Analisi Esplorativa Training** - Serie, istogramma, box plot
2. **Analisi Esplorativa Test** - Serie, istogramma, box plot  
3. **Autocorrelazione** - ACF e PACF plots

### 9.3 Report Statistici
- Statistiche descrittive complete
- Risultati test di stazionarietà
- Analisi outliers e valori zero
- Correlazioni stagionali

## 10. Raccomandazioni per l'Uso

### 10.1 Prima dell'Addestramento
- Verificare i grafici di autocorrelazione per confermare la stagionalità
- Controllare i risultati del test ADF per la stazionarietà
- Esaminare la distribuzione per comprendere la variabilità

### 10.2 Durante il Modeling
- Usare `look_back=52` per catturare la stagionalità annuale
- Considerare la presenza di valori zero nella loss function
- Monitorare le metriche multiple (non solo RMSE)

### 10.3 Validazione dei Risultati
- Confrontare previsioni con pattern stagionali storici
- Verificare che i modelli catturino sia trend che stagionalità
- Analizzare residui per pattern non catturati
