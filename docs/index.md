---
layout: default
title: Homepage
nav_order: 1
---

# Operational Analytics - Forecasting delle Piogge settimanali

## Obiettivo del Progetto

Questo progetto ha l’obiettivo di analizzare e confrontare diversi modelli di **previsione delle precipitazioni settimanali**
per l’anno 2024. 

Il problema trattato è una classica **serie temporale univariata**: dati storici di pioggia (in mm) registrati 
settimanalmente vengono utilizzati per predire i valori futuri.

Il sistema confronta le prestazioni di tre approcci principali:

- **SARIMAX**: modello statistico con stagionalità esplicita,
- **Rete Neurale Feedforward (PyTorch)**: modello non lineare basato su apprendimento supervisionato,
- **XGBoost**: modello basato su boosting di alberi decisionali.

L’obiettivo è determinare quale metodo offra il miglior compromesso tra accuratezza, stabilità e capacità di apprendere
dinamiche stagionali e fluttuazioni settimanali.

---

## Motivazione

La previsione delle piogge è un problema critico per numerosi settori:

- Agricoltura e pianificazione delle irrigazioni,
- Gestione del rischio idrologico (alluvioni, siccità),
- Pianificazione urbana e ambientale.

Utilizzando tecniche moderne di machine learning e modelli statistici classici, il progetto mira a:

- Testare la capacità predittiva di ciascun modello,
- Valutare la coerenza temporale delle previsioni,
- Esplorare tecniche diverse per migliorare l’accuratezza nel dominio delle **serie temporali meteorologiche**.

---

## Dati
I dati storici utilizzati per l’addestramento dei modelli sono stati ottenuti da fonti meteorologiche ufficiali e
certificate. 
Alcune delle fonti di riferimento includono:

- [ARPAE](https://www.arpae.it/it/notizie/archivio/archivio-meteo)
- [ISPRA - Istituto Superiore per la Protezione e la Ricerca Ambientale](https://www.isprambiente.gov.it/)

I dati sono stati aggregati su base settimanale e pre-processati per garantire coerenza temporale e qualità statistica.

- **Frequenza**: settimanale
- **Unità di misura**: mm di pioggia
- **Intervallo temporale**: dati storici dal 2019 fino al 2023, previsione per tutto il 2024

---

## Approccio

Per ogni modello:

- Le previsioni vengono prodotte **in modo ricorsivo**, settimana dopo settimana, per simulare un utilizzo in tempo reale.
- Le performance sono valutate con metriche classiche di regressione:
  - **MAE** (Errore Assoluto Medio),
  - **RMSE** (Errore Quadratico Medio),
  - **MAPE** (Errore Percentuale Medio Assoluto),
  - **Correlazione** tra valori previsti e osservati.

---

## Confronto e Analisi

I modelli sono comparati sulla base di:

- Accuratezza numerica,
- Capacità di catturare dinamiche stagionali,
- Stabilità delle previsioni nel lungo periodo.

Ogni modello ha una sezione dedicata con:

- Descrizione tecnica,
- Codice di implementazione,
- Grafico dei risultati,
- Metriche dettagliate.

---

## Contenuti
- [Preprocessing dei Dati](preprocessing.md)
- [Forecasting con Metodi Statistici (SARIMAX)](forecasting_statistico.md)
- [Forecasting con Reti Neurali](forecasting_neurale.md)
- [Forecasting con Modelli ad Albero Decisionale](forecasting_decision_tree.md)
