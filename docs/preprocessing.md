---
layout: default
title: Preprocessing
nav_order: 2
---

# Preprocessing dei Dati

## Struttura dei Dati

I dati storici delle precipitazioni settimanali provengono da serie regionali (es. Emilia-Romagna), organizzate in formato tabellare:

- **Righe**: settimane (1–52)
- **Colonne**: anni (es. 2020, 2021, 2022, 2023, 2024)

Ogni cella contiene la quantità di pioggia settimanale in **mm**.

---

## Visualizzazione Dati Storici

Per verificare coerenza, presenza di pattern stagionali e outlier, è stato prodotto un grafico multiserie per anno:

![Serie Storica](img/dataset.png)

> Il grafico mostra andamenti simili nei diversi anni, con picchi nella stagione invernale e primaverile.

---

## Statistiche Descrittive

I dati grezzi sono stati appiattiti in un'unica serie temporale per facilitare il training e l’analisi. 
Le statistiche ottenute sono:

| Statistica           | Valore   |
|----------------------|----------|
| **Min**              | 0.00 mm  |
| **Max**              | 65.53 mm |
| **Media (Mean)**     | 17.45 mm |
| **Mediana (Median)** | 17.34 mm |
| **Dev. Standard**    | 11.81 mm |

---

## Verifica della Stazionarietà

Per l’analisi di serie temporali, è fondamentale verificare se la serie è **stazionaria** 
(cioè se le sue proprietà statistiche non cambiano nel tempo).

### Autocorrelazione (ACF)

![ACF](img/acf-plot.png)

> L’autocorrelazione decresce gradualmente, segno di una possibile componente stagionale o trend.

### Test di Dickey-Fuller Aumentato (ADF)

```text
ADF stats: -8.82
P-value: 1.79e-14
```

### Conclusione:
La serie è stazionaria secondo il test ADF (p-value < 0.05). 
Non è stato necessario applicare differenziazione o trasformazioni logaritmiche.



