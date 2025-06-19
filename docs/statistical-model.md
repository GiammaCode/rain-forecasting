## valutazione auto arima

 ARIMA(0,0,0)(0,1,1)[52] intercept   : AIC=inf, Time=24.98 sec
 ARIMA(0,0,0)(0,1,0)[52] intercept   : AIC=1660.704, Time=0.24 sec
 ARIMA(1,0,0)(1,1,0)[52] intercept   : AIC=1612.680, Time=21.66 sec
 ARIMA(0,0,1)(0,1,1)[52] intercept   : AIC=inf, Time=29.05 sec
 ARIMA(0,0,0)(0,1,0)[52]             : AIC=1658.743, Time=0.16 sec
 ARIMA(1,0,0)(0,1,0)[52] intercept   : AIC=1661.627, Time=0.93 sec
 ARIMA(1,0,0)(1,1,1)[52] intercept   : AIC=inf, Time=58.69 sec

 | Parte          | Significato                                                             |
| -------------- | ----------------------------------------------------------------------- |
| `ARIMA(p,d,q)` | Componente **non stagionale**: autoregressiva, differenza, media mobile |
| `(P,D,Q)[s]`   | Componente **stagionale**: stessa cosa ma con periodo `s` (qui `s=52`)  |
| `intercept`    | Se il modello include una costante/intercetta                           |
| `AIC=...`      | Valore dell’AIC: più **basso** è, meglio è                              |
| `Time=... sec` | Quanto ha impiegato quel modello per il fitting                         |
