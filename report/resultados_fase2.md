# Resultados — Fase 2 (representaciones clásicas + logreg)

Tarea: clasificación multi-label de módulo en SciPy. **Split temporal** (train ≤2024 =
7.396, eval 2025+ = 699). 20 clases (17 con soporte en eval). Logreg numpy (Adam, pesos
de clase acotados cap=10, umbral fijo 0.5 — sin tunear en test). Reproducir:
`python -m src.experiment`.

| modelo | macroF1 | macroF1 (sup) | microF1 | subset acc | hamming | feats |
|---|--:|--:|--:|--:|--:|--:|
| majority | 0.014 | 0.017 | 0.166 | 0.162 | 0.085 | — |
| **keyword** | **0.616** | **0.725** | **0.832** | **0.717** | 0.020 | — |
| BoW + logreg | 0.442 | 0.519 | 0.711 | 0.581 | 0.029 | 7 986 |
| **TF-IDF(1) + logreg** | 0.489 | 0.575 | 0.737 | 0.611 | 0.024 | 7 986 |
| TF-IDF(1,2) + logreg | 0.411 | 0.484 | 0.723 | 0.589 | 0.024 | 36 637 |
| TF-IDF(1) sin split-dotted | 0.405 | 0.476 | 0.683 | 0.557 | 0.029 | 8 443 |

## Hallazgos

1. **El NLP aprende:** todos los modelos aplastan al *majority* (macro 0.49 vs 0.01).
2. **TF-IDF > BoW** (macro 0.489 vs 0.442; micro 0.737 vs 0.711): el IDF aporta — pesar
   por información ayuda. Confirma empíricamente el peldaño "IDF como medida de información".
3. **Splitear identificadores con punto importa** (+0.084 macro, +0.054 micro): exponer
   `stats` de `scipy.stats.norm` da el feature de módulo directo; sin split queda
   fragmentado en tokens raros. Decisión de tokenizer con impacto medible.
4. **Los bigramas NO ayudan aquí** (macro 0.411 < 0.489): cuadruplican el vocabulario y
   empeoran. La señal de módulo vive en identificadores unigrama, no en el orden local.
   Hallazgo honesto: n-gramas no siempre suman.
5. **El keyword es el rey (macro 0.62, micro 0.83) y le gana a todo logreg.** Es el
   resultado central de §9.2: *los issues nombran su propio módulo*, así que un match de
   keyword es casi un oráculo. El TF-IDF logreg queda competitivo en micro (0.74) pero
   detrás en macro — sobre todo en módulos raros, donde no hay datos para aprender pero el
   nombre igual aparece en el texto.

## Por clase (TF-IDF(1), eval)
Clases frecuentes muy bien: stats F1 0.84, optimize 0.80, sparse 0.79, interpolate 0.79,
special 0.73. Clases raras se hunden: `fft`/`_lib`/`differentiate` F1 0 (3, 3, 1 ejemplos
en eval); `misc`/`fftpack`/`datasets` sin soporte en eval. → el macro-F1 está topado por
el largo tail; se reporta también `macroF1 (sup)` sobre las clases con soporte.

## Pregunta que deja abierta para las siguientes fases
El keyword gana cuando el módulo se nombra literalmente. ¿Qué pasa donde el nombre **no**
aparece —issues descritos por síntomas, no por módulo? El análisis de errores (Fase 5)
mide ese régimen: ahí está el valor incremental de una representación que generaliza
(TF-IDF) sobre el match literal.
