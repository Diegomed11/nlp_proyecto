# Resultados — Fase 5 (análisis de errores lingüístico)

El corazón de la evaluación NLP (§9, §17). Reproducir: `python -m src.eval.error_analysis`.

## ¿Sirve la representación donde el nombre del módulo NO aparece?

Partición de los 699 issues de eval según si **algún módulo verdadero se nombra** en el
texto: 655 lo nombran, 44 no.

| subset | keyword | TF-IDF + logreg |
|---|--:|--:|
| todo (699) | 0.832 | 0.737 |
| nombre presente (655) | 0.859 | 0.746 |
| **nombre AUSENTE (44)** | **0.000** | **0.602** |

(micro-F1). **El resultado central.** El keyword es un baseline durísimo *cuando el issue
nombra su módulo* (micro 0.86), pero **colapsa a 0** cuando no lo nombra — por construcción
no puede predecir sin la palabra. **TF-IDF aguanta 0.60 ahí**: generaliza desde términos
que co-ocurren con el módulo (nombres de función, vocabulario del área) aunque el módulo no
se mencione. Ese es, medido, el valor de una representación aprendida sobre el match literal.

## Confusiones por vocabulario compartido (TF-IDF, issues de 1 módulo)

| confusión | veces |
|---|--:|
| sparse ↔ linalg | 10 + 10 |
| stats → special | 5 |
| linalg → special | 5 |
| special → signal | 4 |

`sparse ↔ linalg` es simétrica y la más fuerte — exactamente lo que anticipaba el plan
(§9): álgebra lineal dispersa y densa comparten vocabulario (`matrix`, `solve`,
`eigenvalue`, `factorization`). `special`/`stats`/`signal` comparten léxico matemático.

## Ejemplos en vivo (nombre ausente · keyword no puede · TF-IDF acierta)

| issue | título (recortado) | módulo |
|---|---|---|
| #22544 | "condition in `scalar_search_wolfe2` can never be true" | optimize |
| #22571 | "`lfilter` returns a leaked state from a previous call" | signal |
| #23076 | "Incorrect use of NumPy scalar types instead of np.dtype in _spu…" | sparse |
| #23229 | "Parallel optimization of batch of parameters" | optimize |

`scalar_search_wolfe2` es una rutina interna de line-search (optimize) y `lfilter` es de
señales: TF-IDF aprendió que esos identificadores co-ocurren con su módulo aunque el
nombre `optimize`/`signal` no aparezca. El keyword, sin el token del módulo, no acierta.

## Síntesis para la pregunta de investigación
- En el **régimen fácil** (655/699 issues nombran su módulo) un keyword casi-oráculo ya
  da micro 0.86: la representación apenas tiene que trabajar.
- En el **régimen difícil** (nombre ausente) el keyword cae a 0 y solo una representación
  que generaliza —TF-IDF aquí— se sostiene (0.60). **El nivel de NLP que hace falta depende
  del régimen:** para el grueso del triaje basta con detectar el nombre; el valor de
  modelar el texto está en los issues descritos por síntomas, que son los que más tiempo le
  cuestan al maintainer.
