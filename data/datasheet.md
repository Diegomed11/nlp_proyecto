# Datasheet — Dataset de issues de SciPy etiquetados por módulo

> Estructura de *Datasheets for Datasets* (Gebru et al., 2021). Números reales de la
> extracción del 2026-06-21. Alcance **enfocado a SciPy con etiqueta directa**
> (decisión de proyecto: ver `plan_triaje_nlp.md` y el EDA en `src/eval/eda.py`).

## Motivación
- **¿Para qué se creó?** Entrenar y evaluar un clasificador multi-label de *módulo
  afectado* para triaje de issues de SciPy (núcleo del proyecto, §3 del plan).
- **¿Quién?** Proyecto académico de la materia de NLP.

## Composición
- **Instancia:** un issue de GitHub (título, cuerpo, módulos, metadatos).
- **Fuente bruta:** `scipy/scipy`, **25.408 items** extraídos (11.488 issues + 13.920
  PRs). Los PRs se extrajeron pero **no** se usan en esta tarea.
- **Dataset curado:** **8.095 issues** = los que traen ≥1 label de módulo `scipy.X`
  (70,5% de cobertura, consistente todos los años: 62–77%, incl. 2025 y 2026).
- **Etiqueta (ground truth):** labels de módulo puestos por los maintainers de SciPy
  (p. ej. `scipy.sparse`). **No es weak supervision**: es etiqueta humana directa.
  Granularidad = subpaquete `scipy.X` (`scipy.sparse.linalg` → `sparse`).
- **Clases (20):** stats, optimize, sparse, signal, linalg, special, interpolate,
  spatial, integrate, io, ndimage, cluster, fft, misc, fftpack, odr, _lib, constants,
  differentiate, datasets.
- **Multi-label:** sí, pero cola corta — solo 91 issues (1,1%) tienen >1 módulo.
- **Campos:** number, title, body, modules, created_at, year, state, html_url.

## Proceso de recolección
- **Fuente:** API REST de GitHub (no scraping). Cliente `urllib` stdlib
  (`src/extract/`), paginación por cursor, resumible. ToS de la API respetados.
- **Caché:** JSON crudo en `data/raw/` (gitignored), para no re-pegarle a la API.

## Preprocesamiento / etiquetado
- Filtro: solo issues (no PRs) con ≥1 label `scipy.X`. Folding de submódulos a `X`.
- **Split temporal** (§5): train ≤2024 = **7.396**, eval 2025+ = **699**.

## Distribución de clases (train · eval)
| clase | train | eval | | clase | train | eval |
|---|--:|--:|---|---|--:|--:|
| stats | 1675 | 118 | | io | 289 | 16 |
| optimize | 1030 | 95 | | cluster | 137 | 4 |
| sparse | 958 | 90 | | fft | 62 | 3 |
| signal | 626 | 76 | | misc | 51 | 0 |
| linalg | 544 | 69 | | fftpack | 49 | 0 |
| special | 532 | 68 | | odr | 42 | 5 |
| interpolate | 450 | 67 | | _lib | 27 | 3 |
| spatial | 406 | 55 | | constants | 16 | 1 |
| integrate | 290 | 22 | | differentiate | 6 | 1 |
| ndimage | 289 | 26 | | datasets | 4 | 0 |

## Usos y sesgos conocidos
- **Desbalance fuerte** (stats 1675 → datasets 4): macro-F1 obligatorio; clases raras
  con 0–1 ejemplos en eval (`misc`, `fftpack`, `datasets`) → se reporta el piso del
  macro-F1 y se considerará un umbral de frecuencia mínima en la evaluación.
- **Spike 2013 + "Migrated from Trac"** (1.891 issues): muchos issues migrados del
  tracker viejo en 2013; estilo de texto distinto. Quedan en train (no contaminan eval).
- **Cola larga de longitud** (mediana 1.090 chars, p90 4.515, máx 252k): relevante para
  el truncado de BERT a 512 tokens (pierde info) vs. TF-IDF sobre texto completo.
- **Cobertura parcial:** el 29,5% de issues sin label de módulo no entra (no hay ground
  truth directo); podrían ser objetivos de inferencia, no de train/eval.

## Distribución y mantenimiento
- Curado versionado en `data/processed/scipy/{train,eval}.jsonl` + `meta.json`.
- Regenerable: `python -m src.extract.fetch scipy/scipy && python -m src.dataset`.
