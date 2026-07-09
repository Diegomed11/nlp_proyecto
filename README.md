# issue-triage-nlp

Sistema de **triaje automático de issues** de repositorios open source masivos
SciPy : lee título + cuerpo de un issue y predice el **módulo afectado**
(multi-label), con extensiones a severidad y reviewer sugerido.

El proyecto no se evalúa por el clasificador sino por la **tubería texto →
significado**: un tokenizer técnico defendible y una *escalera de representaciones*
implementada desde cero (BoW, TF-IDF, n-grams, BPE, word2vec) y un análisis de errores
lingüístico. Ver [`plan_triaje_nlp.md`](plan_triaje_nlp.md).

> **Pregunta de investigación:** ¿cuánto del trabajo lo hace la representación del
> texto vs. el modelo encima, y qué representación basta para enrutar issues de este
> dominio antes de que su límite aparezca?

## Estado — completo

| Fase | Contenido | Estado | Evidencia |
|---|---|---|---|
| 0 | Extracción GitHub API (stdlib, cursor, resumible) | ✅ | 25.408 items |
| — | Curación + split temporal | ✅ | 8.095 issues, 20 módulos |
| 1 | Tokenizer técnico (código/prosa, traceback) | ✅ | [tokenizer_decisiones](report/tokenizer_decisiones.md) |
| 2 | BoW/TF-IDF/n-grams + logreg numpy | ✅ | [resultados_fase2](report/resultados_fase2.md) |
| 3 | BPE + word2vec from scratch | ✅ | [resultados_fase3](report/resultados_fase3.md) |
| 5 | Ablación + análisis de errores | ✅ | [resultados_fase5](report/resultados_fase5.md) |
| 6 | Despliegue numpy + GitHub Action + reporte | ✅ | [reporte](report/reporte.md), [action](action/) |

**Resultado central** (micro-F1, split temporal): cuando el issue **nombra** su módulo,
el keyword casi-oráculo da 0.83; cuando **no lo nombra** (44 issues), keyword→0.00 (no
puede) pero **TF-IDF→0.60** (generaliza por co-ocurrencias). El valor de modelar el texto
está en la cola difícil. Síntesis completa en [report/reporte.md](report/reporte.md).

## Flujo del proyecto

El pipeline va de texto crudo a módulo predicho en **6 etapas**. Cada carpeta de `src/`
es una etapa, en orden:

```
src/
├── data/             1. DATOS         extracción (GitHub API) + curación + split + EDA
├── preprocess/       2. PREPROCESAR   tokenizer técnico (separa código/prosa, traceback)
├── representations/  3. REPRESENTAR   bow, tfidf, ngrams, bpe, word2vec  (texto → números)
├── models/           4. MODELO        regresión logística multi-label (numpy)
├── eval/             5. EVALUAR       métricas, baselines, ablación, análisis de errores
└── serve/            6. DESPLEGAR     artefacto baseline + http.server (numpy + stdlib)

action/    GitHub Action (triaje en vivo)        notebooks/  demos por etapa (ver su README)
artifact/  modelo baseline serializado (1.5 MB)  report/     reporte técnico + resultados
data/      raw/ (gitignored) · processed/ (dataset curado) · datasheet.md
```

### Correr el pipeline, en orden

```bash
# 1. DATOS  (la extracción requiere GITHUB_TOKEN en .env — lectura pública)
python -m src.data.fetch scipy/scipy     # extrae issues+PRs → data/raw/
python -m src.data.dataset                # curación + split temporal → data/processed/
python -m src.data.eda scipy/scipy        # EDA del dataset

# 2-3. PREPROCESAR + REPRESENTAR  (demos — ver notebooks/README.md)
python notebooks/01_eda.py                # panorama del dataset
python notebooks/02_tokenizer.py          # tokenizer: tabla de decisiones
python notebooks/03_bpe.py                # BPE: merges + OOV
python notebooks/04_word2vec.py           # word2vec: vecinos de dominio (~20 min)

# 4-5. MODELO + EVALUAR
python -m src.eval.experiment             # ablación: baselines vs BoW/TF-IDF/n-grams
python -m src.eval.error_analysis         # análisis de errores (nombre presente vs ausente)

# 6. DESPLEGAR
python -m src.serve.build_artifact        # (re)genera el modelo en artifact/baseline/
python -m src.serve.predict --title "csr_matrix bug" --body "..."   # predicción suelta
python -m src.serve.server                # servicio HTTP local (numpy + stdlib)
```

> **Token (etapa 1):** crea uno en https://github.com/settings/tokens (lectura pública) y
> ponlo en `.env` como `GITHUB_TOKEN=ghp_xxx` (sube el rate limit de 60 a 5000 req/h).
> Lo demás: `pip install -r requirements.txt` (numpy + matplotlib).

## Filosofía de dependencias

La *lógica de NLP* es from scratch. `numpy` es solo motor de álgebra (logreg,
word2vec); `matplotlib` solo para EDA. La extracción y el despliegue corren con la
stdlib (el baseline en producción usa numpy como única dependencia no-stdlib).
