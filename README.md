# issue-triage-nlp

Sistema de **triaje automático de issues** de repositorios open source masivos
(SciPy / PyMC): lee título + cuerpo de un issue y predice el **módulo afectado**
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

## Cómo correr cada fase

```bash
python -m src.extract.fetch scipy/scipy   # Fase 0 (requiere GITHUB_TOKEN en .env)
python -m src.dataset                      # curación + split temporal
python -m src.eval.eda scipy/scipy         # EDA
python notebooks/02_tokenizer_demo.py      # Fase 1
python -m src.experiment                   # Fase 2 (baselines + clásicas)
python notebooks/03_bpe_demo.py            # Fase 3 — BPE
python notebooks/03b_word2vec_demo.py      # Fase 3 — word2vec
python -m src.error_analysis               # Fase 5
python -m src.serve.build_artifact         # Fase 6 — artefacto baseline
python -m src.serve.server                 # Fase 6 — servicio HTTP (numpy + stdlib)
```

## Estructura

```
src/extract/      cliente GitHub API (urllib, stdlib) + caché de JSON crudo
src/preprocess/   tokenizer, normalización, segmentación código/prosa
src/representations/  bow, tfidf, ngrams, bpe, word2vec  (from scratch)
src/models/       logreg.py (numpy, multi-label)
src/eval/         métricas multi-label, split temporal, análisis de errores
src/serve/        baseline serializado + http.server (numpy + stdlib)
data/             raw/ (gitignored)  ·  processed/  ·  datasheet.md
notebooks/        01_eda … 05_error_analysis
report/           reporte técnico
```

## Uso rápido

### Fase 0 — extracción de datos

La extracción solo usa la **stdlib** (cero dependencias). Necesita un GitHub
Personal Access Token (solo lectura pública) para subir el rate limit de 60 a
5000 req/h.

```bash
# 1. Token: https://github.com/settings/tokens  (scope: public_repo o solo lectura)
export GITHUB_TOKEN=ghp_xxx          # PowerShell: $env:GITHUB_TOKEN="ghp_xxx"

# 2. Extraer issues + PRs de un repo (se cachean en data/raw/<repo>/)
python -m src.extract.fetch scipy/scipy
python -m src.extract.fetch pymc-devs/pymc
```

El resto de fases se instalan con `pip install -r requirements.txt` (numpy +
matplotlib).

## Filosofía de dependencias

La *lógica de NLP* es from scratch. `numpy` es solo motor de álgebra (logreg,
word2vec); `matplotlib` solo para EDA. La extracción y el despliegue corren con la
stdlib (el baseline en producción usa numpy como única dependencia no-stdlib).
