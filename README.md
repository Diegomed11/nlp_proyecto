# issue-triage-nlp

Sistema de **triaje automático de issues** de repositorios open source masivos
(SciPy / PyMC): lee título + cuerpo de un issue y predice el **módulo afectado**
(multi-label), con extensiones a severidad y reviewer sugerido.

El proyecto no se evalúa por el clasificador sino por la **tubería texto →
significado**: un tokenizer técnico defendible y una *escalera de representaciones*
implementada desde cero (BoW, TF-IDF, n-grams, BPE, word2vec, n-gram LM) comparada
contra un transformer fine-tuneado. Ver [`plan_triaje_nlp.md`](plan_triaje_nlp.md).

> **Pregunta de investigación:** ¿cuánto del trabajo lo hace la representación del
> texto vs. el modelo encima, y qué nivel de NLP basta para este dominio antes de
> que el costo de un transformer deje de pagarse?

## Estado

| Fase | Contenido | Estado |
|---|---|---|
| 0 | Extracción de datos (cliente GitHub API, stdlib) | 🚧 en curso |
| 1 | Tokenizer técnico (código/prosa, normalización) | ⬜ |
| 2 | Clásicas: BoW, TF-IDF, n-grams + logreg numpy | ⬜ |
| 3 | Neuronales from scratch: BPE / word2vec | ⬜ |
| 4 | Transformer: fine-tuning + probe de attention | ⬜ |
| 5 | Experimentos: ablation + análisis de errores | ⬜ |
| 6 | Despliegue: ONNX / numpy + GitHub Action | ⬜ |

## Estructura

```
src/extract/      cliente GitHub API (urllib, stdlib) + caché de JSON crudo
src/preprocess/   tokenizer, normalización, segmentación código/prosa
src/representations/  bow, tfidf, ngrams, bpe, word2vec, ngram_lm  (from scratch)
src/models/       logreg.py (numpy)  +  bert_finetune.py (solo Nivel 2)
src/eval/         métricas multi-label, split temporal, análisis de errores
src/serve/        onnxruntime + http.server
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
matplotlib). El transformer y el despliegue tienen sus propios
`requirements-*.txt`.

## Filosofía de dependencias

La *lógica de NLP* es from scratch. `numpy` es solo motor de álgebra (logreg,
word2vec); `matplotlib` solo para EDA; `transformers`/`torch` viven exclusivamente
en el Nivel 2. La extracción y el despliegue corren con la stdlib.
