# GitHub Action — triaje automático de issues

Empaqueta el baseline (TF-IDF+logreg, numpy) como Action que clasifica el módulo de cada
issue nuevo. Es la vía que lleva el proyecto de "notebook de tarea" a algo instalable
(§3, §10). Predice con [`label_issue.py`](label_issue.py) y comenta el resultado.

## Dos formas de desplegarlo

1. **En este mismo repo** → ya está activo en
   [`.github/workflows/triage.yml`](../.github/workflows/triage.yml). Usa `checkout` del
   propio repo, predice y **comenta** el módulo en el issue.
2. **En un fork de SciPy** → copia [`triage_fork.yml`](triage_fork.yml) a
   `.github/workflows/triage.yml` del fork. Como el modelo no está en el fork, el workflow
   **clona este repo** para traerlo, y comenta (los labels `scipy.X` no se copian a los forks).

En ambos casos hay que versionar el artefacto `artifact/baseline/` (≈1.5 MB) y el paquete
`src/`. Regenerable con `python -m src.serve.build_artifact`.

## Filosofía
- **Asistente, no automatización total** (§2): el comentario es una sugerencia; el humano
  confirma o corrige, y esa corrección es dato nuevo para reentrenar. (`label_issue.py`
  además marca como auto-etiqueta solo lo de confianza ≥ `CONF_THRESHOLD`, def. 0.9.)
- **1 dependencia en producción**: el contenedor solo necesita `numpy`. Inferencia en µs.

## Prueba local
```bash
ISSUE_TITLE="csr_matrix multiplication wrong with complex dtype" \
ISSUE_BODY="scipy.sparse gives incorrect results" \
python -m action.label_issue
```
