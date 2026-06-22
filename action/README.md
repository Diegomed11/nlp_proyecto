# GitHub Action — triaje automático de issues

Empaqueta el baseline (TF-IDF+logreg, numpy) como Action que etiqueta el módulo de cada
issue nuevo. Es la vía que lleva el proyecto de "notebook de tarea" a algo instalable
(§3, §10).

## Uso

1. Copia [`triage.yml`](triage.yml) a `.github/workflows/triage.yml` del repo destino.
2. Asegúrate de versionar el artefacto del baseline: `artifact/baseline/` (≈1.5 MB) y
   el paquete `src/`. Regenerable con `python -m src.serve.build_artifact`.
3. En cada `issues.opened`, el workflow:
   - instala **solo numpy**,
   - predice el módulo con [`action/label_issue.py`](label_issue.py),
   - aplica el label `scipy.<módulo>` **solo si la confianza ≥ 0.9** (umbral configurable
     en `CONF_THRESHOLD`); lo dudoso se deja al maintainer.

## Filosofía
- **Asistente, no automatización total** (§2): auto-etiqueta solo con confianza alta;
  el resto escala al humano, cuya corrección es dato nuevo para reentrenar.
- **1 dependencia en producción**: el contenedor solo necesita `numpy`. Inferencia en µs.

## Prueba local
```bash
ISSUE_TITLE="csr_matrix multiplication wrong with complex dtype" \
ISSUE_BODY="scipy.sparse gives incorrect results" \
CONF_THRESHOLD=0.9 python -m action.label_issue
```
