# Notebooks — demos del pipeline (en orden de flujo)

Cada archivo es un **script ejecutable** que muestra una etapa del proyecto. Correr desde
la raíz del repo, p. ej. `python notebooks/02_tokenizer.py`.

| # | archivo | etapa | qué muestra |
|---|---|---|---|
| 01 | `01_eda.py` | Datos | panorama del dataset: split temporal, distribución de módulos, % con código/traceback |
| 02 | `02_tokenizer.py` | Preprocesamiento | tabla de decisiones del tokenizer + manejo de tracebacks (señal vs ruido) |
| 03 | `03_bpe.py` | Representación | BPE: tabla de merges aprendidos + descomposición de identificadores OOV |
| 04 | `04_word2vec.py` | Representación | word2vec: vecinos de dominio (entrena ~20 min) |

La **ablación** y el **análisis de errores** no son notebooks; se corren como módulos:

```bash
python -m src.eval.experiment        # matriz de ablación: baselines vs BoW/TF-IDF/n-grams
python -m src.eval.error_analysis    # análisis de errores: nombre presente vs ausente
```
