# Triaje automático de issues open source con NLP
### Predicción de módulo afectado en SciPy con representaciones from scratch

> Reporte técnico (estructura §12 del plan). Resultados detallados por fase en
> `report/resultados_fase{2,3,5}.md`, decisiones del tokenizer en
> `report/tokenizer_decisiones.md`, dataset en `data/datasheet.md`.

---

## 1. Introducción y motivación

SciPy tiene ~11.500 issues y un puñado de maintainers voluntarios. El triaje manual
—leer un issue, adivinar qué módulo toca, ponerle labels y decidir quién lo revisa— es el
cuello de botella. Este proyecto construye un asistente que, al abrirse un issue, predice
el **módulo afectado** (`stats`, `sparse`, `optimize`…) para pre-clasificarlo.

El dominio es la materia prima ideal para NLP porque un issue **no es texto natural**: es
code-switching entre prosa en inglés y código, tracebacks de decenas de líneas,
identificadores (`scipy.sparse.linalg`, `csr_matrix`), paths, versiones, hashes,
`@menciones` y `#1234`. EDA sobre 11.488 issues: 61% traen bloque de código, 48% código
inline, 15% traceback. Tratar ese registro bien **es** la demostración de dominio; un
`.lower().split()` lo destruye.

## 2. Pregunta de investigación e hipótesis

> **¿Cuánto del trabajo lo hace la representación del texto vs. el modelo encima? ¿Qué
> representación basta para enrutar issues de este dominio, y dónde está su límite?**

Hipótesis: en el régimen donde el issue **nombra** su módulo, una representación mínima
(detectar el nombre) basta; el valor de modelar el texto aparece donde el módulo se
describe por **síntomas**, sin nombrarse. La "escalera" de representaciones (BoW → TF-IDF →
n-gramas → BPE → word2vec, todo from scratch) existe para responder esto con números y
análisis de errores, no para perseguir SOTA.

## 3. Trabajo relacionado

Triaje/bug-localization (asignación de componentes y reviewers en repos grandes);
clasificación de texto técnico mixto código/prosa; segmentación subword (BPE, Sennrich et
al. 2016) para manejo de OOV; semántica distribucional (word2vec, Mikolov et al. 2013).

## 4. Datos

- **Fuente:** API REST de GitHub (no scraping), cliente `urllib` stdlib con paginación por
  **cursor** (la paginación por `page` corta en 10.000 items) y caché resumible.
- **Extracción:** 25.408 items de `scipy/scipy` (11.488 issues + 13.920 PRs).
- **Ground truth = etiqueta directa.** El EDA reveló que **70,5% de los issues ya traen un
  label de módulo `scipy.X`** puesto por los maintainers (cobertura consistente 62-77% todos
  los años, incl. 2025/2026). Decisión de alcance: usar esa etiqueta humana directa en vez
  de weak supervision por archivos de PRs. Dataset curado: **8.095 issues, 20 módulos**,
  multi-label (solo 1,1% con >1 módulo). Granularidad `scipy.X` (`sparse.linalg`→`sparse`).
- **Split TEMPORAL** (no aleatorio): train ≤2024 (**7.396**), eval 2025+ (**699**). Simula
  predecir el futuro y no filtra vocabulario de releases entre train y test. Detalle y
  sesgos (desbalance stats 1675 → datasets 4; spike Trac 2013) en el datasheet.

## 5. Metodología

**Tokenizer técnico (Fase 1).** Pipeline: segmentar (prosa / code_block / inline_code /
traceback) → normalizar → partir. Normaliza lo de alta cardinalidad a tokens especiales
(`<VERSION> <PATH> <HASH> <URL> <ISSUEREF> <USER> <NUM>`) y conserva identificadores
informativos. Decisión central y configurable: el **traceback** es el 43% de los tokens de
los issues que lo traen → se ablaciona keep/drop/special. Detección por líneas que cubre
CPython, IPython y Trac (100% de los issues con marcador).

**Escalera de representaciones (from scratch sobre numpy):** BoW → TF-IDF → n-gramas → BPE
→ word2vec, todo implementado a mano (incluida una matriz dispersa CSR con matmul
vectorizado y una regresión logística multi-label con Adam y pesos de clase acotados). La
lógica de NLP es propia; `numpy` es solo motor de álgebra.

## 6. Experimentos y resultados

Matriz de ablation bajo el split temporal (micro/macro-F1, subset accuracy):

| modelo | macroF1 | macroF1·sup | microF1 | subset |
|---|--:|--:|--:|--:|
| majority | 0.014 | 0.017 | 0.166 | 0.162 |
| keyword (nombra el módulo) | 0.616 | 0.725 | 0.832 | 0.717 |
| BoW + logreg | 0.442 | 0.519 | 0.711 | 0.581 |
| **TF-IDF(1) + logreg** | **0.489** | **0.575** | **0.737** | **0.611** |
| TF-IDF(1,2) + logreg | 0.411 | 0.484 | 0.723 | 0.589 |
| TF-IDF sin split-dotted | 0.405 | 0.476 | 0.683 | 0.557 |

Hallazgos: (i) **TF-IDF > BoW** → el IDF aporta. (ii) **Splitear identificadores con punto
ayuda** (+0.08 macro): expone `stats` de `scipy.stats.norm`. (iii) **Los bigramas empeoran**:
la señal de módulo es unigrama. (iv) El **keyword es un baseline durísimo** (micro 0.83):
los issues nombran su módulo, así que un match casi-oráculo es difícil de superar en el caso
medio.

Las neuronales from scratch (Fase 3) se evalúan cualitativamente: BPE aprende subpalabras
reutilizables (`_matrix`, `__`, `scipy.`) y descompone OOV (`csr_matrix`→`cs·r·_matrix`);
word2vec agrupa por dominio sin etiquetas (`sparse`~dok/lil/csr; `optimize`~slsqp/cobyla;
`minimize`~rosen) — la señal semántica que el modelo de bolsa ignora.

## 7. Análisis de errores lingüístico

El resultado central. Particionando eval según si el módulo se **nombra** en el texto:

| subset (micro-F1) | keyword | TF-IDF |
|---|--:|--:|
| todo (699) | 0.832 | 0.737 |
| nombre presente (655) | 0.859 | 0.746 |
| **nombre AUSENTE (44)** | **0.000** | **0.602** |

Cuando el módulo **no se menciona**, el keyword **colapsa a 0** (no puede), y solo una
representación que generaliza —TF-IDF— se sostiene (0.60): predice desde identificadores y
vocabulario que co-ocurren con el módulo (`scalar_search_wolfe2`→optimize, `lfilter`→signal).
Las confusiones clásicas son por **vocabulario compartido**: `sparse↔linalg` ×10 simétrica
(álgebra lineal dispersa y densa comparten léxico), tal como anticipaba la hipótesis.

## 8. Despliegue y trade-offs

El baseline TF-IDF+logreg se serializa en un artefacto de **1,5 MB** y se sirve con
`http.server` de la stdlib y **numpy como única dependencia** — inferencia en µs. Se empaqueta
además como **GitHub Action** que etiqueta issues nuevos, auto-etiquetando solo con confianza
≥0.9 (asistente, no automatización total; lo dudoso escala al humano, cuya corrección es dato
nuevo para reentrenar).

## 9. Conclusiones

**La representación hace casi todo el trabajo.** Cuando el módulo se nombra, un keyword
casi-oráculo ya da micro 0.83 y el clasificador apenas tiene que modelar nada. El valor de
una representación aprendida (TF-IDF, y por encima BPE/word2vec que capturan subpalabra y
semántica) aparece exactamente donde el nombre desaparece: ahí el keyword cae a 0 y TF-IDF
se sostiene en 0.60. La respuesta a la pregunta es condicional y medible: para el grueso del
triaje basta con detectar el nombre; modelar el texto importa en la cola difícil —issues
descritos por síntomas— que es donde el maintainer más tiempo pierde.

**Trabajo futuro:** representación con embeddings word2vec promediados como features de
clasificación; weak supervision por archivos de PRs para etiquetar el 30% sin label y
extender a PyMC (comparación de dominio); severidad por proxies; reviewer por `git blame`;
calibración por clase y umbral por validación temporal.
