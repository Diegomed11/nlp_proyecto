# Resultados — Fase 3 (neuronales from scratch: BPE + word2vec)

Ambos implementados a mano sobre numpy/stdlib. Reproducir:
`python notebooks/03_bpe_demo.py` y `python notebooks/03b_word2vec_demo.py`.

## BPE — tabla de merges (corpus: 1.42M tokens, 54k palabras únicas)

Primeros merges = subpalabras frecuentes del inglés (`in`, `th`, `the`, `er`…), pero el
dominio asoma rápido: **merge 21 = `__`**, merge 32 = `py`, y aprende `scipy.`, `_matrix`.

Descomposición de identificadores **OOV** (lo que BoW/TF-IDF no pueden representar):

| identificador | subpalabras BPE |
|---|---|
| `csr_matrix` | `cs · r · _matrix` |
| `__init__` | `__ · in · it · __` |
| `scipy.sparse` | `scipy. · sparse` |
| `convergencewarning` | `con · ver · gen · ce · warning` |
| `gammaln` | `g · am · m · al · n` |
| `spsolve` | `sp · sol · ve` |

Subtokens como `_matrix`, `__`, `scipy.`, `warning` **generalizan entre identificadores**:
un identificador nunca visto se representa por piezas conocidas en vez de quedar OOV. Es el
mismo mecanismo de subpalabras que usan los tokenizers subword modernos.

## word2vec — vecinos de dominio (skip-gram + neg. sampling, dim 100, 5 épocas)

vocab=7.847, 11,6M pares. La hipótesis distribucional captura semántica de dominio sin
etiquetas (coseno):

| consulta | vecinos más cercanos |
|---|---|
| **sparse** | matrix, matrices, dense, slicing, **dok, compressed, lil** (formatos dispersos) |
| **optimize** | **cobyla, slsqp, dual_annealing, minimize_scalar, fmin_bfgs, shgo** (algoritmos) |
| **matrix** | matrices, dominant, multiplications, dense, diagonal, sparse, csr |
| **convergence** | stopping, criteria, converged, achieved, successful, converge |
| **minimize** | **rosen, rosen_der** (Rosenbrock), trust, constr, shgo, newton |
| **eigenvalues** | **eigenvectors, eigsh, eigh**, smallest, determinant, decomposition |
| **interpolate** | akima1dinterpolator, rbf, meshgrid, RegularGridInterpolator |
| **memory** | ram, consumption, usage, **leak, leaks**, grows, gb |

Las palabras que comparten **módulo/concepto** caen juntas: formatos de matriz dispersa,
algoritmos de optimización, primitivas de álgebra lineal. Eso es exactamente la señal
semántica que el modelo de bolsa (TF-IDF) ignora — la hipótesis para que ayude donde el
nombre literal del módulo NO aparece.

## Notas
- Entrenamiento word2vec ~20 min (cuello: `np.add.at` en el scatter del gradiente). Para
  la Fase 5 conviene cachear los vectores entrenados.
- Extensiones pendientes (opcionales): comparación vs embeddings genéricos (GloVe) para
  mostrar *domain shift*/polisemia (`special`, `sparse`), y visualización 2D (PCA/t-SNE).
