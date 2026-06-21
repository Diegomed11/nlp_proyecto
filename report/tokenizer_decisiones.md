# Tokenizer técnico — tabla de decisiones (Fase 1)

Evidencia del peldaño firma (§7, §17). Cada decisión es justificable con un ejemplo y
configurable por flag para ablacionar. Reproducible: `python notebooks/02_tokenizer_demo.py`.
Implementación: [`src/preprocess/tokenizer.py`](../src/preprocess/tokenizer.py).

## Por qué no `.lower().split()`
Un issue de SciPy es code-switching entre prosa, código, tracebacks, identificadores,
paths, versiones, hashes, `#refs` y `@menciones`. EDA sobre 11.488 issues: **61%** traen
bloque de código, **48%** código inline, **15%** traceback. Aplanar eso destruye la señal.

## Tubería
`segmentar (prosa / code_block / inline_code / traceback)  →  normalizar  →  partir`

## Tabla de decisiones (antes → después)

| Decisión | Regla | Ejemplo |
|---|---|---|
| **Versión ≠ número** | Solo 3 componentes, prefijo `v`, o sufijo `rc/dev` → `<VERSION>`; `3.14` queda `<NUM>` | `numpy 1.2.3` → `numpy <VERSION>` · `pi 3.14` → `pi <NUM>` |
| **Path ≠ identificador** | Con `/` o `\` y `.ext` → `<PATH>`; nombre con puntos sin slash se conserva | `scipy/sparse/_compressed.py:42` → `<PATH>` · `scipy.sparse.linalg.spsolve` → intacto |
| **Conservar identificadores** | snake/dunder/dotted se mantienen enteros (señal directa de módulo) | `csr_matrix`, `complex`, `pm.sample` → intactos |
| **Hash / dirección** | `0x…` y hex de 7–40 con ≥1 dígito → `<HASH>` | `a3f9e21d99`, `0x7f3abc12` → `<HASH>` |
| **Refs y menciones** | `#\d+` → `<ISSUEREF>` · `@user` → `<USER>` | `#1234`, `@rgommers` → `<ISSUEREF> <USER>` |
| **URL** | `https?://…` → `<URL>` | enlaces de Trac → `<URL>` |

Motivo común de los tokens especiales: lo de **alta cardinalidad** (un path, un sha)
aparece una sola vez y no generaliza; como *tipo* (`<PATH>`, `<HASH>`) sí.

## Decisión central: traceback ¿señal o ruido?
Configurable: `traceback = keep | drop | special`. Se prueban ambos en los experimentos.

Hallazgos sobre los 1.033 issues de train con traceback:
- **100% se segmentan** como `traceback` (detector línea-a-línea que cubre CPython
  estándar, IPython y migrados de Trac, fenced y bare por igual).
- **93%** vienen dentro de un fence ```` ``` ```` (uso moderno); ~7% son *bare* (estilo
  IPython/Trac viejo). Un fence con el marcador se re-etiqueta como `traceback` para que
  la ablación cubra **todos**, no solo el 7%.
- **`drop` elimina el 43% de los tokens** en promedio en esos issues → el traceback es
  voluminoso. La pregunta empírica: ¿esos tokens (archivos `_compressed.py`→sparse,
  excepciones `LinAlgError`→linalg) ayudan o ahogan al título? Lo decide el experimento.

## Limitaciones conocidas (honestas)
- El bounding del traceback bare es heurístico (formatos heterogéneos); se corta en la
  línea de excepción, una de prosa o doble salto. Puede sobre/sub-capturar en casos raros.
- Versiones de 2 componentes (`1.10`) caen en `<NUM>` a propósito (evita falsos positivos).
- `split_dotted=False` por defecto: `scipy.sparse` se conserva entero (excelente para
  BoW/TF-IDF); el troceo en subpalabras se delega a BPE (Fase 3).
