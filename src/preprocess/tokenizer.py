"""Tokenizer técnico para issues de GitHub

Un issue de SciPy no es texto natural: es code-switching entre prosa en inglés y
código, tracebacks, identificadores (`scipy.sparse.linalg`, `csr_matrix`), paths,
versiones, hashes, `#refs` y `@menciones`. Un `.lower().split()` destruye justo lo
informativo. Este tokenizer trata ese registro de forma explícita y *defendible*.

Tubería:  segmentar (código/prosa/traceback)  →  normalizar  →  partir en tokens.

DECISIONES 

1. **Segmentar antes de tokenizar.** Se separan bloques ``` ``` ```, código inline
   `` `x` `` y tracebacks de la prosa, para poder tratarlos distinto (p. ej. *dropear*
   el traceback) y para no mezclar la sintaxis de código con la gramática del inglés.

2. **Normalizar lo de alta cardinalidad a tokens especiales** para no reventar el
   vocabulario con cosas únicas que no generalizan:
     `<URL> <PATH> <VERSION> <HASH> <ISSUEREF> <USER> <NUM>`
   Un path `scipy/sparse/_compressed.py` o un sha `a3f9e21` aparecen una sola vez; como
   tipo (`<PATH>`, `<HASH>`) sí generalizan.

3. **Conservar identificadores informativos.** `scipy.sparse`, `csr_matrix`, `pm.sample`
   se mantienen enteros: son la señal más directa del módulo. Los nombres con punto NO
   se normalizan a `<PATH>` (no llevan `/`); se distinguen de los paths de archivo.

4. **Versión vs número.** Solo se trata como `<VERSION>` lo inequívoco (3 componentes
   `1.2.3`, prefijo `v`, o sufijo `rc/dev/...`). `3.14` se deja como `<NUM>`: una
   versión de 2 componentes es ambigua y normalizarla daría falsos positivos.

5. **Traceback: ¿señal o ruido?** Decisión EXPLÍCITA y configurable (`traceback=`
   keep|drop|special). Se prueban ambos en los experimentos (§7): un traceback nombra
   archivos (`_compressed.py`→sparse) y excepciones (`LinAlgError`→linalg), pero también
   mete cientos de tokens de stack que pueden ahogar el título.

Config por flags (`TokenizerConfig`) para correr ablations sin tocar el código.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

T_URL = "<URL>"
T_PATH = "<PATH>"
T_VERSION = "<VERSION>"
T_HASH = "<HASH>"
T_REF = "<ISSUEREF>"
T_USER = "<USER>"
T_NUM = "<NUM>"
T_TRACEBACK = "<TRACEBACK>"

FENCE_RE = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~")
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")

# Traceback: en este corpus conviven el formato CPython estándar, el de IPython
# (frames a columna 0 `...file.py in func(...)`, líneas numeradas, `-->`) y los
# migrados de Trac. Un solo regex rígido falla; se detecta por líneas: ancla en el
# marcador y se consume hasta la línea de excepción, una línea de prosa, o un doble
# salto. Es heurístico a propósito (se mide y documenta su cobertura).
_TRACE_MARKER = re.compile(r"Traceback \(most recent call last\)")
_EXC_LINE = re.compile(
    r"^[ \t]*[A-Za-z_][\w.]*"
    r"(?:Error|Exception|Warning|Interrupt|Exit|Iteration|Overflow|Stop|Fault|Timeout)\b"
)
_FRAME_HINT = re.compile(r'File "|-->|<ipython|<module>|, line \d+|\.py\b', re.IGNORECASE)
_NUMBERED = re.compile(r"^\s*\d+\s")


def _is_trace_line(ln: str) -> bool:
    """¿La línea parece parte de un traceback (no de la prosa)?"""
    if not ln.strip():
        return False  # los blancos se manejan aparte
    if ln[:1] in " \t":              # indentada
        return True
    return bool(_FRAME_HINT.search(ln) or _NUMBERED.match(ln))


def _traceback_spans(text: str, max_lines: int = 200) -> list[tuple[int, int]]:
    """Spans `(start, end)` de tracebacks, por detección línea a línea."""
    lines = text.splitlines(keepends=True)
    offsets, acc = [], 0
    for ln in lines:
        offsets.append(acc)
        acc += len(ln)
    offsets.append(acc)

    spans: list[tuple[int, int]] = []
    n, i = len(lines), 0
    while i < n:
        if not _TRACE_MARKER.search(lines[i]):
            i += 1
            continue
        start = offsets[i]
        j, blanks, consumed = i + 1, 0, 0
        while j < n and consumed < max_lines:
            ln = lines[j]
            if not ln.strip():
                blanks += 1
                if blanks >= 2:
                    break
                j += 1
                consumed += 1
                continue
            blanks = 0
            if _EXC_LINE.match(ln):
                j += 1  # incluir la línea de excepción y terminar
                break
            if _is_trace_line(ln):
                j += 1
                consumed += 1
                continue
            break  # línea de prosa → fin del traceback
        spans.append((start, offsets[min(j, n)]))
        i = j if j > i else i + 1
    return spans


# Frames de un traceback: archivo, línea y función. CPython estándar + IPython.
_FRAME_CPYTHON = re.compile(r'File "([^"]+)", line (\d+), in (\S+)')
_FRAME_IPYTHON = re.compile(r"^[> ]*([^\s][^\n]*?\.py)c?o?\s+in\s+(\w+)", re.MULTILINE)


def traceback_frames(text: str) -> list[dict]:
    """Extrae los frames `(file, line, function)` de los tracebacks del texto.

    Sirve para localizar el problema: el frame más profundo dentro del proyecto suele
    ser el sospechoso. Devuelve dicts {file, line, function} en orden del traceback.
    """
    frames: list[dict] = []
    for s, e in _traceback_spans(text):
        block = text[s:e]
        matched = False
        for f, ln, fn in _FRAME_CPYTHON.findall(block):
            frames.append({"file": f, "line": int(ln), "function": fn})
            matched = True
        if not matched:  # formato IPython (sin línea en la cabecera del frame)
            for f, fn in _FRAME_IPYTHON.findall(block):
                frames.append({"file": f, "line": None, "function": fn})
    return frames


# --- Normalización (orden = más específico primero) -------------------------
URL_RE = re.compile(r"https?://[^\s<>()\[\]]+")
# Path de archivo: lleva '/' o '\' y termina en .ext (opcional :línea). No matchea
# nombres con punto sin slash (esos son identificadores: scipy.sparse.linalg).
PATH_RE = re.compile(
    r"(?:[A-Za-z]:\\|\.{0,2}/)?(?:[\w.\-]+[\\/])+[\w.\-]+\.[A-Za-z]\w*(?::\d+)?"
)
VERSION_RE = re.compile(
    r"\b(?:v\d+(?:\.\d+)+|\d+\.\d+\.\d+(?:[.\-]?(?:a|b|rc|dev|post|alpha|beta)\d*)?)\b"
)
HEXADDR_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")          # dirección de memoria
HASH_RE = re.compile(r"\b(?=[0-9a-f]*\d)[0-9a-f]{7,40}\b")  # sha/hex con ≥1 dígito
REF_RE = re.compile(r"#\d+")                             # issue/PR ref
USER_RE = re.compile(r"@[A-Za-z0-9_\-]+")               # @mención
NUM_RE = re.compile(r"\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b")

# --- Partición en tokens ----------------------------------------------------
# Orden: token especial | identificador con punto | palabra/identificador | número.
TOKEN_RE = re.compile(
    r"<[A-Z_]+>|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+|[A-Za-z_]\w*|\d+"
)


@dataclass
class TokenizerConfig:
    lowercase: bool = True          # baja todo menos los <TOKENS> especiales
    normalize_numbers: bool = True  # números sueltos → <NUM>
    traceback: str = "keep"         # keep | drop | special
    code: str = "keep"              # keep | drop  (bloques ``` y código inline)
    split_dotted: bool = False      # scipy.sparse → [scipy, sparse] (lo hará BPE luego)


class Tokenizer:
    """Tokenizer técnico configurable. `tokenize(text) -> list[str]`."""

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.cfg = config or TokenizerConfig()

    # -- segmentación -----------------------------------------------------
    def segment(self, text: str) -> list[tuple[str, str]]:
        """Parte el texto en spans `(kind, chunk)` en orden.

        kind ∈ {prose, code_block, inline_code, traceback}. Resuelve solapes por
        prioridad: bloque de código > traceback > código inline.
        """
        occupied: list[tuple[int, int]] = []

        def overlaps(s: int, e: int) -> bool:
            return any(s < oe and os < e for os, oe in occupied)

        spans: list[tuple[int, int, str]] = []
        for m in FENCE_RE.finditer(text):
            # Un fence que contiene el marcador es un traceback (lo gobierna el flag
            # `traceback`, no `code`): así la ablación señal-vs-ruido cubre TODOS los
            # tracebacks, no solo el ~7% que va sin fence.
            kind = "traceback" if _TRACE_MARKER.search(m.group()) else "code_block"
            spans.append((m.start(), m.end(), kind))
            occupied.append((m.start(), m.end()))
        for s, e in _traceback_spans(text):
            if not overlaps(s, e):
                spans.append((s, e, "traceback"))
                occupied.append((s, e))
        for m in INLINE_CODE_RE.finditer(text):
            if not overlaps(m.start(), m.end()):
                spans.append((m.start(), m.end(), "inline_code"))
                occupied.append((m.start(), m.end()))

        spans.sort()
        result: list[tuple[str, str]] = []
        pos = 0
        for s, e, kind in spans:
            if s > pos:
                result.append(("prose", text[pos:s]))
            result.append((kind, text[s:e]))
            pos = e
        if pos < len(text):
            result.append(("prose", text[pos:]))
        return result

    # -- normalización ----------------------------------------------------
    def normalize(self, text: str) -> str:
        text = URL_RE.sub(T_URL, text)
        text = PATH_RE.sub(T_PATH, text)
        text = VERSION_RE.sub(T_VERSION, text)
        text = HEXADDR_RE.sub(T_HASH, text)
        text = HASH_RE.sub(T_HASH, text)
        text = REF_RE.sub(T_REF, text)
        text = USER_RE.sub(T_USER, text)
        if self.cfg.normalize_numbers:
            text = NUM_RE.sub(T_NUM, text)
        return text

    # -- partición --------------------------------------------------------
    def _split(self, text: str) -> list[str]:
        out: list[str] = []
        for tok in TOKEN_RE.findall(text):
            if tok.startswith("<") and tok.endswith(">"):
                out.append(tok)  # token especial: intacto
            elif self.cfg.split_dotted and "." in tok:
                out.extend(p.lower() if self.cfg.lowercase else p for p in tok.split(".") if p)
            else:
                out.append(tok.lower() if self.cfg.lowercase else tok)
        return out

    # -- API --------------------------------------------------------------
    def tokenize(self, text: str | None) -> list[str]:
        if not text:
            return []
        tokens: list[str] = []
        for kind, chunk in self.segment(text):
            if kind == "traceback":
                if self.cfg.traceback == "drop":
                    continue
                if self.cfg.traceback == "special":
                    tokens.append(T_TRACEBACK)
                    continue
            if kind in ("code_block", "inline_code") and self.cfg.code == "drop":
                continue
            tokens.extend(self._split(self.normalize(chunk)))
        return tokens

    def tokenize_typed(self, text: str | None) -> list[tuple[str, str]]:
        """Como `tokenize` pero devuelve `(token, kind)` para análisis/visualización."""
        if not text:
            return []
        out: list[tuple[str, str]] = []
        for kind, chunk in self.segment(text):
            if kind == "traceback" and self.cfg.traceback == "drop":
                continue
            if kind == "traceback" and self.cfg.traceback == "special":
                out.append((T_TRACEBACK, kind))
                continue
            if kind in ("code_block", "inline_code") and self.cfg.code == "drop":
                continue
            for t in self._split(self.normalize(chunk)):
                out.append((t, kind))
        return out


def tokenize(text: str | None, config: TokenizerConfig | None = None) -> list[str]:
    """Atajo funcional."""
    return Tokenizer(config).tokenize(text)
