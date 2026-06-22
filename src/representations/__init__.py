"""Representaciones del texto (escalera del §7), from scratch sobre numpy."""

from .bow import CountVectorizer
from .matrix import CSR
from .tfidf import TfidfVectorizer

__all__ = ["CSR", "CountVectorizer", "TfidfVectorizer"]
