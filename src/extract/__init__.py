"""Extracción de datos desde la API REST de GitHub (Fase 0).

Solo stdlib: `urllib`, `json`, `time`, `os`. Cero dependencias (§5, §14).
"""

from .github_client import GitHubClient, GitHubError

__all__ = ["GitHubClient", "GitHubError"]
