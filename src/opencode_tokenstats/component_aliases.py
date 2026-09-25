from __future__ import annotations

import os
from pathlib import Path


class ComponentAliasError(ValueError):
    """Raised when a components.conf rule is malformed or conflicting."""


def load_component_aliases(file_path: str | None = None) -> dict[str, str]:
    """Load component name aliases from components.conf file.

    Format: one rule per line, e.g. 'canonical = alias1,alias2'
    Rules match exact original component names case-insensitively;
    wildcards are not supported.

    Search locations (in order):
    1. Explicit file_path parameter
    2. OPTOKEN_COMPONENT_ALIAS_FILE environment variable
    3. Current working directory: components.conf (only if no explicit path or env var set)

    Repeated rules for the same canonical label (case-insensitive) combine
    their aliases. An alias assigned to two different canonical labels is
    a conflict and raises ComponentAliasError.
    """
    candidates: list[Path] = []
    explicit_path = False

    if file_path:
        candidates.append(Path(file_path))
        explicit_path = True

    env_file = os.environ.get("OPTOKEN_COMPONENT_ALIAS_FILE", "")
    if env_file:
        candidates.append(Path(env_file))
        explicit_path = True

    # Only check default components.conf if no explicit path or env var set
    if not explicit_path:
        candidates.append(Path.cwd() / "components.conf")

    lookup: dict[str, str] = {}  # lowercased original name -> canonical label
    first_spelling: dict[str, str] = {}
    for conf_path in candidates:
        if conf_path.exists():
            for lineno, line in enumerate(conf_path.read_text(encoding="utf-8").splitlines(), start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    raise ComponentAliasError(
                        f"{conf_path}:{lineno}: expected 'canonical = alias1,alias2'"
                    )
                key, val = line.split("=", 1)
                canonical = key.strip()
                if not canonical:
                    raise ComponentAliasError(f"{conf_path}:{lineno}: missing canonical label")
                aliases = [a.strip() for a in val.split(",") if a.strip()]
                if not aliases:
                    raise ComponentAliasError(f"{conf_path}:{lineno}: rule has no aliases")
                canonical_key = canonical.lower()
                spelling = first_spelling.setdefault(canonical_key, canonical)
                for alias in aliases:
                    if "*" in alias:
                        raise ComponentAliasError(
                            f"{conf_path}:{lineno}: wildcards are not supported"
                        )
                    lowered = alias.lower()
                    previous = lookup.get(lowered)
                    if previous is not None and previous.lower() != canonical_key:
                        raise ComponentAliasError(
                            f"{conf_path}:{lineno}: alias '{alias}' is assigned to "
                            f"both '{previous}' and '{canonical}'"
                        )
                    lookup[lowered] = spelling
            break
    return lookup
