from __future__ import annotations

import pytest

from opencode_tokenstats.component_aliases import ComponentAliasError, load_component_aliases


def test_load_component_aliases_parses_rules_comments_and_whitespace(tmp_path) -> None:
    conf = tmp_path / "components.conf"
    conf.write_text(
        "# component alias rules\n"
        "\n"
        "  make-suite = make-plan , make-implement ,make-test\n"
        "lean-ctx = lean-ctx_ctx_search\n"
    )
    aliases = load_component_aliases(str(conf))
    assert aliases == {
        "make-plan": "make-suite",
        "make-implement": "make-suite",
        "make-test": "make-suite",
        "lean-ctx_ctx_search": "lean-ctx",
    }


def test_load_component_aliases_keys_are_case_insensitive(tmp_path) -> None:
    conf = tmp_path / "components.conf"
    conf.write_text("Mixed = Alpha-Tool\n")
    aliases = load_component_aliases(str(conf))
    assert aliases == {"alpha-tool": "Mixed"}


def test_load_component_aliases_repeated_canonical_targets_combine(tmp_path) -> None:
    conf = tmp_path / "components.conf"
    conf.write_text("suite = one\nsuite = two\n")
    aliases = load_component_aliases(str(conf))
    assert aliases == {"one": "suite", "two": "suite"}


def test_load_component_aliases_repeated_case_variants_combine(tmp_path) -> None:
    conf = tmp_path / "components.conf"
    conf.write_text("Suite = one\nsuite = two\n")
    aliases = load_component_aliases(str(conf))
    assert aliases == {"one": "Suite", "two": "Suite"}


def test_load_component_aliases_conflicting_targets_fail(tmp_path) -> None:
    conf = tmp_path / "components.conf"
    conf.write_text("suite = one\nother = one\n")
    with pytest.raises(ComponentAliasError, match="assigned to both"):
        load_component_aliases(str(conf))


def test_load_component_aliases_malformed_rules_fail(tmp_path) -> None:
    cases = [
        "no-equals-sign\n",
        " = alpha\n",
        "suite =\n",
        "suite = alpha*\n",
    ]
    for index, text in enumerate(cases):
        conf = tmp_path / f"bad-{index}.conf"
        conf.write_text(text)
        with pytest.raises(ComponentAliasError):
            load_component_aliases(str(conf))


def test_load_component_aliases_missing_file_returns_empty(tmp_path) -> None:
    assert load_component_aliases(str(tmp_path / "missing.conf")) == {}


def test_load_component_aliases_lookup_precedence(tmp_path, monkeypatch) -> None:
    explicit = tmp_path / "explicit.conf"
    explicit.write_text("explicit-target = alpha\n")
    env_file = tmp_path / "env.conf"
    env_file.write_text("env-target = alpha\n")
    default = tmp_path / "components.conf"
    default.write_text("default-target = alpha\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPTOKEN_COMPONENT_ALIAS_FILE", str(env_file))

    # Explicit path wins over env var and default.
    assert load_component_aliases(str(explicit)) == {"alpha": "explicit-target"}
    # A missing explicit path falls back to the env var file.
    assert load_component_aliases(str(tmp_path / "missing.conf")) == {"alpha": "env-target"}
    # Env var wins over the current-directory default.
    assert load_component_aliases(None) == {"alpha": "env-target"}

    monkeypatch.delenv("OPTOKEN_COMPONENT_ALIAS_FILE")
    # Without explicit path or env var, ./components.conf is used.
    assert load_component_aliases(None) == {"alpha": "default-target"}
