"""提示词资源的加载边界与模板插值；不依赖当前工作目录。"""

import tomllib
from importlib.resources import files
from pathlib import Path
from string import Formatter

import pytest

from flowpay.core.json import require_object, require_text
from flowpay.core.prompts import prompt
from flowpay.core.prompts.loader import GROUPS


def test_resources_are_loaded_independently_of_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert "检索" in prompt("tools", "no_search_results")


def test_template_preserves_inserted_braces_and_unicode() -> None:
    error = '无效参数 {"limit": true}'
    assert prompt("agent", "output_correction", error=error) == (
        f"上次输出无法使用：{error} 请纠正后重试。"
    )


@pytest.mark.parametrize("values", [{}, {"wrong": "value"}, {"error": "x", "extra": "y"}])
def test_template_requires_exact_parameters(values: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="parameters"):
        prompt("agent", "output_correction", **values)


def test_unknown_group_and_key_are_explicit_errors() -> None:
    with pytest.raises(ValueError, match="group"):
        prompt("../s1", "system")
    with pytest.raises(ValueError, match="key"):
        prompt("agent", "missing")


def test_every_packaged_template_can_be_rendered() -> None:
    for group in GROUPS:
        resource = files("flowpay.core.prompts").joinpath(f"{group}.toml")
        entries = require_object(tomllib.loads(resource.read_text(encoding="utf-8")))
        for key, value in entries.items():
            text = require_text(value)
            fields = {field for _, field, _, _ in Formatter().parse(text) if field is not None}
            rendered = prompt(group, key, **{field: "测试{原样保留}" for field in fields})
            assert rendered.strip()
            if fields:
                assert "测试{原样保留}" in rendered
