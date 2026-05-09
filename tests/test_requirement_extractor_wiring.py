"""Tests for requirement extractor dependency wiring."""

from collections.abc import Iterator

import pytest

from smartour.api import dependencies
from smartour.application.requirement_extractor import RuleBasedRequirementExtractor
from smartour.integrations.requirement_model import RequirementModelExtractor


@pytest.fixture(autouse=True)
def clear_dependency_caches() -> Iterator[None]:
    """
    Clear cached dependency factories before and after each test.

    Yields:
        Control to the test.
    """
    dependencies.get_settings.cache_clear()
    dependencies.get_requirement_extractor.cache_clear()
    yield
    dependencies.get_settings.cache_clear()
    dependencies.get_requirement_extractor.cache_clear()


def test_requirement_extractor_uses_model_when_path_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that model configuration selects the supervised extractor.
    """
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "test-key")
    monkeypatch.setenv("REQUIREMENT_MODEL_PATH", "models/requirement_model/quick")
    monkeypatch.setenv("REQUIREMENT_MODEL_CONFIDENCE_THRESHOLD", "0.42")

    extractor = dependencies.get_requirement_extractor()

    assert isinstance(extractor, RequirementModelExtractor)
    assert extractor.model_path.as_posix() == "models/requirement_model/quick"
    assert extractor.confidence_threshold == 0.42


def test_requirement_extractor_uses_development_fallback_without_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that local development can run without a trained model artifact.
    """
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "test-key")
    monkeypatch.delenv("REQUIREMENT_MODEL_PATH", raising=False)

    extractor = dependencies.get_requirement_extractor()

    assert isinstance(extractor, RuleBasedRequirementExtractor)


def test_requirement_extractor_requires_model_when_fallback_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that production-style configuration fails clearly without a model path.
    """
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "test-key")
    monkeypatch.delenv("REQUIREMENT_MODEL_PATH", raising=False)
    monkeypatch.setenv("REQUIREMENT_MODEL_DEVELOPMENT_FALLBACK_ENABLED", "false")

    with pytest.raises(RuntimeError, match="REQUIREMENT_MODEL_PATH is required"):
        dependencies.get_requirement_extractor()
