"""Tests for the Smartour FastAPI entrypoint."""

import pytest

import smartour.main as main


def test_configure_windows_event_loop_policy_uses_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify Windows API serving switches away from the Proactor event loop.
    """
    fake_policy = object()
    configured_policies: list[object] = []

    monkeypatch.setattr(main.sys, "platform", "win32")
    monkeypatch.setattr(
        main.asyncio,
        "WindowsSelectorEventLoopPolicy",
        lambda: fake_policy,
        raising=False,
    )
    monkeypatch.setattr(
        main.asyncio,
        "set_event_loop_policy",
        configured_policies.append,
    )

    main._configure_windows_event_loop_policy()

    assert configured_policies == [fake_policy]


def test_configure_windows_event_loop_policy_skips_other_platforms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify non-Windows API serving leaves the event loop policy untouched.
    """
    configured_policies: list[object] = []

    monkeypatch.setattr(main.sys, "platform", "linux")
    monkeypatch.setattr(
        main.asyncio,
        "set_event_loop_policy",
        configured_policies.append,
    )

    main._configure_windows_event_loop_policy()

    assert configured_policies == []
