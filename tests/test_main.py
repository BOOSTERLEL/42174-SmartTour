"""Tests for the Smartour FastAPI entrypoint."""

import asyncio
from typing import Any, cast

import pytest

import smartour.main as main


class WindowsConnectionResetError(ConnectionResetError):
    """
    Test double for Windows connection reset errors that expose winerror.
    """

    winerror: int


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


def test_windows_proactor_connection_reset_context_is_detected() -> None:
    """
    Verify the known Windows Proactor reset callback is recognized.
    """
    exception = WindowsConnectionResetError(10054, "reset")
    exception.winerror = 10054
    context = {
        "exception": exception,
        "handle": "<Handle _ProactorBasePipeTransport._call_connection_lost(None)>",
        "message": "Exception in callback",
    }

    assert main._is_windows_proactor_connection_reset(context)


def test_windows_connection_reset_handler_delegates_other_errors() -> None:
    """
    Verify the Windows exception handler only suppresses the known reset callback.
    """
    delegated_contexts: list[dict[str, Any]] = []

    def previous_handler(
        loop: asyncio.AbstractEventLoop, context: dict[str, Any]
    ) -> None:
        """
        Record delegated exception contexts.

        Args:
            loop: The asyncio event loop.
            context: The delegated exception context.
        """
        delegated_contexts.append(context)

    handler = main._windows_connection_reset_exception_handler(previous_handler)
    reset_exception = WindowsConnectionResetError(10054, "reset")
    reset_exception.winerror = 10054
    reset_context = {
        "exception": reset_exception,
        "handle": "<Handle _ProactorBasePipeTransport._call_connection_lost(None)>",
        "message": "Exception in callback",
    }
    other_context = {
        "exception": RuntimeError("boom"),
        "handle": "<Handle other>",
        "message": "Exception in callback",
    }

    handler(cast(asyncio.AbstractEventLoop, object()), reset_context)
    handler(cast(asyncio.AbstractEventLoop, object()), other_context)

    assert delegated_contexts == [other_context]


@pytest.mark.asyncio
async def test_configure_windows_connection_reset_handler_installs_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify Windows startup installs an asyncio exception handler.
    """
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    monkeypatch.setattr(main.sys, "platform", "win32")
    loop.set_exception_handler(None)

    try:
        main._configure_windows_connection_reset_handler()

        assert loop.get_exception_handler() is not None
    finally:
        loop.set_exception_handler(previous_handler)
