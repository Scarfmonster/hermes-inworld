"""Regression tests for retrying the TTS steering-rewrite LLM call."""

import importlib
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PLUGIN_DIR = Path(__file__).resolve().parents[1]
_PACKAGE_NAME = "hermes_inworld_under_test"


def _install_agent_stubs() -> None:
    """Provide minimal Hermes modules when hermes-agent is unavailable."""
    agent_pkg = sys.modules.get("agent")
    if agent_pkg is None:
        agent_pkg = types.ModuleType("agent")
        agent_pkg.__path__ = []
        sys.modules["agent"] = agent_pkg

    tts_provider = types.ModuleType("agent.tts_provider")

    class _TTSProvider:
        pass

    setattr(tts_provider, "TTSProvider", _TTSProvider)
    sys.modules["agent.tts_provider"] = tts_provider
    setattr(agent_pkg, "tts_provider", tts_provider)

    auxiliary_client = types.ModuleType("agent.auxiliary_client")
    setattr(auxiliary_client, "call_llm", mock.Mock())
    sys.modules["agent.auxiliary_client"] = auxiliary_client
    setattr(agent_pkg, "auxiliary_client", auxiliary_client)


def _ensure_agent_modules() -> None:
    try:
        importlib.import_module("agent.tts_provider")
        importlib.import_module("agent.auxiliary_client")
    except ImportError:
        _install_agent_stubs()


def _load_tts_module():
    _ensure_agent_modules()
    package = types.ModuleType(_PACKAGE_NAME)
    package.__path__ = [str(PLUGIN_DIR)]
    sys.modules[_PACKAGE_NAME] = package
    return importlib.import_module(f"{_PACKAGE_NAME}.tts")


tts = _load_tts_module()
auxiliary_client = importlib.import_module("agent.auxiliary_client")


class _StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _ResponseError(Exception):
    def __init__(self, status_code: int, headers: dict[str, str]) -> None:
        super().__init__(f"HTTP {status_code}")
        self.response = SimpleNamespace(status_code=status_code, headers=headers)


class _Cancellation(BaseException):
    pass


def _policy(**overrides: object) -> dict[str, object]:
    policy: dict[str, object] = {
        "max_retries": 2,
        "initial_backoff": 1.0,
        "max_backoff": 30.0,
        "backoff_multiplier": 2.0,
        "retry_on_5xx": True,
        "retry_on_429": True,
        "retry_on_network_error": True,
    }
    policy.update(overrides)
    return policy


def _completion(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class TTSRewriteRetryTests(unittest.TestCase):
    def _run_rewrite(self, side_effect, policy=None):
        if policy is None:
            policy = _policy()
        with (
            mock.patch.object(
                auxiliary_client, "call_llm", side_effect=side_effect
            ) as call_llm,
            mock.patch.object(tts.time, "sleep") as sleep,
        ):
            result = tts._rewrite_tts_text("Hello", "SYSTEM PROMPT", policy)
        return result, call_llm, sleep

    def _assert_fallback_without_retry(self, result, call_llm, sleep) -> None:
        self.assertEqual(result, "Hello")
        self.assertEqual(call_llm.call_count, 1)
        sleep.assert_not_called()

    def test_retries_transient_statuses_then_succeeds(self) -> None:
        result, call_llm, sleep = self._run_rewrite(
            [
                _StatusError(500),
                _StatusError(503),
                _completion("[excited] Hello"),
            ]
        )

        self.assertEqual(result, "[excited] Hello")
        self.assertEqual(call_llm.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1.0, 2.0])

    def test_400_is_not_retried(self) -> None:
        result, call_llm, sleep = self._run_rewrite([_StatusError(400)])

        self._assert_fallback_without_retry(result, call_llm, sleep)

    def test_zero_retries_makes_one_attempt(self) -> None:
        result, call_llm, sleep = self._run_rewrite(
            [_StatusError(500)], _policy(max_retries=0)
        )

        self._assert_fallback_without_retry(result, call_llm, sleep)

    def test_429_honors_retry_after_response_header(self) -> None:
        result, call_llm, sleep = self._run_rewrite(
            [
                _ResponseError(429, {"Retry-After": "7"}),
                _completion("rewritten"),
            ],
            _policy(max_retries=1),
        )

        self.assertEqual(result, "rewritten")
        self.assertEqual(call_llm.call_count, 2)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [7.0])

    def test_empty_completion_is_not_retried(self) -> None:
        result, call_llm, sleep = self._run_rewrite([_completion("   ")])

        self._assert_fallback_without_retry(result, call_llm, sleep)

    def test_wrapped_500_is_retried_via_cause_chain(self) -> None:
        wrapped = RuntimeError("provider call failed")
        wrapped.__cause__ = _StatusError(500)

        result, call_llm, sleep = self._run_rewrite(
            [wrapped, _completion("rewritten")], _policy(max_retries=1)
        )

        self.assertEqual(result, "rewritten")
        self.assertEqual(call_llm.call_count, 2)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1.0])

    def test_base_exception_cancellation_propagates(self) -> None:
        with (
            mock.patch.object(
                auxiliary_client, "call_llm", side_effect=_Cancellation("stop")
            ) as call_llm,
            mock.patch.object(tts.time, "sleep") as sleep,
        ):
            with self.assertRaises(_Cancellation):
                tts._rewrite_tts_text("Hello", "SYSTEM PROMPT", _policy())

        self.assertEqual(call_llm.call_count, 1)
        sleep.assert_not_called()

    def test_500_not_retried_when_retry_on_5xx_disabled(self) -> None:
        result, call_llm, sleep = self._run_rewrite(
            [_StatusError(500)], _policy(retry_on_5xx=False)
        )

        self._assert_fallback_without_retry(result, call_llm, sleep)

    def test_429_not_retried_when_retry_on_429_disabled(self) -> None:
        result, call_llm, sleep = self._run_rewrite(
            [_StatusError(429)], _policy(retry_on_429=False)
        )

        self._assert_fallback_without_retry(result, call_llm, sleep)


if __name__ == "__main__":
    unittest.main()
