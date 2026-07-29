# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the MiniMax provider.

The MiniMax provider is a thin subclass of ``OpenAILanguageModel`` that
overrides only the default ``base_url`` and ``model_id``. These tests mock
the underlying OpenAI client so they can run offline while still verifying
that:

* construction wires ``base_url`` and ``model_id`` correctly,
* factory routing resolves ``MiniMax-*`` and ``minimax-*`` model IDs to
  :class:`MiniMaxLanguageModel` (and not to the generic OpenAI provider),
* the ``MINIMAX_PRIORITY`` strictly exceeds ``OPENAI_PRIORITY`` so the
  router prefers MiniMax whenever both patterns match,
* ``infer()`` honors LangExtract's structured-output contract: it forwards
  ``response_format={"type": "json_object"}`` for JSON output and uses
  raw text (no fences) for ``FormatType.JSON``, and it leaves the prompt
  text unchanged (no ``prompt_description`` rewriting),
* invalid arguments raise the same ``InferenceConfigError`` that
  ``OpenAILanguageModel`` raises.
"""

from __future__ import annotations

import dataclasses
import sys
from typing import Any
from unittest import mock

from absl.testing import absltest

from langextract import exceptions
from langextract.core import base_model
from langextract.core import data
from langextract.core import types as core_types
from langextract.providers import minimax
from langextract.providers import openai
from langextract.providers import patterns
from langextract.providers import router


def _make_message(content: str) -> mock.Mock:
  """Build a minimal OpenAI chat completion choice/message mock."""
  message = mock.Mock()
  message.content = content
  choice = mock.Mock()
  choice.message = message
  response = mock.Mock()
  response.choices = [choice]
  return response


def _patched_openai_module(test_case: absltest.TestCase) -> mock.Mock:
  """Patch ``openai.OpenAI`` so the constructor never touches the network.

  ``OpenAILanguageModel.__init__`` does ``import openai`` locally, so the
  reference is to the ``openai`` SDK module. We patch the SDK class itself
  via ``sys.modules`` so the patched reference is picked up by the local
  ``import openai`` inside ``OpenAILanguageModel``.
  """
  real_openai = sys.modules.get("openai")
  fake_module = mock.Mock()
  fake_module.OpenAI.side_effect = lambda **kw: mock.Mock(**kw)
  sys.modules["openai"] = fake_module
  test_case.addCleanup(
      lambda: (
          sys.modules.__setitem__("openai", real_openai)
          if real_openai is not None
          else sys.modules.pop("openai", None)
      )
  )
  return fake_module


class _RecordingClient:
  """Captures every chat.completions.create call for assertion."""

  def __init__(self, response_content: str = '{"answer": 42}') -> None:
    self.response_content = response_content
    self.calls: list[dict[str, Any]] = []

  def completions(self):  # pragma: no cover - trivial accessor
    return self

  def create(self, **kwargs):  # noqa: D401 - test double
    self.calls.append(kwargs)
    return _make_message(self.response_content)


class MiniMaxConstructionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    _patched_openai_module(self)

  def test_defaults_use_minimax_base_url_and_model_id(self):
    model = minimax.MiniMaxLanguageModel(api_key="key")
    self.assertEqual(
        model.base_url, minimax.MiniMaxLanguageModel.DEFAULT_BASE_URL
    )
    self.assertEqual(
        model.base_url, "https://api.minimax.io/v1"
    )
    self.assertEqual(
        model.model_id, minimax.MiniMaxLanguageModel.DEFAULT_MODEL_ID
    )
    self.assertEqual(model.model_id, "MiniMax-M2.5")

  def test_explicit_arguments_override_defaults(self):
    model = minimax.MiniMaxLanguageModel(
        model_id="MiniMax-Reasoning-1",
        api_key="key",
        base_url="https://staging.minimax.io/v1",
        organization="org-123",
    )
    self.assertEqual(model.model_id, "MiniMax-Reasoning-1")
    self.assertEqual(model.base_url, "https://staging.minimax.io/v1")
    self.assertEqual(model.organization, "org-123")

  def test_missing_api_key_raises_inference_config_error(self):
    with self.assertRaises(exceptions.InferenceConfigError):
      minimax.MiniMaxLanguageModel(api_key=None)

  def test_inherits_openai_language_model(self):
    self.assertTrue(
        issubclass(minimax.MiniMaxLanguageModel, openai.OpenAILanguageModel)
    )
    self.assertTrue(
        issubclass(minimax.MiniMaxLanguageModel, base_model.BaseLanguageModel)
    )

  def test_is_a_dataclass_subclass(self):
    # OpenAI provider uses ``dataclasses.dataclass(init=False)``; the MiniMax
    # subclass must remain compatible so future attribute additions keep
    # working.
    self.assertTrue(dataclasses.is_dataclass(minimax.MiniMaxLanguageModel))


class MiniMaxRoutingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    router.clear()
    # Force a clean re-import so the decorator-driven registration runs.
    import importlib  # pylint: disable=import-outside-toplevel

    importlib.reload(minimax)
    importlib.reload(openai)
    router.clear()
    # Built-in lazy path mirrors what ``load_builtins_once`` does in
    # ``langextract.providers``.
    from langextract.providers import builtin_registry  # pylint: disable=import-outside-toplevel

    for config in builtin_registry.BUILTIN_PROVIDERS:
      router.register_lazy(
          *config["patterns"],
          target=config["target"],
          priority=config["priority"],
      )

  def tearDown(self):
    router.clear()
    super().tearDown()

  def test_minimax_pattern_resolves_to_minimax_provider(self):
    self.assertIs(
        router.resolve("MiniMax-M2.5"), minimax.MiniMaxLanguageModel
    )
    self.assertIs(
        router.resolve("minimax-m2.5"), minimax.MiniMaxLanguageModel
    )

  def test_openai_patterns_still_resolve_to_openai_provider(self):
    self.assertIs(router.resolve("gpt-4o-mini"), openai.OpenAILanguageModel)
    self.assertIs(router.resolve("gpt-5-turbo"), openai.OpenAILanguageModel)

  def test_minimax_priority_strictly_above_openai_priority(self):
    self.assertGreater(patterns.MINIMAX_PRIORITY, patterns.OPENAI_PRIORITY)

  def test_minimax_provider_not_overridden_by_openai_for_minimax_ids(self):
    """Regression: the merged PR lost factory routing for ``MiniMax-*``."""
    entry = router.resolve("MiniMax-M2.5")
    self.assertIsNot(entry, openai.OpenAILanguageModel)


class MiniMaxInferTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    real_openai = sys.modules.get("openai")
    self.recorder = _RecordingClient()
    fake_client = mock.Mock()
    fake_client.chat.completions.create = self.recorder.create
    fake_module = mock.Mock()
    fake_module.OpenAI.return_value = fake_client
    sys.modules["openai"] = fake_module
    self.addCleanup(
        lambda: (
            sys.modules.__setitem__("openai", real_openai)
            if real_openai is not None
            else sys.modules.pop("openai", None)
        )
    )

  def test_infer_emits_json_response_format_for_json_output(self):
    model = minimax.MiniMaxLanguageModel(
        model_id="MiniMax-M2.5", api_key="key"
    )
    list(model.infer(["Prompt A"]))
    self.assertEqual(len(self.recorder.calls), 1)
    call = self.recorder.calls[0]
    self.assertEqual(call["model"], "MiniMax-M2.5")
    self.assertEqual(call["response_format"], {"type": "json_object"})
    self.assertEqual(call["messages"][-1]["role"], "user")
    self.assertEqual(call["messages"][-1]["content"], "Prompt A")

  def test_infer_does_not_rewrite_prompt_with_prompt_description(self):
    """The merged PR appended ``f"{prompt_description}\\n\\nText: {text}"``.

    LangExtract composes prompts centrally; the provider must forward them
    verbatim so examples, fences, and ``format_type`` formatting survive.
    """
    model = minimax.MiniMaxLanguageModel(api_key="key")
    list(model.infer(["Raw prompt text"]))
    call = self.recorder.calls[0]
    self.assertEqual(call["messages"][-1]["content"], "Raw prompt text")
    self.assertNotIn("Text:", call["messages"][-1]["content"])

  def test_infer_yields_scored_output_with_full_response_text(self):
    self.recorder.response_content = '{"items": ["a", "b"]}'
    model = minimax.MiniMaxLanguageModel(api_key="key")
    results = list(model.infer(["Prompt"]))
    self.assertLen(results, 1)
    self.assertLen(results[0], 1)
    scored = results[0][0]
    self.assertIsInstance(scored, core_types.ScoredOutput)
    self.assertEqual(scored.score, 1.0)
    self.assertEqual(scored.output, '{"items": ["a", "b"]}')

  def test_infer_with_yaml_format_does_not_force_response_format(self):
    model = minimax.MiniMaxLanguageModel(
        api_key="key", format_type=data.FormatType.YAML
    )
    list(model.infer(["Prompt"]))
    call = self.recorder.calls[0]
    # YAML mode must not request JSON response_format; the OpenAI base class
    # already enforces this and the MiniMax subclass must not regress it.
    self.assertNotIn("response_format", call)
    self.assertIn(call["messages"][0]["role"], {"system", "user"})

  def test_infer_runs_prompts_in_parallel_for_batches(self):
    """Verify the OpenAI-style parallel batch path is inherited unchanged."""
    model = minimax.MiniMaxLanguageModel(
        api_key="key", max_workers=4
    )
    list(model.infer(["p1", "p2", "p3"]))
    self.assertLen(self.recorder.calls, 3)

  def test_infer_propagates_temperature_from_runtime_kwargs(self):
    model = minimax.MiniMaxLanguageModel(api_key="key", temperature=0.7)
    list(model.infer(["Prompt"], temperature=0.2))
    call = self.recorder.calls[0]
    self.assertEqual(call["temperature"], 0.2)


class MiniMaxFactoryTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    router.clear()
    import importlib  # pylint: disable=import-outside-toplevel

    importlib.reload(minimax)
    importlib.reload(openai)
    from langextract.providers import builtin_registry  # pylint: disable=import-outside-toplevel

    for config in builtin_registry.BUILTIN_PROVIDERS:
      router.register_lazy(
          *config["patterns"],
          target=config["target"],
          priority=config["priority"],
      )

  def tearDown(self):
    router.clear()
    super().tearDown()

  def test_factory_explicit_provider_string_returns_minimax_model(self):
    from langextract import factory  # pylint: disable=import-outside-toplevel

    real_openai = sys.modules.get("openai")
    fake_module = mock.Mock()
    fake_module.OpenAI.return_value = mock.Mock()
    sys.modules["openai"] = fake_module
    try:
      config = factory.ModelConfig(
          model_id="MiniMax-M2.5",
          provider="MiniMaxLanguageModel",
          provider_kwargs={"api_key": "key"},
      )
      model = factory.create_model(config)
    finally:
      if real_openai is not None:
        sys.modules["openai"] = real_openai
      else:
        sys.modules.pop("openai", None)
    self.assertIsInstance(model, minimax.MiniMaxLanguageModel)


if __name__ == "__main__":
  absltest.main()