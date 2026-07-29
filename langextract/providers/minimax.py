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

"""MiniMax provider for LangExtract.

MiniMax exposes an OpenAI-compatible chat completions API, so this provider
is a thin subclass of :class:`OpenAILanguageModel` that pre-populates the
MiniMax default ``base_url`` and ``model_id``. All inference semantics
(prompt formatting, fence handling, ``response_format`` routing, parallel
batching, ``merge_kwargs``) are inherited unchanged from the OpenAI
provider, which keeps LangExtract's structured-output contract intact.

Usage::

    import langextract as lx
    from langextract.providers.minimax import MiniMaxLanguageModel

    model = MiniMaxLanguageModel(
        model_id="MiniMax-M2.5",
        api_key="<MINIMAX_API_KEY>",
    )
    result = lx.extract(
        text_or_documents=text,
        prompt_description=instructions,
        model=model,
    )

Or via the factory by selecting the provider explicitly::

    from langextract import factory

    config = factory.ModelConfig(
        model_id="MiniMax-M2.5",
        provider="MiniMaxLanguageModel",
        provider_kwargs={"api_key": "<MINIMAX_API_KEY>"},
    )
    model = factory.create_model(config)
"""

from __future__ import annotations

from langextract.providers import openai
from langextract.providers import patterns
from langextract.providers import router

# MiniMax's public OpenAI-compatible endpoint. Callers may override ``base_url``
# (e.g. for staging or self-hosted gateways) via ``provider_kwargs``.
_DEFAULT_BASE_URL = "https://api.minimax.io/v1"


@router.register(
    *patterns.MINIMAX_PATTERNS,
    priority=patterns.MINIMAX_PRIORITY,
)
class MiniMaxLanguageModel(openai.OpenAILanguageModel):
  """Language model inference using MiniMax's OpenAI-compatible API.

  Inherits all inference behaviour from :class:`OpenAILanguageModel`; only
  the default ``model_id`` and ``base_url`` are MiniMax-specific.
  """

  DEFAULT_MODEL_ID = "MiniMax-M2.5"
  DEFAULT_BASE_URL = _DEFAULT_BASE_URL

  def __init__(  # pylint: disable=too-many-arguments
      self,
      model_id: str = DEFAULT_MODEL_ID,
      api_key: str | None = None,
      base_url: str | None = DEFAULT_BASE_URL,
      organization: str | None = None,
      format_type=None,
      temperature: float | None = None,
      max_workers: int = 10,
      **kwargs,
  ) -> None:
    """Initialize the MiniMax language model.

    Args:
      model_id: MiniMax model identifier (e.g. ``"MiniMax-M2.5"``).
      api_key: MiniMax API key. Required, matching the OpenAI provider.
      base_url: MiniMax OpenAI-compatible endpoint. Defaults to the
        public ``https://api.minimax.io/v1`` URL.
      organization: Unused by MiniMax; accepted for API parity.
      format_type: Output format (``FormatType.JSON`` or
        ``FormatType.YAML``). Defaults to JSON to match OpenAI.
      temperature: Sampling temperature.
      max_workers: Maximum parallel chat-completion requests.
      **kwargs: Forwarded to ``OpenAILanguageModel`` for parity.
    """
    if format_type is None:
      from langextract.core import data  # pylint: disable=import-outside-toplevel

      format_type = data.FormatType.JSON

    super().__init__(
        model_id=model_id,
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        format_type=format_type,
        temperature=temperature,
        max_workers=max_workers,
        **kwargs,
    )