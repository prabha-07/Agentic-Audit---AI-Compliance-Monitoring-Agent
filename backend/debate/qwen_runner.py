"""QwenRunner — unified wrapper around a Qwen (or compatible) LLM for the
compliance debate and reporter.

Two execution modes are supported, selected at runtime:

1. **Remote (OpenAI-compatible API)** — preferred for small-RAM deploys such as
   Render's free tier. Activated whenever ``REMOTE_LLM_BASE_URL`` *or*
   ``GROQ_API_KEY`` is set in the environment. Any OpenAI-compatible endpoint
   works (Groq, Together, OpenAI, Fireworks, vLLM, Ollama, ...).

   Environment variables (with convenience aliases for Groq):

   - ``REMOTE_LLM_BASE_URL``  (alias: ``GROQ_BASE_URL``)
         Base URL, e.g. ``https://api.groq.com/openai/v1``. If only
         ``GROQ_API_KEY`` is set we assume Groq and default the URL.
   - ``REMOTE_LLM_API_KEY``   (alias: ``GROQ_API_KEY``)
         API key for the provider.
   - ``REMOTE_LLM_MODEL``     (alias: ``GROQ_MODEL``)
         Model slug, e.g. ``qwen/qwen3-32b``. Defaults to ``qwen/qwen3-32b``
         which is the current Qwen model hosted on Groq.

2. **Local Transformers fallback** — used when no remote config is present and
   the machine has enough RAM/VRAM. Reads ``QWEN_MODEL_ID`` (default
   ``Qwen/Qwen3-8B``; set to ``Qwen/Qwen2.5-0.5B-Instruct`` for CPU smoke
   tests).

Both modes expose the same public surface:

    qwen.generate(prompt: str, thinking: bool = True, max_new_tokens: int = 1024) -> dict

Returning ``{"thinking_trace", "response", "full_output"}``.
"""

from __future__ import annotations

import os
from typing import Any

import backend.hf_setup  # noqa: F401 — load `.env` + HF auth before Hub access
from backend.hf_setup import hub_auth_token


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _remote_config() -> dict[str, str] | None:
    """Return ``{base_url, api_key, model}`` if remote mode is configured, else None."""
    base_url = (
        os.environ.get("REMOTE_LLM_BASE_URL")
        or os.environ.get("GROQ_BASE_URL")
        or ""
    ).strip()
    api_key = (
        os.environ.get("REMOTE_LLM_API_KEY")
        or os.environ.get("GROQ_API_KEY")
        or ""
    ).strip()
    model = (
        os.environ.get("REMOTE_LLM_MODEL")
        or os.environ.get("GROQ_MODEL")
        or "qwen/qwen3-32b"
    ).strip()

    # If a GROQ_API_KEY is supplied without an explicit base URL, assume Groq.
    if api_key and not base_url and os.environ.get("GROQ_API_KEY"):
        base_url = "https://api.groq.com/openai/v1"

    if not (base_url and api_key):
        return None
    return {"base_url": base_url, "api_key": api_key, "model": model}


def _split_thinking(full_output: str) -> tuple[str, str]:
    """Split ``<think>...</think>`` from the trailing response."""
    if "<think>" in full_output and "</think>" in full_output:
        thinking_trace = full_output.split("<think>", 1)[1].split("</think>", 1)[0].strip()
        response = full_output.split("</think>", 1)[-1].strip()
        return thinking_trace, response
    return "", full_output.strip()


# --------------------------------------------------------------------------- #
# QwenRunner
# --------------------------------------------------------------------------- #


class QwenRunner:
    """Singleton LLM runner used by the debate and reporter agents."""

    LOCAL_MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3-8B")

    def __init__(self) -> None:
        self._remote_cfg = _remote_config()
        self._remote_client: Any | None = None
        self.tokenizer = None
        self.model = None

        if self._remote_cfg is not None:
            # Remote mode — no local weights loaded. Defer client construction
            # to first use so import stays cheap on serverless cold starts.
            self.mode = "remote"
            self.model_id = self._remote_cfg["model"]
        else:
            self.mode = "local"
            self.model_id = self.LOCAL_MODEL_ID
            self._load_local()

    # --------------- local backend --------------- #

    def _load_local(self) -> None:
        """Eagerly load the local HF model. Only invoked in local mode."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _tok = hub_auth_token()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=_tok)

        try:
            if torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_mem
                if vram < 16 * 1024**3:
                    from transformers import BitsAndBytesConfig

                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        token=_tok,
                        quantization_config=bnb_config,
                        device_map="auto",
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        token=_tok,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                    )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    token=_tok,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                token=_tok,
                torch_dtype=torch.float32,
                device_map="cpu",
            )
        self.model.eval()

    def _generate_local(self, prompt: str, thinking: bool, max_new_tokens: int) -> str:
        import torch  # local import keeps remote-only deploys torch-free

        system_msg = (
            "You are an expert compliance auditor. Think step by step through the legal "
            "requirements before answering. Show your reasoning in <think>...</think> tags."
            if thinking
            else "You are an expert compliance auditor. Answer concisely and directly."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )

        return self.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

    # --------------- remote backend --------------- #

    def _get_remote_client(self) -> Any:
        if self._remote_client is not None:
            return self._remote_client
        from openai import OpenAI  # imported lazily so it's not required locally

        assert self._remote_cfg is not None
        self._remote_client = OpenAI(
            base_url=self._remote_cfg["base_url"],
            api_key=self._remote_cfg["api_key"],
            timeout=float(os.environ.get("REMOTE_LLM_TIMEOUT_SEC", "120")),
        )
        return self._remote_client

    def _generate_remote(self, prompt: str, thinking: bool, max_new_tokens: int) -> str:
        client = self._get_remote_client()
        system_msg = (
            "You are an expert compliance auditor. Think step by step through the legal "
            "requirements before answering. Show your reasoning in <think>...</think> tags."
            if thinking
            else "You are an expert compliance auditor. Answer concisely and directly."
        )
        # Providers diverge on max-token parameter names; pass both safely.
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": max_new_tokens,
        }
        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        content = (choice.message.content or "").strip()

        # Some providers (Groq, DeepSeek) expose reasoning as a separate field.
        reasoning = getattr(choice.message, "reasoning", None) or getattr(
            choice.message, "reasoning_content", None
        )
        if thinking and reasoning and "<think>" not in content:
            content = f"<think>{reasoning}</think>\n{content}"
        return content

    # --------------- public API --------------- #

    def generate(
        self, prompt: str, thinking: bool = True, max_new_tokens: int = 1024
    ) -> dict:
        """Run a single inference pass.

        Parameters
        ----------
        prompt : str
            The user-facing prompt (system message is added automatically).
        thinking : bool
            When True, ask the model for ``<think>...</think>`` reasoning tags;
            the response is then split accordingly. Remote providers that
            expose reasoning via a dedicated field are normalized into the
            same tag layout.
        max_new_tokens : int
            Generation budget.

        Returns
        -------
        dict with keys ``thinking_trace``, ``response``, ``full_output``.
        """
        if self.mode == "remote":
            full_output = self._generate_remote(prompt, thinking, max_new_tokens)
        else:
            full_output = self._generate_local(prompt, thinking, max_new_tokens)

        thinking_trace, response = _split_thinking(full_output)
        return {
            "thinking_trace": thinking_trace,
            "response": response,
            "full_output": full_output,
        }


# Module-level singleton. In remote mode this is cheap (no weights loaded).
qwen = QwenRunner()
