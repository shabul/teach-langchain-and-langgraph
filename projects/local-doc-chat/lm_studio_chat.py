"""
Custom LangChain chat model wrapper for LM Studio's REST API.

LM Studio exposes a non-OpenAI-compatible endpoint:
  POST http://localhost:1234/api/v1/chat
  { "model": "...", "system_prompt": "...", "input": "..." }

This wrapper converts LangChain's list[BaseMessage] into that format
so ChatLMStudio works as a drop-in for ChatOpenAI across all snippets.
"""
from __future__ import annotations

from typing import Any, Iterator, List, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field


class ChatLMStudio(BaseChatModel):
    """LangChain chat model backed by LM Studio's /api/v1/chat endpoint."""

    model: str = "openai/gpt-oss-20b"
    base_url: str = "http://localhost:1234"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: int = 120

    @property
    def _llm_type(self) -> str:
        return "lm-studio"

    def _convert_messages(self, messages: list[BaseMessage]) -> tuple[str, str]:
        """
        Split a LangChain message list into (system_prompt, input).

        LM Studio expects a single system_prompt string and a single input string.
        We handle multi-turn history by formatting it as a conversation block
        that gets appended to the user input.
        """
        system_parts: list[str] = []
        history: list[BaseMessage] = []

        for m in messages:
            if isinstance(m, SystemMessage):
                system_parts.append(m.content)
            else:
                history.append(m)

        system_prompt = "\n\n".join(system_parts)

        # Format conversation history + final user turn into a single input string
        if len(history) == 1:
            user_input = history[0].content
        else:
            lines: list[str] = []
            for m in history[:-1]:
                role = "User" if isinstance(m, HumanMessage) else "Assistant"
                lines.append(f"{role}: {m.content}")
            lines.append(f"User: {history[-1].content}")
            user_input = "\n".join(lines)

        return system_prompt, user_input

    def _parse_response(self, data: dict) -> str:
        """Extract the assistant text from LM Studio's response, with fallbacks."""
        # Standard OpenAI-style choices array
        if "choices" in data and data["choices"]:
            choice = data["choices"][0]
            if "message" in choice:
                return choice["message"].get("content", "")
            if "text" in choice:
                return choice["text"]
        # LM Studio REST-specific flat output key
        if "output" in data:
            return data["output"]
        # Last resort
        return str(data)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        system_prompt, user_input = self._convert_messages(messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "system_prompt": system_prompt,
            "input": user_input,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if stop:
            payload["stop"] = stop

        resp = requests.post(
            f"{self.base_url}/api/v1/chat",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        content = self._parse_response(resp.json())
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "base_url": self.base_url, "temperature": self.temperature}
