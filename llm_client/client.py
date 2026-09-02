"""Unified access layer for the hosted LLMs used by the travel assistant."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# Providers that speak the plain OpenAI chat-completions dialect.
STANDARD_PROVIDERS = frozenset({"openai", "anthropic", "google"})
# Providers routed through the Qwen-specific code path.
QWEN_PROVIDERS = frozenset({"qwen"})
# Providers that can stream a separate reasoning channel.
THINKING_PROVIDERS = frozenset({"deepseek", "glm"})

# Width of the "=" rules printed around streamed reasoning output.
SEPARATOR_WIDTH = 20


class LLMManager:
    """Single entry point for calling every configured LLM model."""


    def __init__(self):
        self.models = {
            "gpt-5": {
                "api_key": "",
                "base_url": "https://api.openai.com/v1",
                "model_name": "gpt-5-2025-08-07",
                "provider": "openai"
            },
            "claude-opus": {
                "api_key": "",
                "base_url": "https://api.anthropic.com/v1/",
                "model_name": "claude-opus-4-1-20250805",
                "provider": "anthropic"
            },
            "gemini-pro": {
                "api_key": "",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                "model_name": "gemini-2.5-pro",
                "provider": "google",
                "extra_params": {"reasoning_effort": "low"}
            },
            "gemini-flash": {
                "api_key": "",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                "model_name": "gemini-2.5-flash",
                "provider": "google",
                "extra_params": {"reasoning_effort": "low"}
            },
            "qwen-plus": {
                "api_key": "",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model_name": "qwen-plus",
                "provider": "qwen"
            },
            "qwen-32b": {
                "api_key": "",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model_name": "qwen3-32b",
                "provider": "qwen"
            },
            "qwen-14b": {
                "api_key": "",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model_name": "qwen3-14b",
                "provider": "qwen"
            },
            "deepseek-v3": {
                "api_key": "",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model_name": "deepseek-v3.1",
                "provider": "deepseek"
            },
            "deepseek-r1": {
                "api_key": "",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model_name": "deepseek-r1",
                "provider": "deepseek",
                "enable_thinking": True
            },
            "glm-4.5": {
                "api_key": "",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model_name": "glm-4.5",
                "provider": "glm",
                "enable_thinking": True
            }
        }

    def get_available_models(self) -> List[str]:
        """Return the names of every configured model."""
        return list(self.models.keys())

    def call_model(self, model_name: str, prompt: Optional[str] = None, system_prompt: Optional[str] = None, messages: Optional[List[Dict]] = None, tools: Optional[List[Dict]] = None, tool_call_handler: Optional[Callable[..., Any]] = None, stream: bool = False, show_thinking: bool = True) -> str:
        """Run inference with the named model.

        Args:
            model_name: Name of the configured model to call.
            prompt: User prompt text (alternative to ``messages``).
            system_prompt: Optional system prompt.
            messages: Message list (alternative to ``prompt``; takes precedence).
            tools: Optional tool definitions.
            tool_call_handler: Optional callback handling tool calls.
            stream: Whether to stream the response.
            show_thinking: Whether to print the reasoning trace (thinking models only).

        Returns:
            The model's response content.

        Raises:
            ValueError: If ``model_name`` is not a configured model.
        """
        if model_name not in self.models:
            raise ValueError(f"模型 '{model_name}' 不存在。可用模型: {self.get_available_models()}")
        
        model_config = self.models[model_name]

        # Build the message list.
        if messages is not None:
            # Use the caller-supplied message list as-is.
            final_messages = messages
        else:
            # Fall back to the simple prompt/system-prompt form.
            final_messages = []
            if system_prompt:
                final_messages.append({"role": "system", "content": system_prompt})
            if prompt:
                final_messages.append({"role": "user", "content": prompt})
        
        # Create the client.
        client = OpenAI(
            api_key=model_config["api_key"],
            base_url=model_config["base_url"]
        )

        # Pick the call path that matches the provider.
        if model_config["provider"] in STANDARD_PROVIDERS:
            return self._call_standard_model(client, model_config, final_messages, tools, tool_call_handler, stream)
        elif model_config["provider"] in QWEN_PROVIDERS:
            return self._call_qwen_model(client, model_config, final_messages, tools, tool_call_handler, stream)
        elif model_config["provider"] in THINKING_PROVIDERS and model_config.get("enable_thinking", False):
            return self._call_thinking_model(client, model_config, final_messages, stream, show_thinking)
        else:
            return self._call_standard_model(client, model_config, final_messages, tools, tool_call_handler, stream)

    def _call_standard_model(self, client: OpenAI, model_config: Dict, messages: List[Dict], tools: Optional[List[Dict]], tool_call_handler: Optional[Callable[..., Any]], stream: bool) -> str:
        """Call a standard OpenAI-compatible model (GPT, Claude, Gemini, ...)."""
        try:
            extra_params = model_config.get("extra_params", {})
            
            if stream:
                response_stream = client.chat.completions.create(
                    model=model_config["model_name"],
                    messages=messages,
                    stream=True,
                    **extra_params
                )
                
                content_parts = []
                for chunk in response_stream:
                    if chunk.choices:
                        content = chunk.choices[0].delta.content or ""
                        print(content, end="", flush=True)
                        content_parts.append(content)
                
                print()
                return "".join(content_parts)
            else:
                response = client.chat.completions.create(
                    model=model_config["model_name"],
                    messages=messages,
                    **extra_params
                )
                return response.choices[0].message.content

        except Exception as e:
            logger.exception("Standard model call failed: model=%s", model_config.get("model_name"))
            return f"调用模型时出错: {e}"

    def _call_qwen_model(self, client: OpenAI, model_config: Dict, messages: List[Dict], tools: Optional[List[Dict]], tool_call_handler: Optional[Callable[..., Any]], stream: bool) -> str:
        """Call a Qwen model."""
        try:
            if stream:
                response_stream = client.chat.completions.create(
                    model=model_config["model_name"],
                    messages=messages,
                    stream=True
                )
                
                content_parts = []
                print("AI: ", end="", flush=True)
                
                for chunk in response_stream:
                    if chunk.choices:
                        content = chunk.choices[0].delta.content or ""
                        print(content, end="", flush=True)
                        content_parts.append(content)
                
                print()
                return "".join(content_parts)
            else:
                response = client.chat.completions.create(
                    model=model_config["model_name"],
                    messages=messages
                )
                return response.choices[0].message.content

        except Exception as e:
            logger.exception("Qwen model call failed: model=%s", model_config.get("model_name"))
            return f"调用Qwen模型时出错: {e}"

    def _call_thinking_model(self, client: OpenAI, model_config: Dict, messages: List[Dict], stream: bool, show_thinking: bool) -> str:
        """Call a model that streams a reasoning trace (DeepSeek, GLM, ...)."""
        try:
            response_stream = client.chat.completions.create(
                model=model_config["model_name"],
                messages=messages,
                extra_body={"enable_thinking": True},
                stream=True,
                stream_options={"include_usage": True}
            )

            reasoning_content = ""
            answer_content = ""
            is_answering = False
            
            if show_thinking:
                print("\n" + "=" * SEPARATOR_WIDTH + "思考过程" + "=" * SEPARATOR_WIDTH + "\n")

            for chunk in response_stream:
                if not chunk.choices:
                    if show_thinking:
                        print("\n" + "=" * SEPARATOR_WIDTH + "Usage" + "=" * SEPARATOR_WIDTH)
                        print(chunk.usage)
                    continue
                    
                delta = chunk.choices[0].delta

                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    if show_thinking and not is_answering:
                        print(delta.reasoning_content, end="", flush=True)
                    reasoning_content += delta.reasoning_content

                if hasattr(delta, "content") and delta.content:
                    if show_thinking and not is_answering:
                        print("\n" + "=" * SEPARATOR_WIDTH + "完整回复" + "=" * SEPARATOR_WIDTH + "\n")
                        is_answering = True
                    if show_thinking:
                        print(delta.content, end="", flush=True)
                    answer_content += delta.content

            if show_thinking:
                print()
            
            return f"[思考]\n{reasoning_content}\n\n[结果]\n{answer_content}" if reasoning_content else answer_content

        except Exception as e:
            logger.exception("Thinking model call failed: model=%s", model_config.get("model_name"))
            return f"调用思考模型时出错: {e}"


# Global instance shared by every caller.
llm_manager = LLMManager()


def call_llm(model_name: str, prompt: Optional[str] = None, system_prompt: Optional[str] = None, messages: Optional[List[Dict]] = None, stream: bool = False, show_thinking: bool = True) -> str:
    """Call the named LLM model.

    Args:
        model_name: Name of the configured model to call.
        prompt: User prompt text (alternative to ``messages``).
        system_prompt: Optional system prompt.
        messages: Message list (alternative to ``prompt``; takes precedence).
        stream: Whether to stream the response.
        show_thinking: Whether to print the reasoning trace.

    Returns:
        The model's response.
    """
    return llm_manager.call_model(
        model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        messages=messages,
        stream=stream,
        show_thinking=show_thinking,
    )


def get_model_list() -> List[str]:
    """Return the list of every available model name."""
    return llm_manager.get_available_models()


# Example usage.
if __name__ == "__main__":
    # List the available models.
    print("可用模型:", get_model_list())

    # Example call.
    response = call_llm("gpt-5", "你好，请介绍一下自己")
    print(response)
