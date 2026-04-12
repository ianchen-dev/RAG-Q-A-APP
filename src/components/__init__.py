"""Components package for RAG ChatBot.

This package contains reusable components for the application,
such as prompt builders, templates, and other utilities.
"""

from src.components.prompt import create_chat_prompts

__all__ = ["create_chat_prompts"]
