"""
Generate full LLM prompts (system_prompt + user_prompt) from a short user description.
Uses a dedicated deployment when FREE_TEXT_PROMPT_DEPLOYMENT_NAME is set (e.g. gpt-5-nano for speed),
otherwise the main AI service deployment. Falls back to a template if LLM output is invalid.
"""

import json
import logging
from typing import Dict

from openai import AzureOpenAI
from config import config
from azure_auth import get_openai_client_with_auth

logger = logging.getLogger(__name__)


def _load_style_examples() -> str:
    """Load short structural examples from prompts.json for the meta-prompt."""
    try:
        with open("prompts.json", "r", encoding="utf-8") as f:
            all_prompts = json.load(f)
        examples = []
        for name in ("mitigation", "escalation"):
            if name in all_prompts:
                p = all_prompts[name]
                sys_preview = (p.get("system_prompt") or "")[:200]
                user_preview = (p.get("user_prompt") or "")[:200]
                examples.append(
                    f"Example '{name}': system_prompt (first 200 chars): {sys_preview}... "
                    f"user_prompt (first 200 chars): {user_preview}..."
                )
        return "\n".join(examples) if examples else "No examples available."
    except Exception as exc:
        logger.debug("Could not load prompt style examples: %s", exc)
        return "No examples available."


def _fallback_prompts(user_description: str) -> Dict[str, str]:
    """Return a simple template when LLM output cannot be parsed."""
    return {
        "system_prompt": "You are an expert incident analyst. Follow the user's instructions precisely. Use plain text only, no markdown.",
        "user_prompt": (
            f"User request: {user_description}\n\n"
            "Analyze the provided incident data and produce output that satisfies the above request. "
            "Use plain text, no markdown."
        ),
    }


def _parse_llm_response(text: str) -> Dict[str, str] | None:
    """Parse JSON from LLM response; strip markdown code fences if present. Return None on failure."""
    if not text or not text.strip():
        return None
    s = text.strip()
    # Remove optional markdown code fence
    if s.startswith("```"):
        lines = s.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "system_prompt" in obj and "user_prompt" in obj:
            return {
                "system_prompt": str(obj["system_prompt"]).strip(),
                "user_prompt": str(obj["user_prompt"]).strip(),
            }
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def generate_prompts_from_free_text(user_description: str) -> Dict[str, str]:
    """
    Turn a short user description into a full { system_prompt, user_prompt } dict
    compatible with the incident processor. Uses one LLM call; falls back to
    a simple template if the response is not valid JSON.
    """
    user_description = (user_description or "").strip()
    if not user_description:
        user_description = "Summarize this incident clearly."

    style_examples = _load_style_examples()

    meta_system = (
        "You are a prompt engineer. Your task is to produce a single JSON object with exactly two keys: "
        "'system_prompt' and 'user_prompt'. These prompts will be used to analyze support incident data "
        "(conversation and summary). The analyst model will receive the system_prompt as its role and the "
        "user_prompt plus the incident content as the user message. Output must be plain text, no markdown, "
        "consistent with support incident summarization. Match the style and structure of the examples below."
    )

    meta_user = (
        f"Generate a system_prompt and user_prompt that satisfy this user request:\n\n"
        f"\"{user_description}\"\n\n"
        f"Constraints: (1) Output a single JSON object with keys system_prompt and user_prompt. "
        f"(2) system_prompt defines the analyst role and high-level instructions. "
        f"(3) user_prompt must instruct the analyst to analyze the provided incident data and produce "
        f"output that fulfills the user request; use plain text, no markdown. "
        f"(4) Keep both prompts clear and actionable.\n\n"
        f"Style reference (excerpts from existing prompts):\n{style_examples}\n\n"
        f"Output only the JSON object, no other text."
    )

    try:
        client, _ = get_openai_client_with_auth(config)
        # Use a smaller/faster model for this easy task when configured (e.g. gpt-5-nano)
        deployment = getattr(config, 'free_text_prompt_deployment_name', None) or config.ai_service_deployment_name
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": meta_system},
                {"role": "user", "content": meta_user},
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        content = (response.choices[0].message.content or "").strip()
        parsed = _parse_llm_response(content)
        if parsed:
            return parsed
    except Exception as exc:
        logger.warning("Free-text prompt generation failed; using fallback template: %s", exc)

    return _fallback_prompts(user_description)
