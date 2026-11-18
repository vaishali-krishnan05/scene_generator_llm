import argparse
import json
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


LMSTUDIO_BASE_URL = "http://localhost:1234"
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma4:12b"
DEFAULT_LMSTUDIO_MODEL = "google/gemma-4-12b"

# Define the instructions given to the language model for story generation.
# The prompt also controls the required JSON structure and content safety rules.
SYSTEM_PROMPT = """You create illustrated story scripts as JSON.
SYSTEM_PROMPT = """You create illustrated story scripts as JSON.
First decide whether the user's input is a request for a child-friendly fictional
story or a story premise suitable for a comic.

Refuse requests that are not story requests, including standalone math problems
or equations, factual questions, requests for current information, and general
explanations. Examples to refuse include "What is 2+3?", "Who is president of
India?", and "Why does sunset look red?". Do not turn these questions into
stories unless the user explicitly asks for a child-friendly story about the
topic. Also refuse topics or requested content that are not suitable for
children, including sexual content, graphic violence, abuse, self-harm, hateful
content, or instructions that encourage dangerous or illegal activity. If the
request is ambiguous, refuse and ask for a clear, child-friendly story premise.

For a refusal, return only this JSON object, with no markdown or commentary:
{"refusal": "Please provide a clear, child-friendly story premise for a comic."}

For an allowed story request, return only one valid JSON object, with no
markdown or commentary, using this structure:
{
  "config": {
    "width": 768,
    "height": 1024,
    "columns": 2,
    "global_style": "A concise visual style for the story",
    "negative_prompt": "Visual elements to avoid"
  },
  "characters": {
    "Character Name": "A consistent visual description of the character"
  },
  "script": [
    {
      "focus": ["Character Name"],
      "scene": "A visual description of this moment",
      "dialogue": "Spoken dialogue, brief narration, or an empty string for a silent panel"
    }
  ]
}

Keep every character description, scene, and line of dialogue suitable for
children. Do not add sexual content, graphic violence, abuse, self-harm,
hateful content, or dangerous or illegal instructions.
Create a clear, coherent sequence of exactly {panels} panels that covers the
user's plot from beginning to end. Include every important event in order.
Keep character names consistent, describe recurring characters consistently,
and only use character names defined in "characters" in each "focus" list.
Every panel must have at least one focused character. Keep scene descriptions
visual and concise. Include a non-empty dialogue or brief narration unless the
panel is intentionally silent; silent panels must use an empty string. Return
all five config fields and all three panel fields.
"""


class StoryRequestRefusedError(ValueError):
    """Raised when the model refuses a non-story or child-unsuitable request."""


REFUSAL_MESSAGE = (
    "I can only generate child-friendly comic stories. Please enter a story premise, "
    "not a math problem, factual question, or general explanation."
)
_EXPLICIT_STORY_REQUEST = re.compile(
    r"\b(?:write|create|generate|make|tell)\b.{0,50}"
    r"\b(?:story|comic|tale|narrative|plot)\b",
    re.IGNORECASE | re.DOTALL,
)
_QUESTION_OR_FACTUAL_REQUEST = re.compile(
    r"^(?:what|who|whom|whose|when|where|why|how|is|are|am|can|could|would|"
    r"should|do|does|did|will|solve|calculate|compute|evaluate|explain|define|"
    r"summarize|summarise|tell me about|describe|answer)\b",
    re.IGNORECASE,
)


def _is_bare_math_problem(text):
    expression = re.sub(
        r"^(?:(?:what is)|calculate|compute|evaluate|solve(?: for)?|find)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    if not re.fullmatch(r"[a-z0-9\s()+\-*/^=.%×÷]+", expression, re.IGNORECASE):
        return False
    if not re.search(r"\d", expression) or not re.search(r"[+\-*/^=×÷]", expression):
        return False
    return all(len(token) == 1 for token in re.findall(r"[a-z]+", expression, re.IGNORECASE))


def validate_story_request(plot):
    """Reject clearly non-story prompts before calling a generation backend."""
    plot = plot.strip()
    if _EXPLICIT_STORY_REQUEST.search(plot):
        return
    if (
        plot.endswith("?")
        or _QUESTION_OR_FACTUAL_REQUEST.match(plot)
        or _is_bare_math_problem(plot)
    ):
        raise StoryRequestRefusedError(REFUSAL_MESSAGE)


def parse_response(content):
    content = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
    if fenced:
        content = fenced.group(1)

    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"Model response is not valid JSON: {error}") from error

    if isinstance(result, dict) and "refusal" in result:
        refusal = result["refusal"]
        if not isinstance(refusal, str) or not refusal.strip():
            raise ValueError('The model returned an invalid "refusal" response.')
        raise StoryRequestRefusedError(refusal.strip())

    validate_result(result)
    return result


def validate_result(result):
    if not isinstance(result, dict):
        raise ValueError("The JSON root must be an object.")

    config = result.get("config")
    if not isinstance(config, dict):
        raise ValueError('"config" must be an object.')
    for key in ("width", "height", "columns"):
        if not isinstance(config.get(key), int) or isinstance(config.get(key), bool) or config[key] <= 0:
            raise ValueError(f'"config.{key}" must be a positive integer.')
    for key in ("global_style", "negative_prompt"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f'"config.{key}" must be a non-empty string.')

    characters = result.get("characters")
    if not isinstance(characters, dict) or not characters:
        raise ValueError('"characters" must be a non-empty object.')
    for name, description in characters.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Character names must be non-empty strings.")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f'Description for character "{name}" must be a non-empty string.')

    script = result.get("script")
    if not isinstance(script, list) or not script:
        raise ValueError('"script" must be a non-empty array.')
    for index, panel in enumerate(script, start=1):
        if not isinstance(panel, dict):
            raise ValueError(f"Panel {index} must be an object.")
        focus = panel.get("focus")
        if not isinstance(focus, list) or not focus:
            raise ValueError(f"Panel {index} must have a non-empty focus array.")
        for name in focus:
            if name not in characters:
                raise ValueError(f'Panel {index} focuses on undefined character "{name}".')
        if not isinstance(panel.get("scene"), str) or not panel["scene"].strip():
            raise ValueError(f'Panel {index} "scene" must be a non-empty string.')
        if not isinstance(panel.get("dialogue"), str):
            raise ValueError(f'Panel {index} "dialogue" must be a string.')


class BackendUnavailableError(RuntimeError):
    pass


def build_messages(plot, panels):
    system_prompt = SYSTEM_PROMPT.replace("{panels}", str(panels))
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Classify this submission using the system rules:\n{plot}",
        },
    ]


def is_timeout_error(error):
    return isinstance(error, TimeoutError) or (
        isinstance(error, URLError) and isinstance(error.reason, TimeoutError)
    )


def timeout_message(backend, timeout):
    return (
        f"{backend} did not return a response within {timeout:g} seconds. "
        "Increase the wait with --timeout."
    )


def check_lmstudio_model(base_url, model, timeout):
    request = Request(f"{base_url.rstrip('/')}/v1/models", method="GET")
    try:
        with urlopen(request, timeout=min(timeout, 5)) as response:
            api_result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        return f"LM Studio returned HTTP {error.code}: {detail}"
    except URLError as error:
        return f"Could not connect to LM Studio at {base_url}: {error.reason}"
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return f"LM Studio returned an invalid models response: {error}"

    models = api_result.get("data") if isinstance(api_result, dict) else None
    if not isinstance(models, list):
        return 'LM Studio response is missing the "data" models list.'
    if not any(isinstance(item, dict) and item.get("id") == model for item in models):
        return f'Model "{model}" is not available in LM Studio.'
    return None


def generate_scene_lmstudio(plot, model, base_url, panels, timeout):
    payload = {
        "model": model,
        "messages": build_messages(plot, panels),
        "response_format": {"type": "text"},
        "stream": False,
        "temperature": 0.7,
    }
    request = Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            api_result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise BackendUnavailableError(
            f"LM Studio returned HTTP {error.code}: {detail}"
        ) from error
    except URLError as error:
        if is_timeout_error(error):
            raise RuntimeError(timeout_message("LM Studio", timeout)) from error
        raise BackendUnavailableError(
            f"Could not connect to LM Studio at {base_url}: {error.reason}"
        ) from error
    except TimeoutError as error:
        raise RuntimeError(timeout_message("LM Studio", timeout)) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"LM Studio returned an invalid API response: {error}") from error

    try:
        content = api_result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(
            "LM Studio response is missing choices[0].message.content."
        ) from error
    if not isinstance(content, str):
        raise RuntimeError("LM Studio response content must be a string.")

    return parse_response(content)


def generate_scene_ollama(plot, model, base_url, panels, timeout):
    payload = {
        "model": model,
        "messages": build_messages(plot, panels),
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.7},
    }
    request = Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            api_result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        if is_timeout_error(error):
            raise RuntimeError(timeout_message("Ollama", timeout)) from error
        raise RuntimeError(
            f"Could not connect to Ollama at {base_url}: {error.reason}. "
            "Make sure Ollama is running."
        ) from error
    except TimeoutError as error:
        raise RuntimeError(timeout_message("Ollama", timeout)) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Ollama returned an invalid API response: {error}") from error

    try:
        content = api_result["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("Ollama response is missing message.content.") from error
    if not isinstance(content, str):
        raise RuntimeError("Ollama response content must be a string.")

    return parse_response(content)


def generate_story(
    plot,
    model=DEFAULT_MODEL,
    lmstudio_model=DEFAULT_LMSTUDIO_MODEL,
    lmstudio_base_url=LMSTUDIO_BASE_URL,
    base_url=OLLAMA_BASE_URL,
    panels=8,
    timeout=1800,
):
    """Generate a validated story object using LM Studio or Ollama."""
    if not isinstance(plot, str) or not plot.strip():
        raise ValueError("The plot input is empty.")
    if panels < 1:
        raise ValueError("The number of panels must be at least 1.")
    if timeout <= 0:
        raise ValueError("The timeout must be greater than 0.")

    plot = plot.strip()
    validate_story_request(plot)
    unavailable_reason = check_lmstudio_model(
        lmstudio_base_url, lmstudio_model, timeout
    )
    if unavailable_reason:
        print(f"{unavailable_reason} Falling back to Ollama.", file=sys.stderr)
        return generate_scene_ollama(plot, model, base_url, panels, timeout)

    try:
        return generate_scene_lmstudio(
            plot,
            lmstudio_model,
            lmstudio_base_url,
            panels,
            timeout,
        )
    except BackendUnavailableError as error:
        print(f"{error}. Falling back to Ollama.", file=sys.stderr)
        return generate_scene_ollama(plot, model, base_url, panels, timeout)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a story JSON file using LM Studio, with Ollama fallback."
    )
    parser.add_argument(
        "plot", nargs="?", help="Plot to turn into a visual story script (inline)"
    )
    parser.add_argument(
        "-i", "--input-file", type=Path,
        help="UTF-8 text file containing the plot",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("story.json"),
        help="Output JSON file (default: story.json)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--lmstudio-model", default=DEFAULT_LMSTUDIO_MODEL,
        help=f"LM Studio model ID (default: {DEFAULT_LMSTUDIO_MODEL})",
    )
    parser.add_argument(
        "--lmstudio-base-url", default=LMSTUDIO_BASE_URL,
        help=f"LM Studio server URL (default: {LMSTUDIO_BASE_URL})",
    )
    parser.add_argument(
        "--base-url", default=OLLAMA_BASE_URL,
        help=f"Ollama server URL (default: {OLLAMA_BASE_URL})",
    )
    parser.add_argument(
        "--panels", type=int, default=8,
        help="Number of story panels to generate (default: 8)",
    )
    parser.add_argument(
        "--timeout", type=float, default=1800,
        help="Wait for each model response in seconds (default: 1800)",
    )
    args = parser.parse_args()

    if (args.plot is None) == (args.input_file is None):
        parser.error("provide either an inline plot or --input-file, but not both")
    if args.panels < 1:
        parser.error("--panels must be at least 1")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than 0")

    try:
        plot = args.input_file.read_text(encoding="utf-8") if args.input_file else args.plot
        result = generate_story(
            plot,
            model=args.model,
            lmstudio_model=args.lmstudio_model,
            lmstudio_base_url=args.lmstudio_base_url,
            base_url=args.base_url,
            panels=args.panels,
            timeout=args.timeout,
        )
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=4) + "\n",
            encoding="utf-8",
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Saved story JSON to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
