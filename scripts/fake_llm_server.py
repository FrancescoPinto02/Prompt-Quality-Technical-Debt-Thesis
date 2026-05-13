import argparse
import json
import re
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List


TASK_CATEGORIES = {
    "CODE_GENERATION",
    "CODE_MODIFICATION",
    "BUG_FIXING",
    "REFACTORING",
    "TEST_GENERATION",
    "EXPLANATION",
    "CONFIGURATION",
    "DATA_QUERY",
    "OTHER",
    "AMBIGUOUS",
}


def extract_json_block(text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object found in a prompt.
    Used to parse the payload inserted between <<< and >>>.
    """
    match = re.search(r"<<<\s*(\{.*?\})\s*>>>", text, flags=re.DOTALL)
    if not match:
        return {}

    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


def fake_code_nl_response(prompt: str) -> Dict[str, Any]:
    """
    Produces a fake response for CodeNLSeparation.txt.

    Expected input payload:
    {
      "lines": [
        {"line_number": 1, "text": "..."},
        ...
      ]
    }

    Output:
    {
      "lines": [
        {"line_number": 1, "label": "NATURAL_LANGUAGE" | "CODE" | "EMPTY"}
      ]
    }
    """
    payload = extract_json_block(prompt)
    lines = payload.get("lines", [])

    output_lines = []

    for item in lines:
        line_number = int(item.get("line_number", 0))
        text = str(item.get("text", ""))
        stripped = text.strip()

        if not stripped:
            label = "EMPTY"
        elif looks_like_code(stripped):
            label = "CODE"
        else:
            label = "NATURAL_LANGUAGE"

        output_lines.append(
            {
                "line_number": line_number,
                "label": label,
            }
        )

    return {"lines": output_lines}


def looks_like_code(text: str) -> bool:
    """
    Lightweight heuristic used only for tests.
    It is intentionally simple, not meant to be scientifically accurate.
    """
    lower = text.lower()

    code_markers = [
        "def ",
        "class ",
        "function ",
        "const ",
        "let ",
        "var ",
        "return ",
        "import ",
        "from ",
        "public ",
        "private ",
        "protected ",
        "console.log",
        "print(",
        "if (",
        "for (",
        "while (",
        "{",
        "}",
        "</",
        "<div",
        "<span",
        "select ",
        "insert ",
        "update ",
        "delete ",
        "create table",
        "dockerfile",
        "version:",
        "services:",
        "npm ",
        "pip ",
        "git ",
        "traceback",
        "error:",
        "exception",
    ]

    if text.startswith(("```", "$ ", "#!/")):
        return True

    if any(marker in lower for marker in code_markers):
        return True

    # Common assignment/code-like pattern.
    if re.search(r"\w+\s*=\s*.+", text) and not text.endswith("?"):
        return True

    # File path / extension-like.
    if re.search(r"\b[\w\-./]+\.(py|js|ts|java|cpp|c|cs|go|rs|php|html|css|json|yaml|yml|xml|sql)\b", lower):
        return True

    return False


def fake_task_lang_response(prompt: str) -> Dict[str, Any]:
    """
    Produces a fake response for TaskLangFromSeparatedPrompt.txt
    or LangAndTaskClassification.txt.

    Expected input payload in the end-to-end script:
    {
      "natural_language_text": "...",
      "metadata": {
        "original_prompt_contained_code": true,
        "code_line_count": 10
      }
    }

    Output follows the expected task/lang schema.
    """
    payload = extract_json_block(prompt)

    if "natural_language_text" in payload:
        text = str(payload.get("natural_language_text", ""))
        contains_code = bool(
            payload.get("metadata", {}).get("original_prompt_contained_code", False)
        )
    else:
        # Fallback for older prompts that pass a full conversation payload as text.
        text = prompt
        contains_code = "code" in prompt.lower()

    text_lower = text.lower().strip()

    task_category = classify_task_heuristic(text_lower, contains_code)
    is_code_generation = task_category in {
        "CODE_GENERATION",
        "CODE_MODIFICATION",
        "BUG_FIXING",
        "REFACTORING",
        "TEST_GENERATION",
        "CONFIGURATION",
        "DATA_QUERY",
    }

    detected_language = detect_language_heuristic(text)

    return {
        "task_category": task_category,
        "task_confidence": "HIGH" if task_category != "AMBIGUOUS" else "LOW",
        "is_code_generation": is_code_generation,
        "code_generation_confidence": "HIGH" if task_category != "AMBIGUOUS" else "LOW",
        "detected_language": detected_language,
        "language_confidence": "HIGH" if detected_language != "UNKNOWN" else "LOW",
        "short_reason": (
            f"Fake classifier assigned {task_category} and detected {detected_language}."
        ),
    }


def classify_task_heuristic(text_lower: str, contains_code: bool) -> str:
    """
    Very simple task classifier for tests.
    The goal is schema compatibility, not annotation quality.
    """
    if not text_lower:
        return "AMBIGUOUS"

    if any(greeting == text_lower for greeting in ["hi", "hello", "thanks", "thank you", "ciao"]):
        return "OTHER"

    if any(word in text_lower for word in ["test", "unit test", "pytest", "jest", "mock"]):
        return "TEST_GENERATION"

    if any(word in text_lower for word in ["refactor", "cleaner", "maintainable", "optimize", "improve this code"]):
        return "REFACTORING"

    if any(word in text_lower for word in ["fix", "bug", "error", "exception", "doesn't work", "not working", "wrong result"]):
        return "BUG_FIXING"

    if any(word in text_lower for word in ["modify", "update", "convert", "extend", "complete", "add support"]):
        return "CODE_MODIFICATION"

    if any(word in text_lower for word in ["docker", "kubernetes", "ci", "github actions", "yaml", "deployment", "environment"]):
        return "CONFIGURATION"

    if any(word in text_lower for word in ["sql", "query", "pandas", "dataframe", "database", "csv", "data processing"]):
        return "DATA_QUERY"

    if any(word in text_lower for word in ["explain", "what is", "how does", "why does", "tutorial", "example of"]):
        return "EXPLANATION"

    if any(word in text_lower for word in ["write", "create", "generate", "implement", "build", "script", "function", "class", "component", "api"]):
        return "CODE_GENERATION"

    if contains_code and any(word in text_lower for word in ["this", "help", "please"]):
        return "BUG_FIXING"

    return "OTHER"


def detect_language_heuristic(text: str) -> str:
    """
    Tiny heuristic language detector for tests.
    It only aims to produce valid ISO-like codes.
    """
    stripped = text.strip()

    if not stripped:
        return "UNKNOWN"

    # Chinese characters
    if re.search(r"[\u4e00-\u9fff]", stripped):
        return "ZH"

    # Japanese Hiragana/Katakana
    if re.search(r"[\u3040-\u30ff]", stripped):
        return "JA"

    lower = stripped.lower()

    italian_words = ["ciao", "spiegami", "scrivi", "codice", "funzione", "errore", "perché"]
    spanish_words = ["hola", "explica", "escribe", "código", "función", "error", "por qué"]
    french_words = ["bonjour", "explique", "écris", "code", "fonction", "erreur", "pourquoi"]
    german_words = ["hallo", "erkläre", "schreibe", "funktion", "fehler", "warum"]

    if any(word in lower for word in italian_words):
        return "IT"

    if any(word in lower for word in spanish_words):
        return "ES"

    if any(word in lower for word in french_words):
        return "FR"

    if any(word in lower for word in german_words):
        return "DE"

    return "EN"


def choose_fake_response(prompt: str) -> Dict[str, Any]:
    """
    Chooses which fake response to return based on the prompt content.
    """
    if '"line_number"' in prompt and '"text"' in prompt:
        return fake_code_nl_response(prompt)

    if "task_category" in prompt and "detected_language" in prompt:
        return fake_task_lang_response(prompt)

    # Default fallback: valid task/lang response.
    return fake_task_lang_response(prompt)


class FakeLLMHandler(BaseHTTPRequestHandler):
    """
    Minimal OpenAI-compatible HTTP handler.
    Supports:
    - GET /v1/models
    - POST /v1/chat/completions
    """

    server_version = "FakeLLM/0.1"

    def _send_json(self, obj: Dict[str, Any], status_code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")

        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/v1/models":
            self._send_json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "fake-llm",
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "local-test",
                        }
                    ],
                }
            )
            return

        self._send_json({"error": f"Unknown endpoint: {self.path}"}, status_code=404)

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send_json({"error": f"Unknown endpoint: {self.path}"}, status_code=404)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)

        try:
            request_payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON request body."}, status_code=400)
            return

        messages = request_payload.get("messages", [])
        model = request_payload.get("model", "fake-llm")

        prompt = ""
        if messages:
            prompt = str(messages[-1].get("content", ""))

        fake_obj = choose_fake_response(prompt)
        fake_content = json.dumps(fake_obj, ensure_ascii=False, indent=2)

        response = {
            "id": f"chatcmpl-fake-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": fake_content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": max(1, len(prompt.split())),
                "completion_tokens": max(1, len(fake_content.split())),
                "total_tokens": max(1, len(prompt.split()) + len(fake_content.split())),
            },
        }

        self._send_json(response)

    def log_message(self, format: str, *args: Any) -> None:
        """
        Keeps server logs compact.
        """
        print(f"[FakeLLM] {self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a fake OpenAI-compatible LLM server for pipeline tests."
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the fake server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the fake server.",
    )

    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), FakeLLMHandler)

    print(f"Fake LLM server running at http://{args.host}:{args.port}/v1")
    print("Available endpoints:")
    print(f"  GET  http://{args.host}:{args.port}/v1/models")
    print(f"  POST http://{args.host}:{args.port}/v1/chat/completions")
    print("Press CTRL+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping fake LLM server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()