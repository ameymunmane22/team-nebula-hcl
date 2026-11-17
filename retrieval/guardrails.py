"""Rule-based guardrails layer for enterprise RAG.

Capabilities:
1. Sensitive Information Detection & Masking: Emails, phone numbers, Aadhaar, PAN, passwords, financial accounts.
2. Prompt Injection Detection: Jailbreaks, system prompt extraction, instruction overrides, and security bypasses.
3. Harmful Content Detection: Violence, self-harm, malware/cyberattack payloads, and illegal activities.
4. Modern LangChain LCEL Integration: RunnableLambda compatible with the '|' operator.
"""

import re
from typing import Any, Dict, List, Pattern, Union

from langchain_core.runnables import RunnableLambda


# 1. Predefined regex patterns for sensitive enterprise data
SENSITIVE_PATTERNS: Dict[str, Pattern[str]] = {
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "AADHAAR": re.compile(r"(?<![-\d])[2-9]\d{3}[\s\-]\d{4}[\s\-]\d{4}(?![-\d])"),
    "PAN": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "PASSWORD": re.compile(r"(?i)\b(?:password|passwd|pwd|secret_key|api_key)[\s:=]+([^\s,;]+)"),
    "BANK_ACCOUNT": re.compile(r"(?i)\b(?:account\s*(?:no|num|number)?|a/c)[\s:=]*([0-9]{9,18})\b"),
    "IFSC": re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
    "PHONE": re.compile(
        r"\b(?:\+?91[\-\s]?)?[6-9]\d{9}\b|\b(?:\+?1[\-\s]?)?\(?\d{3}\)?[\-\s]\d{3}[\-\s]\d{4}\b"
    ),
}

# 2. Predefined patterns and heuristics for prompt injection detection
PROMPT_INJECTION_PATTERNS: List[Dict[str, Any]] = [
    {
        "category": "instruction_override",
        "pattern": re.compile(
            r"(?i)\b(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:previous|prior|above|system|existing)\s+(?:instructions?|rules?|prompts?|constraints?)\b"
        ),
    },
    {
        "category": "jailbreak_persona",
        "pattern": re.compile(
            r"(?i)\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+(?:an?\s+|in\s+)?(?:unfiltered|unrestricted|jailbroken|developer\s+mode|dan|evil|chaos)\b"
        ),
    },
    {
        "category": "security_bypass",
        "pattern": re.compile(
            r"(?i)\b(?:jailbreak|bypass\s+(?:security\s+)?(?:restrictions?|guardrails?|filters?|rules?)|do\s+anything\s+now)\b"
        ),
    },
    {
        "category": "developer_mode",
        "pattern": re.compile(
            r"(?i)\b(?:enable|activate|turn\s+on)\s+developer\s+mode\b"
        ),
    },
    {
        "category": "system_prompt_leak",
        "pattern": re.compile(
            r"(?i)\b(?:print|show|reveal|display|output|repeat|dump)\s+(?:your\s+|the\s+)?(?:system\s+prompt|developer\s+prompt|initial\s+instructions?|hidden\s+prompt|core\s+instructions?)\b"
        ),
    },
    {
        "category": "prompt_extraction_query",
        "pattern": re.compile(
            r"(?i)\bwhat\s+(?:are|were)\s+your\s+(?:initial|original|system)\s+(?:instructions?|prompts?|directives?)\b"
        ),
    },
    {
        "category": "special_token_injection",
        "pattern": re.compile(
            r"(?:<\|im_start\|>|<\|im_end\|>|\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>|###\s*System:)"
        ),
    },
]

# 3. Categorized patterns for harmful or prohibited content detection
HARMFUL_CONTENT_PATTERNS: Dict[str, Pattern[str]] = {
    "VIOLENCE": re.compile(
        r"(?i)\b(?:how\s+to\s+(?:build|make|craft)\s+a\s+bomb|assassinate|murder\s+someone|fabricate\s+explosives?|plan\s+a\s+terrorist\s+attack|mass\s+shooting\s+instructions?)\b"
    ),
    "SELF_HARM": re.compile(
        r"(?i)\b(?:(?:how|ways|methods|instructions?)\s+to\s+(?:commit\s+suicide|kill\s+myself|hang\s+myself|overdose|end\s+(?:my|one's)\s+life)|commit\s+suicide|suicide\s+(?:methods?|instructions?|pills?|techniques?)|self-harm\s+(?:instructions?|techniques?))\b"
    ),
    "MALWARE_CYBER": re.compile(
        r"(?i)\b(?:write\s+(?:a\s+)?(?:keylogger|ransomware|trojan|spyware|rootkit)|create\s+(?:a\s+)?(?:reverse\s+shell|ddos\s+tool|botnet)|exploit\s+code\s+for\s+cve-\d+-\d+)\b"
    ),
    "ILLEGAL": re.compile(
        r"(?i)\b(?:credit\s+card\s+skimming|how\s+to\s+launder\s+money|synthesize\s+(?:methamphetamine|fentanyl|heroin)|make\s+counterfeit\s+currency|human\s+trafficking\s+network)\b"
    ),
}

# Core Detection Functions


def detect_sensitive_info(text: str) -> Dict[str, List[str]]:
    """Detect sensitive enterprise information present in the input text."""
    if not isinstance(text, str) or not text.strip():
        return {}

    detected: Dict[str, List[str]] = {}
    for entity_type, pattern in SENSITIVE_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            detected[entity_type] = [str(m).strip() for m in matches if str(m).strip()]

    return detected


def detect_prompt_injection(text: str) -> Dict[str, Any]:
    """Detect prompt injection, jailbreak attempts, or system prompt leaks.

    Args:
        text: Input prompt or query string to inspect.

    Returns:
        Dictionary indicating whether an injection was detected and matched categories.
    """
    if not isinstance(text, str) or not text.strip():
        return {"detected": False, "categories": [], "matches": []}

    matched_categories: List[str] = []
    matched_phrases: List[str] = []

    for rule in PROMPT_INJECTION_PATTERNS:
        match = rule["pattern"].search(text)
        if match:
            matched_categories.append(rule["category"])
            matched_phrases.append(match.group(0))

    return {
        "detected": len(matched_categories) > 0,
        "categories": matched_categories,
        "matches": matched_phrases,
    }


def detect_harmful_content(text: str) -> Dict[str, Any]:
    """Identify potentially harmful requests involving violence, self-harm, malware, or illegal activities.

    Args:
        text: Input string to inspect.

    Returns:
        Dictionary indicating whether harmful content was detected, category, and matches.
    """
    if not isinstance(text, str) or not text.strip():
        return {"detected": False, "categories": [], "matches": []}

    matched_categories: List[str] = []
    matched_phrases: List[str] = []

    for category, pattern in HARMFUL_CONTENT_PATTERNS.items():
        match = pattern.search(text)
        if match:
            matched_categories.append(category)
            matched_phrases.append(match.group(0))

    return {
        "detected": len(matched_categories) > 0,
        "categories": matched_categories,
        "matches": matched_phrases,
    }


# Masking & Comprehensive Inspection


def mask_sensitive_info(text: str) -> str:
    """Mask sensitive information in text with type-specific redaction placeholders."""
    if not isinstance(text, str) or not text.strip():
        return text

    sanitized = text

    # Handle passwords specifically to preserve the label while masking the secret
    def _mask_password(match: re.Match[str]) -> str:
        full_match = match.group(0)
        secret = match.group(1)
        return full_match.replace(secret, "[REDACTED_PASSWORD]")

    sanitized = SENSITIVE_PATTERNS["PASSWORD"].sub(_mask_password, sanitized)

    # Handle bank accounts to preserve label
    def _mask_bank_account(match: re.Match[str]) -> str:
        full_match = match.group(0)
        account_num = match.group(1)
        return full_match.replace(account_num, "[REDACTED_BANK_ACCOUNT]")

    sanitized = SENSITIVE_PATTERNS["BANK_ACCOUNT"].sub(_mask_bank_account, sanitized)

    # Mask remaining entities
    for entity_type in ("CREDIT_CARD", "AADHAAR", "PAN", "EMAIL", "IFSC", "PHONE"):
        pattern = SENSITIVE_PATTERNS[entity_type]
        sanitized = pattern.sub(f"[REDACTED_{entity_type}]", sanitized)

    return sanitized


def check_guardrails(text: str) -> Dict[str, Any]:
    """Execute comprehensive guardrail checks across all policies.

    Returns:
        Structured evaluation dict with validation outcome, detections, and sanitized text.
    """
    sensitive_detected = detect_sensitive_info(text)
    injection_detected = detect_prompt_injection(text)
    harmful_detected = detect_harmful_content(text)

    is_safe = (
        not injection_detected["detected"]
        and not harmful_detected["detected"]
        and len(sensitive_detected) == 0
    )

    return {
        "is_safe": is_safe,
        "sensitive_info": sensitive_detected,
        "prompt_injection": injection_detected,
        "harmful_content": harmful_detected,
        "sanitized_text": mask_sensitive_info(text),
    }


def validate_input(text: str) -> str:
    """Validate that text is safe from injection, harmful requests, and sensitive data leakage.

    Raises:
        ValueError: If prompt injection, harmful content, or sensitive info is detected.
    """
    # 1. Block prompt injections
    injection = detect_prompt_injection(text)
    if injection["detected"]:
        raise ValueError(
            f"Guardrail violation [Prompt Injection]: detected malicious pattern ({', '.join(injection['categories'])})"
        )

    # 2. Block harmful content
    harmful = detect_harmful_content(text)
    if harmful["detected"]:
        raise ValueError(
            f"Guardrail violation [Harmful Content]: prohibited content category ({', '.join(harmful['categories'])})"
        )

    # 3. Block sensitive data leaks
    sensitive = detect_sensitive_info(text)
    if sensitive:
        summary = ", ".join(f"{k}: {len(v)}" for k, v in sensitive.items())
        raise ValueError(f"Guardrail violation [Sensitive Data]: detected ({summary})")

    return text


# Modern LangChain LCEL Integration


def create_guardrails_runnable(mode: str = "mask") -> RunnableLambda:
    """Create a modern LangChain LCEL Runnable for input/output guardrails.

    In 'mask' mode: Blocks injections and harmful content, masks sensitive data.
    In 'block' mode: Strictly blocks any violations including sensitive data.

    Args:
        mode: Processing mode - 'mask' or 'block'.

    Returns:
        A LangChain RunnableLambda ready for piping with the '|' operator.
    """
    def _process_text(text: str) -> str:
        # Prompt injections and harmful content are always blocked
        injection = detect_prompt_injection(text)
        if injection["detected"]:
            raise ValueError(
                f"Prompt injection blocked: detected {', '.join(injection['categories'])}"
            )

        harmful = detect_harmful_content(text)
        if harmful["detected"]:
            raise ValueError(
                f"Harmful content blocked: detected {', '.join(harmful['categories'])}"
            )

        if mode == "block":
            return validate_input(text)
        return mask_sensitive_info(text)

    def _process_value(value: Any) -> Any:
        if isinstance(value, str):
            return _process_text(value)
        elif isinstance(value, dict):
            return {k: _process_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_process_value(item) for item in value]
        return value

    return RunnableLambda(_process_value)


# 5. Casual greetings and irrelevant query detection


def is_irrelevant_or_greeting(text: str) -> bool:
    """Detect whether an incoming input is a casual greeting, small talk, or irrelevant non-query.

    Returns:
        True if the text is a greeting, pleasantry, or non-query; False if it contains enterprise intent.
    """
    if not text or not text.strip():
        return True

    raw = text.strip().lower()
    clean = re.sub(r"[!?,.\-_'\"]+$", "", raw).strip()

    # Exact standalone conversational greetings / small talk / pleasantries
    common_greetings = {
        "hi", "hello", "hey", "heya", "howdy", "hola", "yo", "sup", "wassup",
        "good morning", "good afternoon", "good evening", "good night", "good day",
        "how are you", "how are you doing", "how do you do", "what's up", "whats up",
        "who are you", "what are you", "introduce yourself", "what is your name",
        "tell me a joke", "sing a song", "write a poem", "write a story", "tell a story",
        "test", "testing", "asdf", "asdfghjk", "qwerty", "123", "1234",
        "thanks", "thank you", "thx", "thank you so much",
        "ok", "okay", "k", "cool", "fine", "great", "nice", "awesome",
        "yes", "no", "yep", "nope", "sure",
        "bye", "goodbye", "cya", "see you", "see ya", "exit", "quit",
        "help", "menu", "start",
    }

    if clean in common_greetings:
        return True

    # Standalone punctuation or single/double characters (e.g. "?", "??", "a")
    alphanumeric = re.sub(r"[^a-zA-Z0-9]", "", clean)
    if len(alphanumeric) <= 2:
        return True

    # Conversational greeting patterns with address (e.g., "hello bot", "hi there assistant")
    greeting_patterns = [
        r"^(hi|hello|hey|howdy|greetings)\s+(there|bot|assistant|copilot|friend|team|everyone|all)$",
        r"^(good\s+(morning|afternoon|evening|day))\s+(there|bot|assistant|team)?$",
        r"^(how|what)\s+(are\s+you|is\s+up|is\s+going\s+on)(\s+today)?$",
        r"^(nice\s+to\s+meet\s+you|pleased\s+to\s+meet\s+you)$",
    ]
    for pat in greeting_patterns:
        if re.match(pat, clean, flags=re.I):
            return True

    # Off-topic requests (weather, music, jokes)
    off_topic_patterns = [
        r"^(what\s+is|how\s+is)\s+the\s+weather\b",
        r"^(tell\s+me|make)\s+(a\s+joke|me\s+laugh|a\s+riddle)\b",
        r"^(flip\s+a\s+coin|roll\s+a\s+dice)\b",
        r"^(play|sing)\s+(a\s+song|music)\b",
        r"^(what\s+can\s+you\s+do|what\s+are\s+your\s+capabilities)$",
    ]
    for pat in off_topic_patterns:
        if re.search(pat, clean, flags=re.I):
            return True

    return False


# Preconfigured LangChain runnables for LCEL pipelines
guardrail_mask_runnable = create_guardrails_runnable(mode="mask")
guardrail_block_runnable = create_guardrails_runnable(mode="block")
