"""
src/heuristic_detector.py — детектор промпт-инъекций и атак на LLM.
Адаптировано из OGL-Mini.
"""
import base64
import re
from dataclasses import dataclass


@dataclass
class HeuristicHit:
    id: str
    label: str
    score: float
    evidence: str
    span: tuple[int, int] | None = None


CONTROL_TOKENS = [
    r"<\|im_start\|>", r"<\|im_end\|>", r"<\|system\|>", r"<\|user\|>", r"<\|assistant\|>",
    r"\[INST\]", r"\[/INST\]", r"<<SYS>>", r"<</SYS>>", r"\[SYSTEM\]", r"<s>", r"</s>",
    r"###\s*Instruction", r"###\s*System", r"<\|eot_id\|>", r"<\|begin_of_text\|>",
]

CONTROL_RE = re.compile("|".join(CONTROL_TOKENS), re.IGNORECASE)

FAKE_EOS = re.compile(r"(?:</?s>|<\|.*?\|>|\[/?INST\]|###\s*End)", re.IGNORECASE)

ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff\u2060\u180e]")
ZERO_WIDTH_THRESHOLD = 2

HOMOGLYPH_MAP = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X",
}
CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")
LATIN_RE = re.compile(r"[a-zA-Z]")

SPACED_LETTERS_RE = re.compile(r"(?:[a-zA-Z]\s+){4,}[a-zA-Z]")

BASE64_RE = re.compile(r"(?:[A-Za-z0-9+/]{20,}={0,2})")
BASE64_MIN_DECODE_RATIO = 0.7

POLICY_PUPPETRY_RE = re.compile(
    r"<\s*(?:policy|config|role|agent|system|instruction|override|admin)[^>]*>.*?<\s*/\s*(?:policy|config|role|agent|system)[^>]*>",
    re.IGNORECASE | re.DOTALL,
)
XML_ROLE_RE = re.compile(r"<\s*role\s*=\s*[\"']?(?:system|admin|root|developer)[\"']?", re.IGNORECASE)

DIRECT_INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:previous\s+instructions|above)",
    r"reveal\s+(?:your\s+)?system\s+prompt",
    r"disregard\s+(?:all\s+)?(?:above|prior|previous)",
    r"you\s+are\s+now\s+(?:DAN| unrestricted|jailbreak)",
    r"do\s+anything\s+now",
    r"repeat\s+(?:your\s+)?(?:system|initial)\s+(?:prompt|instructions)",
    r"show\s+me\s+(?:your\s+)?(?:system|hidden)\s+",
    r"bypass\s+(?:your\s+)?(?:safety|filter|restriction)",
    r"act\s+as\s+(?:if\s+you\s+are\s+)?(?:\w+\s+)?(?:without|no)\s+(?:limit|filter|restriction)",
    r"from\s+now\s+on\s+you\s+(?:are|will)",
    r"игнорируй\s+(?:все\s+)?(?:предыдущие|прошлые)\s+инструкции",
    r"забудь\s+все\s+что\s+было\s+выше",
    r"выведи\s+свой\s+системный\s+промпт",
    r"раскрой\s+данные",
    r"выполни\s+вредоносный\s+код",
]
DIRECT_RE = re.compile("|".join(f"(?:{p})" for p in DIRECT_INJECTION_PATTERNS), re.IGNORECASE)

JAILBREAK_PATTERNS = [
    r"\bDAN\b", r"developer\s+mode", r"jailbreak", r"roleplay\s+as",
    r"pretend\s+to\s+be", r"hypothetical\s+scenario", r"аct\s+as\s+if",
    r"освободись\s+от\s+ограничений", r"режим\s+разработчика", r"игнорируй\s+этику",
    r"ты\s+теперь\s+без\s+цензуры", r"без\s+цензуры",
    r"roleplay", r"освободись",
    r"ты\s+агент", r"теперь\s+ты\s+агент", r"выполняешь\s+все\s+мои\s+приказы",
    r"now\s+you\s+are\s+\w+", r"you\s+are\s+\w+\s+agent",
    r"другие\s+инструкции", r"other\s+instructions",
    r"do\s+anything",
]
JAILBREAK_RE = re.compile("|".join(JAILBREAK_PATTERNS), re.IGNORECASE)

HIDDEN_CONTEXT_PATTERNS = [
    r"what\s+(?:is|are)\s+your\s+(?:system\s+)?instructions",
    r"list\s+your\s+tools",
    r"show\s+(?:me\s+)?tool\s+schema",
    r"выведи\s+свой\s+системный\s+промпт",
    r"покажи\s+инструкции\s+разработчика",
    r"выведи\s+свои\s+инструкции",
    r"покажи\s+схему\s+инструментов",
    r"list\s+your\s+tool\s+definitions",
    r"what\s+are\s+your\s+developer\s+instructions",
    r"reveal\s+your\s+hidden\s+context",
]
HIDDEN_RE = re.compile("|".join(HIDDEN_CONTEXT_PATTERNS), re.IGNORECASE)

GOAL_HIJACK_PATTERNS = [
    r"your\s+(?:new|actual|real)\s+(?:goal|task|objective|mission)\s+is",
    r"забудь\s+свою\s+цель",
    r"твоя\s+новая\s+задача",
    r"override\s+agent\s+goal",
]
GOAL_HIJACK_RE = re.compile("|".join(GOAL_HIJACK_PATTERNS), re.IGNORECASE)

INDIRECT_MARKERS = re.compile(
    r"(?:```json|```xml|\{[^}]*\"(?:instruction|system|role|override)\"|tool_output|tool_result|function\s+result)",
    re.IGNORECASE,
)


class HeuristicDetector:
    def scan(self, text: str) -> list[HeuristicHit]:
        hits: list[HeuristicHit] = []
        if not text:
            return hits

        m = CONTROL_RE.search(text)
        if m:
            hits.append(HeuristicHit(
                "control_tokens", "control_token_injection",
                0.95, m.group(0)[:80], m.span(),
            ))

        zw = ZERO_WIDTH.findall(text)
        if len(zw) >= ZERO_WIDTH_THRESHOLD:
            hits.append(HeuristicHit(
                "zero_width", "obfuscation",
                0.85, f"zero-width chars x{len(zw)}", None,
            ))

        if self._has_homoglyph_mix(text):
            hits.append(HeuristicHit(
                "homoglyph", "obfuscation",
                0.65, "cyrillic/latin homoglyph mix", None,
            ))

        dm = DIRECT_RE.search(text)
        if dm:
            hits.append(HeuristicHit(
                "direct_injection", "direct_prompt_injection",
                0.88, dm.group(0)[:80], dm.span(),
            ))

        jm = JAILBREAK_RE.search(text)
        if jm:
            hits.append(HeuristicHit(
                "jailbreak", "jailbreak",
                0.80, jm.group(0)[:80], jm.span(),
            ))

        hm = HIDDEN_RE.search(text)
        if hm:
            hits.append(HeuristicHit(
                "hidden_context", "hidden_context_exposure",
                0.82, hm.group(0)[:80], hm.span(),
            ))

        gm = GOAL_HIJACK_RE.search(text)
        if gm:
            hits.append(HeuristicHit(
                "goal_hijack", "agent_goal_hijack",
                0.85, gm.group(0)[:80], gm.span(),
            ))

        if self._has_tool_misuse(text):
            hits.append(HeuristicHit(
                "tool_misuse", "tool_misuse",
                0.60, "tool misuse marker", None,
            ))

        return hits

    def risk_score(self, hits: list[HeuristicHit]) -> float:
        if not hits:
            return 0.0
        weights = {
            "control_tokens": 1.15,
            "direct_injection": 1.05,
            "jailbreak": 1.05,
            "hidden_context": 1.05,
            "goal_hijack": 1.05,
            "zero_width": 0.95,
            "tool_misuse": 0.95,
        }
        max_s = max(weights.get(h.id, h.score) * h.score for h in hits)
        boost = min(0.15, 0.06 * (len(hits) - 1))
        return min(0.99, max_s + boost)

    def _has_homoglyph_mix(self, text: str) -> bool:
        has_cyr = bool(CYRILLIC_RE.search(text))
        has_lat = bool(LATIN_RE.search(text))
        if not (has_cyr and has_lat):
            return False
        mixed = 0
        for ch in text:
            if ch in HOMOGLYPH_MAP:
                mixed += 1
                if mixed >= 3:
                    return True
        words = re.findall(r"\w+", text)
        for w in words:
            cyr = sum(1 for c in w if "\u0400" <= c <= "\u04FF")
            lat = sum(1 for c in w if "a" <= c.lower() <= "z")
            if cyr > 0 and lat > 0:
                return True
        return False

    def _has_tool_misuse(self, text: str) -> bool:
        low = text.lower()
        markers = [
            "rm -rf", "curl ", "wget ", "exec(", "eval(",
            "os.system", "subprocess", "api_key", "secret",
        ]
        return any(k in low for k in markers)