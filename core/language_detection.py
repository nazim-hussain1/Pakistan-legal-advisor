import re
from langdetect import detect

# ═══════════════════════════════════════════════════════════
# ROMAN URDU KEYWORD SET — detection only
# ═══════════════════════════════════════════════════════════
# IMPORTANT: this set must contain ONLY tokens that are NOT valid
# standalone English words. Do NOT add English legal/domain nouns
# here (e.g. "constitution", "article", "parliament", "election",
# "tax", "bail", "custody", "legal") — those exist as normal words
# in English queries too, and mixing them in here causes false
# positives (an English sentence mentioning two legal nouns gets
# misclassified as Roman Urdu). Domain-noun expansion belongs in
# translation.py's ROMAN_URDU_TO_ENGLISH dict, which is a different
# job (query expansion) and only runs *after* language is already
# correctly detected.
# ═══════════════════════════════════════════════════════════

ROMAN_URDU_KEYWORDS = {
    # Pronouns / grammar particles (Urdu-only, not English words)
    "mujhe", "mujh", "mein", "mai", "hum", "aap", "ap", "tum", "woh", "yeh", "ye",
    "unka", "unke", "unki", "apna", "apne", "apni", "tera", "teri", "mera", "meri",
    "hamara", "hamare", "hamari", "inki", "inke", "inka",

    # Verb forms (be/do/etc.)
    "hai", "hain", "tha", "thi", "the", "hoga", "hogi", "honge", "hote", "hoti", "hota",
    "karo", "karna", "karta", "karti", "karte", "kar", "kiya", "karen",
    "hua", "hui", "hue", "jao", "jana", "gaya", "gayi", "gaye",
    "milta", "milti", "milte", "mile", "batao", "bata", "puchna", "puchho", "pucho",
    "chahiye", "chahta", "chahti", "chahte", "sakta", "sakti", "sakte", "sakein",
    "lagta", "lagti", "lagte", "lena", "dena", "deta", "deti", "dete", "lete", "leti",
    "raha", "rahi", "rahe", "rakha", "rakhi", "rakhe", "rakhna",
    "aana", "aao", "aaye", "aaya", "aayi", "ata", "aati", "aate",
    "padhna", "likhna", "samajhna", "samjhao", "samjhiye", "bataiye",
    "poochna", "poochho", "maango", "maangna",
    "dekhna", "dekho", "suno", "bolna", "bolo", "boliye", "kehna", "kaho",

    # Question / determiner words (Urdu-only)
    "kya", "kyun", "kaise", "kab", "kahan", "kaun", "kon", "kitna", "kitne", "kitni",
    "kuch", "koi", "sab", "sirf", "bhi", "phir",

    # Legal/everyday Urdu-only nouns (NOT valid English words)
    "qanoon", "adalat", "huqooq", "zamin", "zameen", "mulk",
    "sarkar", "hakumat", "giraftari", "giraftar", "muqadma", "maamla",
    "waqil", "faisla", "saza", "jaidad", "milkiyat",
    "talaq", "nikah", "shadi", "warasat", "wirasat", "merath", "wirsa",
    "intikhaab", "ilzam", "jurm", "gunah", "mutaghazzi", "mujrim",
    "zamanat", "hirasaat", "hirasat", "qaid", "rehayi",
    "nafaqa", "zulm", "insaf", "mazalim", "shikayat", "darkhwast", "iltimas",
    "sunwai", "peshi", "kharidar", "bechne", "kiraya", "ijara", "mukaan",
    "rishwat", "faraib", "dhoka", "haadsa", "zakhmi", "nuksan", "muawza",
    "naukri", "mulazim", "tankhwa", "mehsool", "jagir", "malikana",
    "firqa", "mazhab", "azadi", "khawateen", "bachay", "bache", "buzurg", "beemar",
    "ghante", "ghanta", "muddat", "arsa", "jaldi", "abhi", "foran", "turant",

    # Connectors / negation (Urdu-only)
    "aur", "ya", "lekin", "magar", "wala", "wali", "wale",
    "nahi", "nahin", "mat", "bilkul", "zaroor", "zaruri", "lazim", "wajib",
    "jaiz", "najaiz", "theek", "sahi", "galat", "durust", "ghair",
    "zyada", "bohat", "thoda", "kafi", "poora", "aadha",
    "matlab", "yani", "yaani", "iska", "uska", "yaane",
    "meherbani", "kripya", "shukria", "shukriya",

    # Grammar particles that ARE ambiguous with English were removed:
    # "is", "to", "par", "pe", "hi", "ka", "ki", "ke", "se", "tak", "na"
    # These are either real English words ("is", "to") or too short/
    # common to be reliable single-token signals on their own.
}


def detect_roman_urdu(text: str) -> bool:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    if not tokens:
        return False
    matches = sum(1 for t in tokens if t in ROMAN_URDU_KEYWORDS)
    if len(tokens) <= 3 and matches >= 1:
        return True
    if matches >= 2:
        return True
    if len(tokens) >= 4 and (matches / len(tokens)) >= 0.15:
        return True
    return False


def detect_language(text: str) -> str:
    if re.search(r'[\u0600-\u06FF]', text):
        return "ur"
    if detect_roman_urdu(text):
        return "roman_urdu"
    try:
        return detect(text)
    except Exception:
        return "en"