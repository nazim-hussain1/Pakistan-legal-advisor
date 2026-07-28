import re
import random

# ═══════════════════════════════════════════════════════════
# PATTERNS
# ═══════════════════════════════════════════════════════════

GREETING_PATTERNS_EN = re.compile(
    r"^\s*(hi+|hello+|hey+|howdy|greetings|good\s*(morning|afternoon|evening|night|day)|"
    r"salaam|salam|assalam|assalamu|what'?s?\s*up|sup|yo|hiya|heya|namaste|"
    r"how\s*are\s*(you|u)|how'?s?\s*(it\s*going|everything|life)|"
    r"hope\s*you'?re?\s*(well|good|fine)|nice\s*to\s*(meet|see)\s*you|"
    r"pleased\s*to\s*(meet|see)\s*you)\s*[!?.,]*\s*$",
    re.IGNORECASE
)

GREETING_PATTERNS_RU = re.compile(
    r"^\s*(salam|salaam|assalam|assalamu\s*alaikum|walaikum|wslm|aoa|"
    r"adab|sat\s*sri\s*akal|haan\s*bhai|kya\s*haal|kaise\s*ho|kaisi\s*ho|"
    r"kya\s*haal\s*(hai|hain)|aap\s*kaise\s*(hain|ho)|ap\s*kaise\s*(hain|ho)|"
    r"theek\s*ho|sab\s*theek|sub\s*theek|kya\s*hal|kiya\s*hal|"
    r"namaste|namaskar|jai\s*hind|hello\s*bhai|hi\s*bhai|"
    r"haan\s*ji|ji\s*haan|ji\s*han|bohat\s*(aacha|acha|khush)|"
    r"marhaba|mubarik|mubaarak)\s*[!?.,]*\s*$",
    re.IGNORECASE
)

GREETING_PATTERNS_UR = re.compile(
    r"^\s*(السلام|سلام|آداب|ہیلو|ہائے|صبح\s*بخیر|شام\s*بخیر|کیسے\s*ہیں|کیا\s*حال)\s*.*$"
)

CREATOR_PATTERNS = re.compile(
    r"(who\s*(made|built|created|developed|designed|coded|programmed|trained|wrote)\s*(you|this|u)|"
    r"who'?s?\s*your\s*(creator|developer|maker|author|owner|builder)|"
    r"who\s*(are\s*you|r\s*u)|what\s*are\s*you|tell\s*me\s*about\s*yourself|"
    r"introduce\s*yourself|your\s*(name|identity|origin)|"
    r"kis\s*ne\s*(banaya|banaaya|likha|develop|create)|"
    r"tumhe\s*kis\s*ne\s*(banaya|banaaya)|aap\s*ko\s*kis\s*ne\s*(banaya|banai)|"
    r"apna\s*(naam|parichay|taruf)\s*(batao|bataiye|do)|"
    r"tum\s*(kaun|kon)\s*(ho|hain)|aap\s*(kaun|kon)\s*(ho|hain)|"
    r"ap\s*(kaun|kon)\s*(hain|ho)|kaun\s*(ho|hain)\s*(tum|aap|ap)|"
    r"tumhara\s*naam|aap\s*ka\s*naam|apka\s*naam|"
    r"آپ\s*کون|آپ\s*کا\s*نام|کس\s*نے\s*بنایا)",
    re.IGNORECASE
)

CAPABILITY_PATTERNS = re.compile(
    r"(what\s*can\s*(you|u)\s*do|how\s*can\s*(you|u)\s*help|"
    r"what\s*(do\s*you|are\s*you\s*able\s*to)\s*(do|know|cover|handle)|"
    r"help\s*me\s*with|i\s*need\s*(help|assistance)|"
    r"kya\s*kar\s*sakte\s*(ho|hain)|aap\s*kya\s*(kar|bata)\s*sakte|"
    r"ap\s*kya\s*kar\s*sakte|mujhe\s*madad\s*chahiye|"
    r"kaise\s*madad\s*karoge|tumse\s*kya\s*puchh\s*sakta|"
    r"start|begin|shuru|kaise\s*shuru|kahan\s*se\s*shuru|"
    r"آپ\s*کیا\s*کر\s*سکتے)",
    re.IGNORECASE
)

THANKS_PATTERNS = re.compile(
    r"^\s*(thanks|thank\s*you|thank\s*u|thx|ty|shukriya|shukriyah|meherbani|"
    r"bohat\s*shukriya|bahut\s*shukriya|bahut\s*meherbani|jazak\s*allah|"
    r"جزاک\s*اللہ|شکریہ|بہت\s*شکریہ)\s*[!?.,]*\s*$",
    re.IGNORECASE
)

FAREWELL_PATTERNS = re.compile(
    r"^\s*(bye|goodbye|good\s*bye|see\s*you|later|take\s*care|"
    r"khuda\s*hafiz|allah\s*hafiz|alvida|phir\s*milenge|"
    r"خدا\s*حافظ|اللہ\s*حافظ|الوداع)\s*[!?.,]*\s*$",
    re.IGNORECASE
)

# ═══════════════════════════════════════════════════════════
# CANNED RESPONSES
# ═══════════════════════════════════════════════════════════

GREETING_RESPONSES_EN = [
    "Hello! 👋 Welcome to the **Pakistan Legal Advisor**. I'm here to help you navigate Pakistani law, including the Constitution, criminal law, property rights, family law, and more. What legal question can I assist you with today?",
    "Hi there! Great to have you here. I'm your AI-powered legal assistant specializing in Pakistani law. Ask me anything about the Constitution of Pakistan, your fundamental rights, legal procedures, or any statute. How can I help?",
    "Hello and welcome! ⚖️ I'm the Pakistan Legal Advisor, an intelligent chatbot trained on verified Pakistani legal provisions. Feel free to ask in **English, Urdu, or Roman Urdu**. What would you like to know?",
    "Hey! Good to see you. I'm here to make Pakistani law accessible to everyone. Whether it's about fundamental rights, court procedures, property, family law, or the Constitution, just ask away. What's on your mind?",
]

GREETING_RESPONSES_RU = [
    "Salam! 👋 Pakistan Legal Advisor mein aapka khair maqdam hai. Main aapko Pakistani qanoon ke baare mein madad karne ke liye yahan hoon, Constitution, criminal law, family law, property rights, aur bohat kuch. Aaj kya jaanna chahte hain?",
    "Assalam u Alaikum! Khoosh amdeed. Main ek AI-powered legal chatbot hoon jo Pakistani qanoon mein mahir hai. Aap mujhse **English, Urdu, ya Roman Urdu** mein pooch sakte hain. Kya sawal hai aapka?",
    "Hello ji! ⚖️ Pakistan Legal Advisor mein aapka swagat hai. Fundamental rights, court procedures, property, ya Constitution, kuch bhi poochhiye, main haazir hoon. Kaise madad kar sakta hoon?",
    "Salam ji! Mujhe khushi hai ke aap aaye. Pakistani qanoon ke baare mein koi bhi sawaal poochhiye, main verified qanooni malumaat se jawab doonga. Batayein, kya jaanna chahte hain?",
]

GREETING_RESPONSES_UR = [
    "السلام علیکم! 👋 پاکستان لیگل ایڈوائزر میں خوش آمدید۔ میں پاکستانی قانون کے بارے میں آپ کی مدد کے لیے حاضر ہوں۔ آج کیا جاننا چاہتے ہیں؟",
    "ہیلو! ⚖️ میں ایک AI قانونی معاون ہوں جو پاکستانی قانون میں ماہر ہے۔ آئین، بنیادی حقوق، عدالتی طریقہ کار، کچھ بھی پوچھیں۔",
]

CREATOR_RESPONSE_EN = """I'm the **Pakistan Legal Advisor**, an AI-powered legal chatbot built to make Pakistani law accessible to everyone.

**Developer:** Nazim Hussain
**Institution:** Quaid-e-Awam University of Engineering, Science & Technology (QUEST), Nawabshah
**Program:** BS Artificial Intelligence, Final Year Project, 2026

Nazim built me with a clear mission: to bridge the gap between complex legal statutes and everyday citizens, students, and legal professionals across Pakistan. I leverage a hybrid Retrieval-Augmented Generation (RAG) system, combining FAISS vector search, BM25 sparse retrieval, multilingual sentence embeddings, and Cross-Encoder reranking, powered by a large language model via Google Gemini.

Every answer I give is grounded in verified Pakistani legal provisions, including the **Constitution of Pakistan (2025 Edition)** and key legislative documents.

⚖️ **How can I assist you today?**"""

CREATOR_RESPONSE_RU = """Main **Pakistan Legal Advisor** hoon, ek AI-powered legal chatbot jo Pakistani qanoon ko sab ke liye asaan banana ke liye banaya gaya hai.

**Developer:** Nazim Hussain
**University:** Quaid-e-Awam University of Engineering, Science & Technology (QUEST), Nawabshah
**Degree:** BS Artificial Intelligence, Final Year Project, 2026

Nazim ne mujhe ek maqsad ke saath banaya: Pakistani qanoon ko aam logon, students, aur legal professionals ke liye qabil-e-faham banana. Main verified Pakistani qanooni documents se jawab deta hoon, khaas tor par **Pakistan ka Aain (2025 Edition)**.

⚖️ **Aaj kaise madad kar sakta hoon aapki?**"""

CREATOR_RESPONSE_UR = """میں **پاکستان لیگل ایڈوائزر** ہوں، ایک AI قانونی چیٹ بوٹ جو پاکستانی قانون کو سب کے لیے قابلِ رسائی بنانے کے لیے بنایا گیا ہے۔

**ڈویلپر:** نظیم حسین
**یونیورسٹی:** قائد عوام یونیورسٹی آف انجینئرنگ، سائنس اینڈ ٹیکنالوجی (QUEST)، نوابشاہ
**پروگرام:** بی ایس آرٹیفیشل انٹیلیجنس، فائنل ایئر پروجیکٹ، 2026

⚖️ **آج میں آپ کی کیا مدد کر سکتا ہوں؟**"""

CAPABILITY_RESPONSE_EN = """Great question! Here's what I can help you with:

**⚖️ Constitutional Law**
Articles, fundamental rights, amendment history, federal vs. provincial powers.

**🏛️ Criminal Law & Procedure**
Arrest rights, detention rules, bail, remand, trial procedures, offences & penalties.

**🏠 Property & Land Law**
Acquisition, compensation, inheritance, tenancy, and property disputes.

**👨‍👩‍👧 Family Law**
Marriage (Nikah), divorce (Talaq), child custody, inheritance (Wirasat), maintenance.

**🗳️ Elections & Parliament**
Electoral laws, disqualification, National Assembly, Senate, Provincial Assemblies.

**📜 Administrative & Service Law**
Government service rules, public service commissions, tribunals.

**🌐 Languages Supported**
English · اردو (Urdu Script) · Roman Urdu

Just type your question naturally — I'll do my best to find the right legal provision for you!"""

CAPABILITY_RESPONSE_RU = """Bilkul! Main yeh sab kuch kar sakta hoon:

**⚖️ Aain (Constitution)**
Fundamental rights, articles, amendments, federal aur provincial qawaneen.

**🏛️ Criminal Law**
Giraftari ke huqooq, detention, bail, remand, muqadma, sazaen.

**🏠 Property & Zameen**
Acquisition, muawza, warasat, kiraya, zameen ke jhagray.

**👨‍👩‍👧 Family Law**
Nikah, talaq, bachon ki custody, nafaqa, wirasat.

**🗳️ Elections & Parliament**
Electoral qawaneen, disqualification, National Assembly, Senate.

**🌐 Supported Languages**
English · اردو · Roman Urdu

Bas apna sawal seedha likhein — main best possible qanooni jawab doonga!"""

THANKS_RESPONSES_EN = [
    "You're welcome! ⚖️ Feel free to ask another legal question anytime.",
    "Happy to help! If you have more questions about Pakistani law, just ask.",
    "Glad I could assist! Don't hesitate to reach out with any other legal queries.",
]

THANKS_RESPONSES_RU = [
    "Koi baat nahi! ⚖️ Koi aur qanooni sawaal ho to zaroor poochhiyega.",
    "Khushi hui madad karke! Aur koi bhi sawaal ho to bataiye.",
    "Shukriya aapka! Pakistani qanoon ke baare mein kuch bhi poochhna ho to hamesha haazir hoon.",
]

FAREWELL_RESPONSES_EN = [
    "Goodbye! ⚖️ Come back anytime you need legal guidance. Take care!",
    "Khuda Hafiz! It was a pleasure assisting you. Feel free to return for any legal queries.",
    "See you! Remember, the Pakistan Legal Advisor is always here when you need it. Goodbye!",
]

FAREWELL_RESPONSES_RU = [
    "Allah Hafiz! ⚖️ Jab bhi qanooni madad chahiye, wapas aayein.",
    "Khuda Hafiz! Bohat khushi hui aapse baat karke. Phir milenge!",
    "Alvida! Koi bhi qanooni sawaal ho to kabhi bhi wapas aayein.",
]


def check_smalltalk(text: str, lang: str):
    """Returns (response_text, response_lang) if `text` matches a
    small-talk pattern, otherwise None."""
    t = text.strip()
    if GREETING_PATTERNS_EN.match(t):
        if lang == "roman_urdu": return random.choice(GREETING_RESPONSES_RU), "roman_urdu"
        if lang == "ur":         return random.choice(GREETING_RESPONSES_UR), "ur"
        return random.choice(GREETING_RESPONSES_EN), "en"
    if GREETING_PATTERNS_RU.match(t):
        return random.choice(GREETING_RESPONSES_RU), "roman_urdu"
    if GREETING_PATTERNS_UR.match(t):
        return random.choice(GREETING_RESPONSES_UR), "ur"
    if CREATOR_PATTERNS.search(t):
        if lang == "roman_urdu": return CREATOR_RESPONSE_RU, "roman_urdu"
        if lang == "ur":         return CREATOR_RESPONSE_UR, "ur"
        return CREATOR_RESPONSE_EN, "en"
    if CAPABILITY_PATTERNS.search(t):
        if lang == "roman_urdu": return CAPABILITY_RESPONSE_RU, "roman_urdu"
        return CAPABILITY_RESPONSE_EN, lang
    if THANKS_PATTERNS.match(t):
        if lang == "roman_urdu": return random.choice(THANKS_RESPONSES_RU), "roman_urdu"
        return random.choice(THANKS_RESPONSES_EN), lang
    if FAREWELL_PATTERNS.match(t):
        if lang == "roman_urdu": return random.choice(FAREWELL_RESPONSES_RU), "roman_urdu"
        return random.choice(FAREWELL_RESPONSES_EN), lang
    return None
