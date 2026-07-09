"""
Query-side translation layer: bridges Roman Urdu and Urdu-script queries
to the English legal vocabulary used in the constitutional dataset, and
expands all queries with related legal synonyms/article references.
"""
import re

# ═══════════════════════════════════════════════════════════
# ROMAN URDU → ENGLISH
# ═══════════════════════════════════════════════════════════

ROMAN_URDU_TO_ENGLISH = {
    "giraftari": "arrest", "giraftaar": "arrested detained",
    "giraftar": "arrested detained", "hirasaat": "custody detention",
    "hirasat": "custody detention", "qaid": "imprisonment custody",
    "nazar band": "detained house arrest", "band karna": "detention imprisonment",
    "pakad liya": "arrested detained", "pakad": "arrest apprehension",
    "harasat": "custody detention", "remand": "remand detention custody",
    "bail": "bail release", "zamanat": "bail surety",
    "riha": "release discharged freed", "chhorna": "release discharge",
    "rehayi": "release discharge bail",
    "ghante mein": "within hours time period",
    "kitne ghante": "how many hours within period",
    "24 ghante": "twenty four hours 24 hours period",
    "48 ghante": "forty eight hours 48 hours period",
    "ghante": "hours period time", "din mein": "within days period",
    "kitne din": "how many days period", "muddat": "period duration time limit",
    "arsa": "period duration", "waqt": "time period limit",
    "magistrate ke saamne pesh": "produced before magistrate court",
    "pesh karna": "produced before appear court",
    "saamne pesh": "produced before appear", "peshi": "appearance hearing court",
    "magistrate": "magistrate court", "adalat": "court tribunal judicature",
    "sunwai": "hearing proceeding trial", "faisla": "judgment order decision",
    "hukum": "order direction decree", "appeal": "appeal appellate review",
    "nazar sani": "review revision", "ijlas": "session sitting",
    "maqadma": "case proceedings lawsuit", "muqadma": "case legal proceedings lawsuit",
    "maamla": "matter case issue", "inquiry": "inquiry investigation",
    "tahqiqat": "investigation inquiry", "gawah": "witness testimony",
    "saboot": "evidence proof", "iqrar": "confession admission",
    "bayan": "statement testimony", "waqil": "advocate lawyer legal practitioner",
    "judge": "judge justice court", "supreme court": "supreme court chief justice",
    "high court": "high court justice",
    "haq": "right entitlement fundamental right",
    "huqooq": "rights fundamental rights", "azadi": "freedom liberty",
    "hurriyat": "freedom liberty rights", "insaf": "justice fair trial due process",
    "barabar": "equality equal rights", "tabheez": "discrimination equal protection",
    "boli ki azadi": "freedom of speech expression",
    "mazhab ki azadi": "freedom of religion",
    "ijtima ki azadi": "freedom of assembly",
    "insan ki izzat": "dignity of man inviolable",
    "free speech": "freedom of speech expression article 19",
    "zamin": "land property immovable", "zameen": "land property immovable",
    "jaidad": "property assets estate", "milkiyat": "ownership property right",
    "mukaan": "house property dwelling", "plot": "plot land property",
    "muawza": "compensation payment indemnity", "zer qabd": "possession acquisition",
    "qabza": "possession occupation", "kiraya": "rent tenancy lease",
    "ijara": "lease tenancy", "bechi": "sale transfer property",
    "khareed": "purchase acquisition",
    "talaq": "divorce dissolution marriage", "talaaq": "divorce dissolution marriage",
    "nikah": "marriage matrimonial contract", "shadi": "marriage matrimonial",
    "warasat": "inheritance succession", "wirasat": "inheritance succession",
    "merath": "inheritance legal heirs estate", "wirsa": "inheritance estate succession",
    "nafaqa": "maintenance alimony financial support",
    "hirasat bachay": "child custody guardianship",
    "bache ki custody": "child custody guardianship",
    "mehr": "dower mahr marriage payment", "iddat": "iddah waiting period divorce",
    "hakumat": "government federal government", "sarkar": "government state",
    "parliament": "parliament majlis-e-shoora national assembly",
    "assembly": "assembly legislature provincial assembly",
    "senate": "senate upper house parliament",
    "vote": "vote election franchise", "intikhaab": "election electoral franchise",
    "wazir-e-azam": "prime minister chief executive", "PM": "prime minister",
    "CM": "chief minister province", "governor": "governor province",
    "president": "president head of state",
    "jurm": "offence crime criminal", "gunah": "offence crime",
    "saza": "punishment sentence penalty", "ilzam": "charge accusation allegation",
    "mujrim": "criminal accused convict", "be-gunah": "innocent not guilty acquittal",
    "rishwat": "bribery corruption", "faraib": "fraud deceit",
    "qatl": "murder homicide", "chori": "theft larceny", "FIR": "FIR first information report",
    "naukri": "employment service job", "mulazim": "employee servant service",
    "tankhwa": "salary remuneration pay", "pension": "pension retirement benefit",
    "tax": "tax levy duty", "mehsool": "tax revenue",
    "taleem": "education right to education",
    "free education": "free compulsory education article 25A",
    "aain": "constitution constitutional", "article": "article provision constitutional",
    "section": "section provision law", "qanoon ki roo se": "according to law legal provision",
}

URDU_SCRIPT_TO_ENGLISH = {
    "گرفتاری": "arrest", "گرفتار": "arrested detained",
    "حراست": "custody detention", "قید": "imprisonment custody",
    "ریمانڈ": "remand detention custody", "ضمانت": "bail release surety",
    "رہائی": "release discharge bail",
    "گھنٹے": "hours period time", "دنوں": "days period",
    "مدت": "period duration time limit",
    "مجسٹریٹ": "magistrate court", "عدالت": "court tribunal judicature",
    "سماعت": "hearing proceeding trial", "فیصلہ": "judgment order decision",
    "اپیل": "appeal appellate review", "مقدمہ": "case proceedings lawsuit",
    "تحقیقات": "investigation inquiry", "گواہ": "witness testimony",
    "ثبوت": "evidence proof", "اقرار": "confession admission",
    "بیان": "statement testimony", "وکیل": "advocate lawyer legal practitioner",
    "جج": "judge justice court", "سپریم کورٹ": "supreme court chief justice",
    "ہائی کورٹ": "high court justice",
    "حق": "right entitlement fundamental right",
    "حقوق": "rights fundamental rights", "آزادی": "freedom liberty",
    "انصاف": "justice fair trial due process",
    "برابری": "equality equal rights", "امتیاز": "discrimination equal protection",
    "آزادی اظہار": "freedom of speech expression",
    "مذہب کی آزادی": "freedom of religion",
    "وقار انسانی": "dignity of man inviolable",
    "زمین": "land property immovable", "جائیداد": "property assets estate",
    "ملکیت": "ownership property right", "مکان": "house property dwelling",
    "معاوضہ": "compensation payment indemnity",
    "قبضہ": "possession occupation", "کرایہ": "rent tenancy lease",
    "طلاق": "divorce dissolution marriage", "نکاح": "marriage matrimonial contract",
    "شادی": "marriage matrimonial", "وراثت": "inheritance succession",
    "میراث": "inheritance estate succession",
    "نفقہ": "maintenance alimony financial support",
    "حضانت": "child custody guardianship", "مہر": "dower mahr marriage payment",
    "عدت": "iddah waiting period divorce",
    "حکومت": "government federal government state",
    "پارلیمنٹ": "parliament majlis-e-shoora national assembly",
    "اسمبلی": "assembly legislature provincial assembly",
    "سینیٹ": "senate upper house parliament",
    "ووٹ": "vote election franchise", "انتخابات": "election electoral franchise",
    "وزیراعظم": "prime minister chief executive",
    "وزیراعلی": "chief minister province", "گورنر": "governor province",
    "صدر": "president head of state",
    "جرم": "offence crime criminal", "سزا": "punishment sentence penalty",
    "الزام": "charge accusation allegation", "مجرم": "criminal accused convict",
    "بے گناہ": "innocent not guilty acquittal",
    "رشوت": "bribery corruption", "فراڈ": "fraud deceit",
    "قتل": "murder homicide", "چوری": "theft larceny",
    "ملازمت": "employment service job", "ملازم": "employee servant service",
    "تنخواہ": "salary remuneration pay", "پنشن": "pension retirement benefit",
    "ٹیکس": "tax levy duty",
    "تعلیم": "education right to education",
    "آئین": "constitution constitutional", "آرٹیکل": "article provision constitutional",
    "دفعہ": "section provision law",
}


def translate_urdu_script_query(query: str) -> str:
    english_expansions = []
    matched_positions = set()
    sorted_mappings = sorted(URDU_SCRIPT_TO_ENGLISH.items(), key=lambda x: len(x[0]), reverse=True)
    for urdu_phrase, english_eq in sorted_mappings:
        idx = query.find(urdu_phrase)
        if idx != -1:
            positions = set(range(idx, idx + len(urdu_phrase)))
            if not positions.intersection(matched_positions):
                matched_positions.update(positions)
                english_expansions.append(english_eq)
    if english_expansions:
        expanded = query + " " + " ".join(english_expansions)
        print(f"[TRANSLATE] Urdu script expansion: {' | '.join(english_expansions[:6])}")
        return expanded
    return query


def translate_roman_urdu_query(query: str) -> str:
    q_lower = query.lower()
    english_expansions = []
    sorted_mappings = sorted(ROMAN_URDU_TO_ENGLISH.items(), key=lambda x: len(x[0]), reverse=True)
    matched_positions = set()
    for roman_phrase, english_eq in sorted_mappings:
        idx = q_lower.find(roman_phrase)
        if idx != -1:
            positions = set(range(idx, idx + len(roman_phrase)))
            if not positions.intersection(matched_positions):
                matched_positions.update(positions)
                english_expansions.append(english_eq)
    if english_expansions:
        expanded = query + " " + " ".join(english_expansions)
        print(f"[TRANSLATE] Roman Urdu expansion: {' | '.join(english_expansions[:6])}")
        return expanded
    return query


# ═══════════════════════════════════════════════════════════
# LEGAL SYNONYM QUERY EXPANSION (all languages)
# ═══════════════════════════════════════════════════════════

LEGAL_SYNONYMS = {
    "land":           ["property", "immovable property", "acquisition", "article 24"],
    "compensation":   ["payment", "compulsory acquisition", "indemnity"],
    "arrest":         ["detention", "custody", "safeguards", "article 10"],
    "detention":      ["arrest", "custody", "preventive detention", "article 10"],
    "hours":          ["twenty-four hours", "24 hours", "period", "produced before magistrate"],
    "magistrate":     ["produced before magistrate", "24 hours", "article 10", "custody"],
    "produced":       ["magistrate", "24 hours", "arrest", "article 10"],
    "freedom":        ["fundamental rights", "liberty"],
    "acquire":        ["compulsory acquisition", "take possession"],
    "parliament":     ["majlis-e-shoora", "national assembly", "senate"],
    "court":          ["judicature", "high court", "supreme court"],
    "equality":       ["equal protection", "non-discrimination", "article 25"],
    "education":      ["right to education", "article 25a", "free compulsory"],
    "religion":       ["freedom of religion", "article 20"],
    "speech":         ["freedom of speech", "article 19"],
    "fair trial":     ["article 10a", "due process", "right to fair trial"],
    "bail":           ["bail", "release", "detention", "article 10"],
    "inheritance":    ["succession", "legal heirs", "estate", "property"],
    "divorce":        ["dissolution of marriage", "family law", "matrimonial"],
    "marriage":       ["nikah", "matrimonial", "family law"],
    "property":       ["immovable property", "acquisition", "article 24"],
    "punishment":     ["sentence", "penalty", "offence", "criminal"],
    "election":       ["electoral", "franchise", "voting rights", "article 51"],
    "president":      ["head of state", "article 41", "article 48"],
    "prime minister": ["chief executive", "article 91", "cabinet"],
    "environment":    ["clean environment", "article 9A", "sustainable"],
    "torture":        ["dignity", "article 14", "no torture"],
    "slavery":        ["forced labour", "article 11", "prohibited"],
    "assembly":       ["freedom of assembly", "article 16", "peaceful"],
    "association":    ["freedom of association", "article 17", "political party"],
    "information":    ["right to information", "article 19A", "public importance"],
    "trade":          ["freedom of trade", "article 18", "business profession"],
    "movement":       ["freedom of movement", "article 15", "reside settle"],
    "discrimination": ["non-discrimination", "article 25", "article 26", "article 27"],
    "language":       ["national language", "article 251", "urdu", "provincial language"],
    "emergency":      ["proclamation of emergency", "article 232", "article 233"],
    "amendment":      ["constitution amendment", "article 238", "article 239"],
}


def expand_query(query: str) -> str:
    expanded = query
    q_lower  = query.lower()
    added = set()
    for term, synonyms in LEGAL_SYNONYMS.items():
        if term in q_lower and term not in added:
            expanded += " " + " ".join(synonyms[:2])
            added.add(term)
    art_match = re.search(r'article\s+(\d+[A-Za-z]*)', query, re.IGNORECASE)
    if art_match:
        expanded += f" Article {art_match.group(1)} constitution Pakistan law provision"
    return expanded
