"""
Language detection: Urdu script (Unicode range), Roman Urdu (custom
keyword classifier), and a langdetect fallback for everything else.
"""
import re
from langdetect import detect

# ═══════════════════════════════════════════════════════════
# ROMAN URDU KEYWORD SET (200+ terms)
# ═══════════════════════════════════════════════════════════

ROMAN_URDU_KEYWORDS = {
    "mujhe","mujh","mein","mai","main","hum","aap","ap","tum","woh","yeh","ye",
    "is","us","unka","unke","unki","apna","apne","apni","tera","teri","mera","meri",
    "hamara","hamare","hamari","inki","inke","inka","unka",
    "hai","hain","tha","thi","the","hoga","hogi","honge","hote","hoti","hota",
    "karo","karna","karta","karti","karte","kar","kiya","ki","karo","karen",
    "ho","hua","hui","hue","ja","jao","jana","gaya","gayi","gaye",
    "milta","milti","milte","mile","batao","bata","puchna","puchho","pucho",
    "chahiye","chahta","chahti","chahte","sakta","sakti","sakte","sakein",
    "lagta","lagti","lagte","lena","dena","deta","deti","dete","lete","leti",
    "raha","rahi","rahe","rakha","rakhi","rakhe","rakhna",
    "aana","aao","aaye","aaya","aayi","ata","aati","aate",
    "padhna","likhna","samajhna","samjhao","samjhiye","bataiye",
    "poochna","poochho","maango","maangna","lena","lene",
    "dekhna","dekho","suno","bolna","bolo","boliye","kehna","kaho",
    "kya","kyun","kaise","kab","kahan","kaun","kon","kitna","kitne","kitni",
    "kuch","koi","sab","sirf","hi","bhi","to","phir",
    "qanoon","adalat","haq","huqooq","zamin","zameen","mulk","desh",
    "sarkar","hakumat","police","arrest","giraftari","muqadma","maamla",
    "waqil","advocate","judge","faisla","saza","jaidad","maal","milkiyat",
    "talaq","nikah","shadi","warasat","wirasat","merath","wirsa",
    "constitution","parliament","assembly","vote","intikhaab",
    "ilzam","jurm","gunah","mutaghazzi","mujrim","be-gunah",
    "bail","zamanat","remand","hirasaat","qaid","rehayi",
    "nafaqa","alaiment","hirasat","wardship","custody",
    "zulm","insaf","mazalim","shikayat","darkhwast","iltimas",
    "appeal","sunwai","peshi","pesh","samaan","saboot","gawah",
    "kharidar","bechne","kiraya","ijara","mukaan","ghar","plot",
    "rishwat","corruption","faraib","dhoka","cheating",
    "accident","haadsa","zakhmi","nuksan","muawza",
    "naukri","mulazim","tankhwa","salary","pension","service",
    "tax","mehsool","jagir","malikana","ownership",
    "firqa","mazhab","religion","mosque","masjid","church","mandir",
    "azadi","khawateen","bachay","bache","buzurg","beemar",
    "election","naib","nazim","councillor","MPA","MNA","PM","CM",
    "ghante","ghanta","minute","din","raat","waqt","muddat","arsa",
    "baad","pehle","jald","jaldi","abhi","foran","turant","jab","jab tak",
    "kitni","kitne","muddat","mein","tak","se",
    "aur","ya","lekin","magar","phir","bhi","hi","to","par","pe",
    "ke","ka","se","tak","wala","wali","wale","nahi","nahin","mat","na",
    "bilkul","zaroor","zaruri","lazim","wajib","jaiz","najaiz",
    "theek","sahi","galat","durust","ghair","illegal","legal",
    "zyada","kam","bohat","thoda","kafi","poora","aadha",
    "matlab","yani","yaani","iska","uska","matlb","yaane",
    "batao","samjhao","bataiye","samjhaiye","bata","samjha",
    "please","meherbani","kripya","shukria","shukriya",
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
