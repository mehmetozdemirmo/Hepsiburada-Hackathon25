import pandas as pd
import re
import sys
import unicodedata

# -------------------- Safe CSV Reader -------------------- #
def read_csv_safe(filepath, sep=',', encoding='utf-8'):
    """
    Safely read a CSV file.
    - Tries to read normally first.
    - If a parsing error occurs, retries while skipping bad lines.
    """
    try:
        df = pd.read_csv(filepath, sep=sep, encoding=encoding)
        print(f"File successfully read: {filepath}", flush=True)
        return df
    except pd.errors.ParserError as e:
        print(f"Parsing error detected: {e}", flush=True)
        print(f"Retrying by skipping problematic lines: {filepath}", flush=True)
        try:
            df = pd.read_csv(filepath, sep=sep, encoding=encoding, on_bad_lines='skip')
            print(f"Bad lines skipped. File successfully read: {filepath}", flush=True)
            return df
        except Exception as e:
            print(f"Failed to read file: {filepath}", flush=True)
            print(f"Error details: {e}", flush=True)
            sys.exit(1)

# -------------------- Province Name Mapping -------------------- #
IL_MAPPING = {
    "ist.": "istanbul",
    "izm.": "izmir",
    "ank.": "ankara",
    "adana": "adana",
    "ist": "istanbul",
    "mug": "muğla",
    "mugla": "muğla",
    "istanbl": "istanbul",
    "istanbol": "istanbul",
    "izm.r": "izmir",
    "izmir.": "izmir",
    "ankra": "ankara",
    "ankar.a": "ankara",
    "sanliurfa": "şanlıurfa",
    "sanlurfa": "şanlıurfa",
    "urfa": "şanlıurfa",
    "sanliurfa.": "şanlıurfa",
    "marmara": "marmaris",
    "marmariis": "marmaris",
    "antalya.": "antalya",
    "antlaya": "antalya",
    "antlya": "antalya",
    "eskshehir": "eskişehir",
    "eskişehır": "eskişehir",
    "kocaeli.": "kocaeli",
    "kocaelii": "kocaeli",
    "erzurum.": "erzurum",
    "erzrum": "erzurum",
    "trabzon.": "trabzon",
    "trazbon": "trabzon",
    "mersin.": "mersin",
    "mrsn": "mersin",
    "aydın.": "aydın",
    "aydin": "aydın",
    "konya.": "konya",
    "konyya": "konya",
    "bursa.": "bursa",
    "brsa": "bursa",
    "balıkesir.": "balıkesir",
    "balikesir": "balıkesir",
    "adiyaman": "adıyaman",
    "afyon": "afyonkarahisar",
    "agri": "ağrı",
    "amasya": "amasya",
    "artvin": "artvin",
    "bilecik": "bilecik",
    "bingol": "bingöl",
    "bitlis": "bitlis",
    "bolu": "bolu",
    "burdur": "burdur",
    "canakkale": "çanakkale",
    "cankiri": "çankırı",
    "corum": "çorum",
    "denizli": "denizli",
    "diyarbakir": "diyarbakır",
    "edirne": "edirne",
    "elazig": "elazığ",
    "erzincan": "erzincan",
    "gaziantep": "gaziantep",
    "giresun": "giresun",
    "gumushane": "gümüşhane",
    "hakkari": "hakkari",
    "hatay": "hatay",
    "isparta": "ısparta",
    "kars": "kars",
    "kastamonu": "kastamonu",
    "kayseri": "kayseri",
    "kirklareli": "kırklareli",
    "kirsehir": "kırşehir",
    "kutahya": "kütahya",
    "malatya": "malatya",
    "manisa": "manisa",
    "kahramanmaras": "kahramanmaraş",
    "maras": "kahramanmaraş",
    "mardin": "mardin",
    "mus": "muş",
    "nevsehir": "nevşehir",
    "nigde": "niğde",
    "ordu": "ordu",
    "rize": "rize",
    "sakarya": "sakarya",
    "samsun": "samsun",
    "siirt": "siirt",
    "sinop": "sinop",
    "sivas": "sivas",
    "tekirdag": "tekirdağ",
    "tokat": "tokat",
    "tunceli": "tunceli",
    "usak": "uşak",
    "van": "van",
    "yozgat": "yozgat",
    "zonguldak": "zonguldak",
    "aksaray": "aksaray",
    "bayburt": "bayburt",
    "karaman": "karaman",
    "kirikkale": "kırıkkale",
    "batman": "batman",
    "sirnak": "şırnak",
    "bartin": "bartın",
    "ardahan": "ardahan",
    "igdir": "ığdır",
    "yalova": "yalova",
    "karabuk": "karabük",
    "kilis": "kilis",
    "osmaniye": "osmaniye",
    "duzce": "düzce",
    "eskisehir": "eskişehir",
    "cankırı": "çankırı",
    "corum": "çorum",
    "elazığ": "elazığ",
    "bingöl": "bingöl",
    "erzincan.": "erzincan",
    "erzrum.": "erzurum",
    "gazantep": "gaziantep",
    "giresn": "giresun",
    "gumushane.": "gümüşhane",
    "hakkari.": "hakkari",
    "hatay.": "hatay",
    "isparta.": "ısparta",
    "kars.": "kars",
    "kayseri.": "kayseri",
    "kirklareli.": "kırklareli",
    "kirsehir.": "kırşehir",
    "konya": "konya",
    "kutahya": "kütahya",
    "malatya.": "malatya",
    "manisa.": "manisa",
    "maras.": "kahramanmaraş",
    "mardin.": "mardin",
    "mus.": "muş",
    "nevsehir.": "nevşehir",
    "nigde.": "niğde",
    "ordu.": "ordu",
    "rize.": "rize",
    "sakarya.": "sakarya",
    "samsun.": "samsun",
    "siirt.": "siirt",
    "sinop.": "sinop",
    "sivas.": "sivas",
    "tekirdag.": "tekirdağ",
    "tokat.": "tokat",
    "trabzon.": "trabzon",
    "tunceli.": "tunceli",
    "usak.": "uşak",
    "van.": "van",
    "yozgat.": "yozgat",
    "zonguldak.": "zonguldak",
    "aksaray.": "aksaray",
    "bayburt.": "bayburt",
    "karaman.": "karaman",
    "kirikkale.": "kırıkkale",
    "batman.": "batman",
    "sirnak.": "şırnak",
    "bartin.": "bartın",
    "ardahan.": "ardahan",
    "igdir.": "ığdır",
    "yalova.": "yalova",
    "karabuk.": "karabük",
    "kilis.": "kilis",
    "osmaniye.": "osmaniye",
    "duzce.": "düzce"
}

# -------------------- Common Abbreviation Mapping -------------------- #
ABBR_MAPPING = {
    r'\b(?:mh|mah|mahallesi|mahall|mahal)\b': 'mahalle',
    r'\b(?:cd|cad|caddesi)\b': 'cadde',
    r'\b(?:blv|bulvarı|bulv|bulvari)\b': 'bulvar',
    r'\b(?:sk|sok|sokağı|sokakı|sokagi)\b': 'sokak',
    r'\b(?:no|nr|nu|numarası)\b': 'numara',
    r'\b(?:dr|dair|daire|dairede)\b': 'daire',
    r'\b(?:apt|ap|apartmanı|apart|apartman)\b': 'apartman',
    r'\b(?:blk|bl|bloku|blok)\b': 'blok',
    r'\b(?:kt|kat|k)\b': 'kat',
    r'\b(?:sit|sitesi|site)\b': 'site',
    r'\b(?:evler)\b': 'evler',
    r'\b(?:mey|myd|meydanı)\b': 'meydan',
    r'\b(?:ilçesi|ilce)\b': 'ilçe',
    r'\b(?:semt|semti)\b': 'semt',
    r'\b(?:koy|köyü)\b': 'köy',
    r'\b(?:mevkii|mevki)\b': 'mevkii',
    r'\b(?:yayla)\b': 'yayla',
    r'\b(?:parsel)\b': 'parsel',
    r'\b(?:pafta)\b': 'pafta',
    r'\b(?:ada)\b': 'ada'
}

# Manual character corrections for special unicode cases
char_fixes = {
    'ė': 'e',
    'i̇': 'i',
    'â': 'a',
    'ö': 'ö',
    'ş': 'ş',
    'ç': 'ç',
    'ü': 'ü',
}

# -------------------- Address Normalization -------------------- #
def normalize_address(address):
    """
    Normalize address text by:
    - Lowercasing
    - Fixing unicode characters
    - Removing punctuation
    - Expanding province names and common abbreviations
    """
    if pd.isna(address):
        return ''
    addr = str(address).lower().strip()

    # Fix special unicode characters
    for wrong, correct in char_fixes.items():
        addr = addr.replace(wrong, correct)

    # Remove punctuation
    addr = re.sub(r"[.,;:/\-()]", " ", addr)

    # Replace multiple spaces with single space
    addr = re.sub(r"\s+", " ", addr)

    # Normalize province names
    for short, full in IL_MAPPING.items():
        addr = re.sub(r"\b{}\b".format(re.escape(short)), full, addr)

    # Expand common abbreviations
    for pattern, full in ABBR_MAPPING.items():
        addr = re.sub(pattern, full, addr)

    return addr.strip()

# -------------------- Read Datasets -------------------- #
try:
    train_df = read_csv_safe("train_sorted.csv", sep=',', encoding='utf-8')
    print("'train.csv' successfully loaded.", flush=True)
except FileNotFoundError:
    print("Error: 'train.csv' not found. Please check the file path.", flush=True)
    sys.exit(1)

try:
    test_df = read_csv_safe("test_normalized.csv", sep=',')
    print("'test.csv' successfully loaded.", flush=True)
except FileNotFoundError:
    print("Error: 'test.csv' not found. Please check the file path.", flush=True)
    sys.exit(1)

# -------------------- Apply Normalization -------------------- #
print("Processing train dataset...")
train_df['address'] = train_df['address_normalized'].apply(normalize_address)
print("Train dataset normalized.\nProcessing test dataset...")
test_df['address'] = test_df['address_normalized'].apply(normalize_address)
print("Test dataset normalized.")

# -------------------- Save Results -------------------- #
train_cleaned = train_df[['label', 'address_normalized']]
test_cleaned = test_df[['id', 'address_normalized']]

test_cleaned.to_csv("test_normalized.csv", index=False)
train_cleaned.to_csv("train_normalized.csv", index=False)
print("train_sorted.csv created and saved according to label order.")
