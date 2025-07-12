import warnings
warnings.filterwarnings("ignore")
import re

def adjust_spacing(text):
    text = re.sub(r' {2,}', '\x01', text)
    text = re.sub(r' ', '\x02', text)
    text = re.sub(r'\x01', ' ', text)
    text = re.sub(r'\x02', '', text)

    return text

def get_next_alphabet(seen_alphabets):
    if len(seen_alphabets) < 1:
        return 'A'

    for i in range(26):
        char = chr(65 + i)  # A-Z
        if char not in seen_alphabets:
            return char
    for i in range(26):
        for j in range(26):

            if i == j:
                continue

            char = chr(65 + i) + chr(65 + j)  # AA-ZZ
            if char not in seen_alphabets and char not in FIXED_PROTECTION:
                return char

def clean_repeated_alphabets(text):
    text = re.sub(r'([A-Z]{2})\1+', r' \1', text)
    text = re.sub(r'([A-Z])\1+', r' \1', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def normalize_address_names(text, phase="input"):
    metropolitan = {
        "서울특별시 ":"서울시 ",
        "인천광역시 ":"인천시 ",
        "부산광역시 ":"부산시 ",
        "대구광역시 ":"대구시 ",
        "대전광역시 ":"대전시 ",
        "울산광역시 ":"울산시 ",
        "광주광역시 ":"광주시 ",
        "세종특별자치시 ":"세종시 ",
    }
    province = {
        "경기도": "경기",
        "강원도": "강원",
        "충청북도": "충북",
        "충청남도": "충남",
        "경상북도": "경북",
        "경상남도": "경남",
        "전라북도": "전북",
        "전라남도": "전남",
        "제주도": "제주",
    }
    
    if phase == "input":
        for replace_from, replace_to in metropolitan.items():
            text = text.replace(replace_from, replace_to)   

        for replace_from, replace_to in province.items():
            pattern = rf"\b{replace_from}(?=\s+[가-힣]+시\b|\s+[가-힣]+군\b)"
            text = re.sub(pattern, replace_to, text)
    
    elif phase == "output":
        for replace_to, replace_from in metropolitan.items():
            text = text.replace(replace_from, replace_to)

        for replace_to, replace_from in province.items():
            pattern = rf"\b{replace_from}(?=\s+[가-힣]+시\b|\s+[가-힣]+군\b)"
            text = re.sub(pattern, replace_to, text)
            
    return text


def adjust_alphabet_spacing(text):
    text = re.sub(r"(?<=\S)(?=['\"‘\“][A-Za-z])", " ", text)
    text = re.sub(r"([\'\"\‘\“\(\[\{])\s+([A-Z]{1,2})", r"\1\2", text)
    text = re.sub(r"([A-Z]{1,2})\s+([\'\"\’\”\)\]\}])", r"\1\2", text)
    
    chunk = r"['\"\‘\“\(\[\{][A-Z]{1,2}['\"\’\”\)\]\}]"
    text = re.sub(rf"(?<=\S)(?={chunk})", " ", text)
    text = re.sub(rf"({chunk})(?=\S)", r"\1 ", text)
    
    text = re.sub(r"('.*?')(?=\S)", r"\1 ", text)
    text = re.sub(r"([가-힣0-9\)\]\}])(?=[A-Z]{1,2})", r"\1 ", text)
    text = re.sub(r"(?<=[A-Z])(?=[0-9가-힣])", " ", text)
    text = re.sub(r"(?<=[0-9가-힣])(?=[A-Z])", " ", text)

    def fix_missing_open_paren(match):
        full_match = match.group(0) 
        prefix = match.group(1)  
        alphabet = match.group(2)
        start_idx = max(0, match.start() - 2)
        preceding_text = text[start_idx:match.start()]
        if '(' not in preceding_text:
            return f"{prefix}({alphabet})"

        return full_match
    text = re.sub(r"([가-힣0-9]+\s+)([A-Z])\)", fix_missing_open_paren, text)
    
    text = re.sub(r"(\))\s+([A-Z가-힣0-9])", r"\1\2", text)
    text = re.sub(r"([A-Z가-힣0-9])\s+(\()", r"\1\2", text)

    text = re.sub(r"([A-Z]{1,2})([\'\"\’\”])\s+(?=(?:에|에서|에게|으로|로|은|는|이|가|을|를|과|와)\b)",
    r"\1\2", text)
    text = re.sub(r"([A-Z]{1,2})\s+(?=(?:에|에서|에게|으로|로|은|는|이|가|을|를|과|와)\b)", r"\1", text)
    
    return text


def protect_word(text, protected, cnt=0):
    # protection
    for p in FIXED_PROTECTION:
        key = f"@#PROTECTION_{cnt}#@"
        protected[key] = p
        cnt += 1
    
    # brands (dynamic)
    word_pattern = r'\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b'
    matches = list(re.finditer(word_pattern, text))
    for match in reversed(matches):
        word = match.group(0)
        key = f"@#PROTECTION_{cnt}#@"
        protected[key] = word
        cnt += 1

    return protected, cnt


def restore_word(text, protected, entity_dict):
    # protected
    for marker, origin in protected.items():
        if marker in text:
            text = text.replace(marker, origin)
                
    pattern = r'@.+?@'
    matches = re.findall(pattern, text)

    # matched
    for m in matches:
        modified = m.replace("@", '').replace('#', '')
        modified = modified.replace("PROTECTED", '').replace('_', '')
        modified = modified.rstrip().lstrip()
        
        alphabet = re.sub(r'[^A-Z]', '', modified)
        text = text.replace(m, alphabet)

    return text


def fix_josa(text):
    batchim = set("LMNR")
    
    # mapping
    no2yes  = {"을":"를","은":"는","이":"가","과":"와","이라는":"라는"}
    yes2no  = {v:k for k,v in no2yes.items()}

    josas = sorted(list(no2yes.keys()) + list(yes2no.keys()), key=len, reverse=True)
    pat = re.compile(
        r"([A-Z]{1,2}(?:\([^)]*\))?)"         
        r"(" + "|".join(map(re.escape, josas)) + r")"
    )
    def repl(m):
        ent, j = m.group(1), m.group(2)
        if ent[-1] in batchim:
            return ent + yes2no.get(j, j)
        else:
            return ent + no2yes.get(j, j)
    return pat.sub(repl, text)


def deidentify(model, tokenizer, model_inputs, model_outputs, protected):

    deidentified = []
    entity_dicts = []
    
    all_seen_alphabets = set()

    for document_idx in range(len(model_inputs)):
        seen_alphabets = set()
        entity_begins = False

        cur_document = [tok for tok in model_inputs[document_idx]]
        entity_dict = dict()
        current_entity_tok = []
        current_entity_pos = []
            
        for pos_idx, (tok, pred_tok) in enumerate(zip(model_inputs[document_idx], model_outputs[document_idx])):
        
            if pred_tok == 1:
                entity_begins = False
                if len(current_entity_tok) > 0:
                    cur_entity = adjust_spacing(tokenizer.decode(current_entity_tok)).strip()

                    # added: 25.07.13
                    if (cur_entity in protected): 
                        if (protected[cur_entity] in FIXED_PROTECTION):
                            current_entity_tok = []
                            current_entity_pos = []
                            continue

                    for key in entity_dict.keys():
                        if entity_dict[key]["entity"] == cur_entity:
                            for pos in current_entity_pos:
                                cur_document[pos] = tokenizer.convert_tokens_to_ids(key)
                            break
                    else:
                        alphabet = get_next_alphabet(seen_alphabets)
                        seen_alphabets.add(alphabet)
                        all_seen_alphabets.add(alphabet)
                        
                        next_width = 15
                        next_tok = cur_document[pos_idx:pos_idx+next_width] if pos_idx + next_width < len(cur_document)-1 else cur_document[pos_idx:]                            
                        next_chars = adjust_spacing(tokenizer.decode(next_tok))
                        
                        entity_dict[alphabet] = {"entity": cur_entity, "next_chars":next_chars, "josa_adjusted":False}
                        for pos in current_entity_pos:
                            cur_document[pos] = tokenizer.convert_tokens_to_ids(alphabet)
                    current_entity_tok = []
                    current_entity_pos = []
            else:
                if not entity_begins:
                    entity_begins = True
                    current_entity_tok = []
                    current_entity_pos = []

                current_entity_tok.append(tok)
                current_entity_pos.append(pos_idx)

        deidentified.append(cur_document)
        entity_dicts.append(entity_dict)

    return deidentified, entity_dicts



FIXED_PROTECTION = [
    "CCTV",
    "TV",
    "KTX",
    "SRT",
    "GPS",
    "ATM",
    "CPU",
    "GPU",
    "SSD",
    "USB",
    "PDF",
    "HTML",
    "HTTP",
    "URL",
    "API",
    "AI",
    "DNA",
    "HIV",
    "IT",
    "NGO",
    
    "신개념",
    "코스닥",
    "코스피",
    "암호화폐",
    "신원미상",
    "임원진",
    "혈중알코올농도",
    
    "이메일",
    "주민등록번호",
    "차량번호",
    "계좌번호",
    
]

