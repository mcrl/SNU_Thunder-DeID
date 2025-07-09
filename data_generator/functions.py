import os
import glob
import json
import re
import random
import copy
import difflib
import math
from tqdm import tqdm
from itertools import groupby
from functools import partial
import pickle
from joblib import Parallel, delayed, parallel_backend
import multiprocessing as mp

from transformers import AddedToken

# private
from . import generator
import custom


        
    

def preprocessing(raw, replace_map, filename="for_debugging"):
    def replace_tag_in_document(document, tag_from, tag_to):
        document = document.replace(f"<<<{tag_from}>>>", f"<<<{tag_to}>>>")
        document = document.replace(f"<<</{tag_from}>>>", f"<<</{tag_to}>>>")
        return document
    
    def get_next_alphabet(seen_alphabets):
        for i in range(26):
            char = chr(65 + i)  # A-Z
            if char not in seen_alphabets:
                # seen_alphabets.add(char)
                return char
        for i in range(26):
            for j in range(26):
                char = chr(65 + i) + chr(65 + j)  # AA-ZZ
                if char not in seen_alphabets:
                    # seen_alphabets.add(char)
                    return char
                
                
    # 0. remove invalid text
    processed = re.sub(r">>> \d+ 생략", ">>>", raw)
    processed = re.sub(r">>>\d+ 생략",  ">>>", processed)
    processed = processed.replace("<<<(", "<<<").replace("<<</(", "<<</")
    processed = processed.replace(")>>>", ">>>")
                
    
    # 1. find tags and alphabets (e.g., <<<구아래주소>>>A<<</구아래주소>>> -> [ ('구아래주소', 'A), ... ])
    tag_pattern = r'<<<(.*?)>>>(.*?)<<</\1>>>'
    tags_and_alphabets = re.findall(tag_pattern, processed)
    tags      = [item[0] for item in tags_and_alphabets]
    alphabets = [item[1] for item in tags_and_alphabets]
    seen_alphabets = set(alphabets)
        
    for tag_idx, (tag, alphabet) in enumerate(zip(tags, alphabets)):
        """ handling invalid alphabets """
        if tags[tag_idx] == alphabet:
            # case : <<<은행>>>은행<<</은행>>>
            # generate new alphabet
            new_alphabet = get_next_alphabet(seen_alphabets)
            seen_alphabets.add(new_alphabet)
            
            # update
            # MODIFIED : 태그 전체 넣어야 함. oooooo 로 동일하게 비식별화 처리 된 경우가 있어서
            processed = processed.replace(f">>>{alphabet}<<<", f">>>{new_alphabet}<<<")
            alphabets[tag_idx] = new_alphabet

        elif not re.fullmatch(r'[A-Za-z]+', alphabet): 
            # case : <<<모텔>>>E모텔<<</모텔>>>
            # generate new alphabet
            new_alphabet = get_next_alphabet(seen_alphabets)
            seen_alphabets.add(new_alphabet)
            
            # update
            processed = processed.replace(f">>>{alphabet}<<<", f">>>{new_alphabet}<<<")
            alphabets[tag_idx] = new_alphabet
        
        
        """ handling invalid tags """
        # (1) 이하주소 -> 아래주소
        if "이하주소" in tags[tag_idx]:
            # update
            new_tag = tags[tag_idx].replace("이하주소", "아래주소")
            processed = replace_tag_in_document(processed,
                                    tag_from=tags[tag_idx],
                                    tag_to=new_tag)
            tags[tag_idx] = new_tag
        elif "아래이름" in tags[tag_idx]:
            # update
            new_tag = tags[tag_idx].replace("아래이름", "아래주소")
            processed = replace_tag_in_document(processed,
                                    tag_from=tags[tag_idx],
                                    tag_to=new_tag)
            tags[tag_idx] = new_tag
            
        # (2) 구아래주소빌딩 -> 빌딩
        if "아래주소" in tags[tag_idx] :
            front, rear = tags[tag_idx].split("아래주소")
            if len(rear) > 0:
                new_tag = rear
                processed = replace_tag_in_document(processed,
                                                    tag_from=tags[tag_idx],
                                                    tag_to=new_tag)
                tags[tag_idx] = new_tag
            
            
        # (3) replace_map.json 의 value로 치환 
        if tags[tag_idx] in replace_map.keys():
            # update
            new_tag = replace_map[tags[tag_idx]]
            processed = replace_tag_in_document(processed,
                                                tag_from=tags[tag_idx],
                                                tag_to=new_tag)
            tags[tag_idx] = new_tag
            
            
            
        # (5) <<</행정동>>>동 -> 삼산동동 등 문제 발생
        if tags[tag_idx] in ["행정구","행정군","행정동","행정시","행정읍"]:
            level = tag[-1]
            processed = processed.replace(f"<<</{tags[tag_idx]}>>>{level}", f"<<</{tags[tag_idx]}>>>")
    
    
    # 일부 tag 의 경우 replace 필요 (천둥주식회사 주식회사, 서울대학교 대학교 등)
    for tag, alphabet in zip(tags, alphabets):
        # 개별처리
        if tag in ["공사현장"]:
            replace_from = f"<<<공사현장>>>{alphabet}<<</공사현장>>>현장"
            replace_to   = f"<<<공사현장>>>{alphabet}<<</공사현장>>>"
            processed = processed.replace(replace_from, replace_to)
        
            
        elif tag in ["주식회사", 
                     "사무실", "사무소", 
                     "공인중개사",
                     "대학교", "고등학교", "중학교", "초등학교",
                     "조성사업",
                     
                     
                     "터미널",
                     
                     "은행",
                     ]:
            # (1) <<<{tag}>>>{alphabet}<<</{tag}>>>{tag}
            replace_from = f"<<<{tag}>>>{alphabet}<<</{tag}>>>{tag}"
            replace_to   = f"<<<{tag}>>>{alphabet}<<</{tag}>>>"
            processed = processed.replace(replace_from, replace_to)
            
            replace_from = f"<<<{tag}>>>{alphabet}<<</{tag}>>> {tag}"
            replace_to   = f"<<<{tag}>>>{alphabet}<<</{tag}>>>"
            processed = processed.replace(replace_from, replace_to)
            
            replace_from = f"{tag}<<<{tag}>>>{alphabet}<<</{tag}>>>"
            replace_to   = f"<<<{tag}>>>{alphabet}<<</{tag}>>>"
            processed = processed.replace(replace_from, replace_to)
            
            replace_from = f"{tag} <<<{tag}>>>{alphabet}<<</{tag}>>>"
            replace_to   = f"<<<{tag}>>>{alphabet}<<</{tag}>>>"
            processed = processed.replace(replace_from, replace_to)
        
        elif tag in ["지하철역", "역", "기차역", "KTX역"]:
            # added. 25.06.02
            replace_from = f"<<<{tag}>>>{alphabet}<<</{tag}>>>역"
            replace_to = f"<<<{tag}>>>{alphabet}<<</{tag}>>>"
            processed = processed.replace(replace_from, replace_to)
 
    
    # spacing
    processed = processed.replace("\n\n", "\n").lstrip().rstrip()
    
    return processed, tags, alphabets




# subset
def get_json_filenames(directory):
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                full_path = os.path.abspath(os.path.join(root, file))
                json_files.append(full_path)
    return json_files



def get_sub_categories(target, root_path="./1_replacement/subset"):
    """
    소분류(JSON 파일) 또는 중분류(디렉토리)에 속한 모든 키 반환
    Parameters:
        root_path (str): 루트 디렉토리 경로
        target (str): 소분류(JSON 파일명, 예: '비트코인개인지갑.json') 또는 중분류(디렉토리명, 예: '고유번호')
    Returns:
        list or str: 키 목록 (중복 제거, 정렬) 또는 에러 메시지
    """
    def get_keys_from_json(json_file_path):
        """
        JSON 파일에서 최상위 키 추출
        Parameters:
            json_file_path (str): JSON 파일 경로
        Returns:
            list: 최상위 키 목록, 오류 시 빈 리스트
        """
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                return list(data.keys())
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                return list(data[0].keys())
            return []
        
    all_keys = set()
    found = False
    is_json = target.endswith('.json')  # 소분류(JSON 파일)인지 확인

    for root, dirs, files in os.walk(root_path):
        # 중분류 확인 (디렉토리명 매칭)
        if not is_json and os.path.basename(root) == target:
            found = True
            for file_name in files:
                if file_name.endswith('.json'):
                    json_file_path = os.path.join(root, file_name)
                    keys = get_keys_from_json(json_file_path)
                    all_keys.update(keys)
        # 소분류 확인 (JSON 파일명 매칭)
        elif is_json and target in files:
            found = True
            json_file_path = os.path.join(root, target)
            keys = get_keys_from_json(json_file_path)
            all_keys.update(keys)
            break  # JSON 파일 찾았으므로 탐색 중단

    if found and all_keys:
        return sorted(list(all_keys))
    
    else:
        return None



# ================================ help functions ===================================

def adjust_spacing(text):
    text = re.sub(r' {2,}', '\x01', text)
    text = re.sub(r' ', '\x02', text)
    text = re.sub(r'\x01', ' ', text)
    text = re.sub(r'\x02', '', text)
    return text


def has_leading_whitespace(processed_text, replace_from_position):
    """
    replace_from_position 바로 앞 문자가 whitespace 인지 확인
    """
    if replace_from_position == 0:
        # 문장 맨 처음 → space 없음
        return False
    elif replace_from_position == -1:
        return False
    else:
        prev_char = processed_text[replace_from_position - 1]
        return prev_char.isspace()


def has_trailing_whitespace(processed_text, replace_from_position):
    """
    replace_from_position 바로 뒤 문자가 whitespace 인지 확인
    """
    if replace_from_position + 1 >= len(processed_text):
        return False
    if replace_from_position == -1:
        return False
    else:
        next_char = processed_text[replace_from_position + 1]
        return next_char.isspace()

def get_trailing_josa(processed_text, replace_from_position, replace_from_length):
    """
    replace_from 뒤에 조사(tokenization에 영향 주는지) 여부 확인
    """
    after_idx = replace_from_position + replace_from_length
    if after_idx >= len(processed_text):
        return ""  # 문장 끝
    else:
        # 조사 후보 추출 (최대 2~3자까지 보는게 일반적)
        following_text = processed_text[after_idx:after_idx+3]
        
        # 조사 리스트 (필요시 확장 가능)
        josa_list = ['은', '는', '이', '가', '을', '를', '에', 
        '에서', '에게', '한테', '으로', '로', '와', '과', '도', 
        '만', '보다', '까지', '부터', '든지', '조차', '마저']
        
        for josa in sorted(josa_list, key=lambda x: -len(x)):  # 긴 조사부터 먼저 체크
            if following_text.startswith(josa):
                return josa
        return ""

def get_trailing_text(processed_text, replace_from_position, replace_from_length):
    """
    replace_from 뒤에 붙어있는 텍스트 (띄어쓰기 기준으로 다음 어절)를 추출
    """
    after_idx = replace_from_position + replace_from_length
    if after_idx >= len(processed_text):
        return ""  # 문장 끝
    
    # after_idx 에서부터 끝까지 남은 부분
    remaining_text = processed_text[after_idx:]
    
    # 앞쪽 공백 제거 (혹시 공백이 있는 경우)
    remaining_text = remaining_text.lstrip()
    
    # 첫 번째 띄어쓰기 위치 찾기
    space_pos = remaining_text.find(" ")
    
    if space_pos == -1:
        # 띄어쓰기 없음 → 남은 텍스트 전체가 붙어있는 단어
        return remaining_text
    else:
        # 첫 단어 (띄어쓰기 전까지)
        return remaining_text[:space_pos]



def entity_tokens_replacement(item, X, Y, tokenizer):
    # item
    replace_from_tokenized = item["replace_from_tokenized"]
    replace_to_tokenized = item["replace_to_tokenized"]
    leading_whitespace = item["leading_whitespace"]
    trailing_josa = item["trailing_josa"]
    label = item["label"]
    encoding_target = item["encoding_target"]
    
    # return 
    new_X = []
    new_Y = []
    
    # insert tokens / generate label sequence
    len_repl_from = len(replace_from_tokenized)
    prev_common_start_idx = 0
    # increase 1 !
    for idx in range(0, len(X)-len_repl_from):
        if X[idx:idx+len_repl_from] == replace_from_tokenized:

            # common part
            if leading_whitespace:
                end_idx = idx-1
            else:
                end_idx = idx
            # X
            new_X += X[prev_common_start_idx:end_idx] 
            # Y
            new_Y += Y[prev_common_start_idx:end_idx]
            
            
            # entity part
            if trailing_josa == '':
                entity_tokens = replace_to_tokenized
                new_Y += [label for _ in range(len(entity_tokens))]
                new_X += entity_tokens

            else:
                josa_tokens = tokenizer.encode(trailing_josa)
                josa_tokens_str = [str(tok) for tok in josa_tokens]
                replace_to_tokenized_str = [str(tok) for tok in replace_to_tokenized]
                if ','.join(josa_tokens_str) not in ','.join(replace_to_tokenized_str):
                    print()
                    print(replace_to_tokenized, tokenizer.decode(replace_to_tokenized))
                    print(josa_tokens, tokenizer.decode(josa_tokens))
                    exit()
                entity_tokens = replace_to_tokenized[:len(josa_tokens)]
                new_Y += [label for _ in range(len(entity_tokens))]
                # new_Y += ['O' for _ in range(len(josa_tokens))]
                
            # if "케이스타타워" in encoding_target:
            #     print(0)
            #     print(entity_tokens)
            #     print(item)

            # update "prev_common_start_idx"
            prev_common_start_idx = idx+len_repl_from
            
    new_X += X[prev_common_start_idx:]
    new_Y += Y[prev_common_start_idx:]

    if len(new_X) != len(new_Y):
        print()
        print(new_X)
        print(new_Y)
        print()
        for x, y in zip(new_X, new_Y):
            print(tokenizer.convert_ids_to_tokens(x), y)
        
        exit()
        
    return new_X, new_Y

    

def split_lists(l1, l2):
    matcher = difflib.SequenceMatcher(None, l1, l2)
    common_subs = []
    diff_subs_1 = []
    diff_subs_2 = []
    order = []

    last1 = 0
    last2 = 0
    count = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        count += 1
        if tag == 'equal':
            common_subs.append(l1[i1:i2])
            order.append(('common', len(common_subs)-1))
            last1 = i2
            last2 = j2
        else:
            # list1 diff
            if i1 != i2:
                diff_subs_1.append(l1[i1:i2])
                order.append(('diff1', len(diff_subs_1)-1))
            # list2 diff
            if j1 != j2:
                diff_subs_2.append(l2[j1:j2])
                order.append(('diff2', len(diff_subs_2)-1))
            last1 = i2
            last2 = j2

    return {
        'common_subs': common_subs,
        'diff_subs_1': diff_subs_1,
        'diff_subs_2': diff_subs_2,
        'order': order
    }



def split_lists(l1, l2, l1_labels=None):
    matcher = difflib.SequenceMatcher(None, l1, l2)
    common_subs = []
    common_labels = []   # 추가됨
    
    diff_subs_1 = []
    diff_labels_1 = []   # 추가됨
    
    diff_subs_2 = []
    order = []

    last1 = 0
    last2 = 0
    count = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        count += 1
        if tag == 'equal':
            common_subs.append(l1[i1:i2])
            if l1_labels is not None:
                common_labels.append(l1_labels[i1:i2])
            else:
                common_labels.append(None)
            order.append(('common', len(common_subs)-1))
            last1 = i2
            last2 = j2
        else:
            # list1 diff
            if i1 != i2:
                diff_subs_1.append(l1[i1:i2])
                if l1_labels is not None:
                    diff_labels_1.append(l1_labels[i1:i2])
                else:
                    diff_labels_1.append(None)
                order.append(('diff1', len(diff_subs_1)-1))
            # list2 diff
            if j1 != j2:
                diff_subs_2.append(l2[j1:j2])
                order.append(('diff2', len(diff_subs_2)-1))
            last1 = i2
            last2 = j2

    return {
        'common_subs': common_subs,
        'common_labels': common_labels,    # 추가됨
        'diff_subs_1': diff_subs_1,
        'diff_labels_1': diff_labels_1,    # 추가됨
        'diff_subs_2': diff_subs_2,
        'order': order
    }
# ================================ help functions ===================================








def generate_dataset_with_modified_tokenizer(
    # inputs
    tokenizer,
    seed=1203,
    num_replica=30,
    raw_dataset_dir="./datasets/0_raw_documents",
    replace_subset_dataset_dir="./datasets/1_replacement/subset",
    replace_addr_dataset_dir="./datasets/1_replacement/address",
    replace_map_path=f"./datasets/replace_map.json",
    
    # output
    dataset_path=f"./datasets/2_deid_dataset/.pkl",
    modified_tokenizer_path=f"./datasets/2_deid_dataset/.pkl",
    ):
        

    PROVINCE = [
        '서울특별시', 
        '인천광역시', '부산광역시', '대구광역시', 
        '대전광역시', '광주광역시', '울산광역시', 
        '세종특별자치시', 
        '경기도', 
        '강원도', '강원특별자치도', 
        '충청북도', '충청남도', 
        '경상북도', '경상남도', 
        '전라북도', '전북특별자치도', 
        '전라남도', 
        '제주특별자치도'
    ]
    random.seed(seed)    
    
    
    """ 1. raw documents """
    raw_documents = {
        "indecent_act_by_compulsion":sorted(glob.glob(f"{raw_dataset_dir}/indecent_act_by_compulsion/*.md")),
        "crime_of_violence":sorted(glob.glob(f"{raw_dataset_dir}/crime_of_violence/*.md")),
        "fraud":sorted(glob.glob(f"{raw_dataset_dir}/fraud/*.md")),
    }
    
    
    """ 2. lists for replacement """
    entity_mentions = dict()
    rep_subset_files = get_json_filenames(replace_subset_dataset_dir)
    for filename in rep_subset_files:
        with open(filename, 'r') as fr:
            data = json.load(fr)
        entity_mentions.update(data)
    
    # addr
    with open(f"{replace_addr_dataset_dir}/REPLACEMENT_LIST-addrs1.json", 'r') as fr1:
        data = json.load(fr1)
        data = {key:value for key, value in data.items()
                     if key in PROVINCE}
        entity_mentions["지번주소"] = data
        for province in PROVINCE:
            if province.endswith("시"):
                entity_mentions["지번주소"][province] = {k:v for k,v in entity_mentions["지번주소"][province].items()
                                              if k.endswith("구")
                                              or k.endswith("군")}
                
    with open(f"{replace_addr_dataset_dir}/REPLACEMENT_LIST-addrs2.json", 'r') as fr2:
        data = json.load(fr2)
        data = {key:value for key, value in data.items()
                     if key in PROVINCE}
        entity_mentions["도로명주소"] = data
        for province in PROVINCE:
            if province.endswith("시"):
                entity_mentions["도로명주소"][province] = {k:v for k,v in entity_mentions["도로명주소"][province].items()
                                              if k.endswith("구")
                                              or k.endswith("군")}
        
    """ 3. for processing (replace map) """
    with open(replace_map_path, 'r') as fr:
        replace_map = json.load(fr)

        
    """ samples """
    samples = list()
    
    """ preprocessing """            
    all_labels = set()
    for category in raw_documents.keys():
        
        for filename in tqdm(raw_documents[category], ncols=80):
        # for f_idx, filename in enumerate(raw_documents[category]):
            # read raw document
            with open(filename, 'r') as fr:
                raw = fr.read()
                                
            # preprocess raw document
            processed, labels, alphabets = preprocessing(
                raw, 
                replace_map=replace_map,
                filename=filename
            )
            replace_link = [(a, l) for a, l in zip(alphabets, labels)]
                            
            samples.append({
                "raw":raw,
                "processed":processed,
                "replace_link":replace_link,
                "filename":filename, # TODO : remove or modify this
                "category":category
            })
            for labels in labels:
                all_labels.add(labels)
    
    # add special tokens to tokenizer
    all_labels = sorted(all_labels)
    special_tokens = []
    special_tokens += [AddedToken(f"<<<{label}>>>", lstrip=False, rstrip=False, single_word=False, normalized=True, special=True) for label in all_labels]
    special_tokens += [AddedToken(f"<<</{label}>>>", lstrip=False, rstrip=False, single_word=False, normalized=True, special=True) for label in all_labels]
    
    tokenizer.add_special_tokens({"additional_special_tokens":special_tokens})
    special_token_set = set(tokenizer.additional_special_tokens)
    
    # save the modified tokenizer    
    tokenizer.save_pretrained(modified_tokenizer_path)
    
    # switch to mecab-bpe mode (can't be serialized)
    tokenizer = custom.switch_dummy(tokenizer)
    
    # generate train / validation dataset

    cnt = 0
    for sample_idx, sample in tqdm(enumerate(samples), total=len(samples), ncols=80):

        filename = sample["filename"]
        replace_link = sample["replace_link"]
        processed = sample["processed"]

        category = sample["category"]
        
        # n 개의 replaced replica 생성
        samples[sample_idx]["replaced"] = dict()
        
        for replica_idx in range(num_replica):
            random.seed(seed+replica_idx)
            samples[sample_idx]["replaced"][replica_idx] = dict()
            
            # replace 된 라벨들 담아놓는 리스트
            replaced_tokens = list()
            
            # 원본 tokenized
            tokenized = tokenizer.encode(processed)
            
            origin_tokenized = copy.deepcopy(tokenized)
            
            # 모든 라벨에 대해 replacement 실행
            for label_idx, (alphabet, label) in enumerate(replace_link):
                
                
                # 교체 대상
                replace_from = f"<<<{label}>>>{alphabet}<<</{label}>>>"
                replace_from_tokenized = tokenizer.encode(replace_from)
                replace_to = None
                                
                # 비식별화 해야되는 부분
                # (1) 이름
                if label == "내국인이름":
                    replace_to = generator.korean_name(processed, alphabet)
                
                elif label == "외국인이름":
                    replace_to = random.choice(entity_mentions[random.choice([
                        "일본인이름","중국인이름","태국인이름","베트남이름","필리핀이름","몽골인이름",
                        "캄보디아이름",
                        "영어이름"
                    ])])
                
                # (2) 주민등록번호
                elif label == "주민등록번호":
                    replace_to = generator.resident_number()
                    
                # (3) 계좌번호
                elif label == "계좌번호":
                    replace_to = generator.account_numbers()
                
                # (4) 나이, 출생연도
                elif label == "나이":
                    label = "연령정보"
                    replace_to = str(random.choice(list(range(18, 75))))
                    
                elif label == "출생연도":
                    label = "연령정보"
                    replace_to = str(random.choice(list(range(1950, 2005))))
                
                # (5) 주소
                elif "아래주소" in label or label == "주소":
                    replace_to = generator.address(
                        processed=processed,
                        label=label, alphabet=alphabet,
                        entity_mentions=entity_mentions,
                        filename=sample["filename"]
                    )
                    
                    if (
                        replace_to.endswith("읍읍") 
                        or replace_to.endswith("면면") 
                        or replace_to.endswith("동동") 
                        or replace_to.endswith("리리") 
                        or replace_to.endswith("시시") 
                        or replace_to.endswith("도도") 
                        ):
                        replace_to = replace_to[:-1]
                    
                # (6) 전화번호 / 휴대폰번호
                elif label in ["전화번호", "휴대폰번호"]:
                    replace_to = generator.phone_numbers(label)
                
                # (7) 숫자/알파벳 정보  
                elif label == "신용카드번호":
                    replace_to = generator.card_numbers()
                
                elif label == "동":
                    if random.random() < 0.5:
                        candid = [i+1 for i in range(10)]
                        replace_to = str(random.choice(candid))
                        if random.random() < 0.1:
                            additional = ['A', 'B', 'C', 'D']
                            replace_to = replace_to + random.choice(additional)
                    else:
                        candid = [(i+1)*100 + j for i in range(22) for j in range(0, 10)]
                        replace_to = str(random.choice(candid))
                        
                elif label in ["수용동", "수용실"]:
                    replace_to = str(random.choice(list(range(1, 10))))
                        
                elif label in ["호", "실", "호실", "객실",
                        "출구", "출구번호",
                        "진료실", "생활관",
                        "룸","승강장","번", "번호", '방',
                        '호', "호실", "객실번호", "실",
                        "출구번호", "탑승장번호", "승강장번호", 
                        "버스번호", "광역버스", "노선번호",
                        "호선",
                        "층",
                        '명수']:
                    replace_to = generator.bun_numbers(processed, label, alphabet)
                    
                elif label == "레일":
                    replace_to = random.choice(["A", "B", "C", "D", "E", "F"])
                
                elif label == "열차번호":
                    replace_to = generator.train_numbers()
                    
                elif label in ["차량번호", "택시", "화물차"]:
                    replace_to = generator.car_numbers()
                
                elif label in ["버스", "버스번호"]:
                    label = "버스번호"
                    replace_to = generator.bus_numbers()
                    
                elif label in ["승차장"]:
                    replace_to = str(random.choice(list(range(1, 41))))
                    
                elif label == "연도":
                    replace_to = str(random.choice(list(range(1960, 2025))))
                    
                
                # (8) 위치기준
                elif label == "위치기준":                    
                    candidates = []
                    # TODO
                    for low_category in ["빌딩", "버스정류장", 
                                         "한식당", "중식당", "일식당", "분식점", "고깃집", "동남아음식점",
                                         "휴대전화대리점", "커피", "주점", 
                                         "기업명",
                                         "금융기관", "은행", "복합시설", "상가", "시장", "아울렛",
                                         "경찰서", "아파트", "편의점", "주유소", "마트"]:
                        candidates += entity_mentions[low_category]
                    
                    replace_to = random.choice(candidates)
                    
                elif label == "소재":
                    label = '행정동' # TODO : 행정군, 행정리, 찾기
                    replace_to = random.choice(entity_mentions[label])
                        
                # (9) 사업체 random choice
                elif label in ["음식점", "식당", "프랜차이즈식당", "외식업"]:
                    # new label
                    label = random.choice(["한식당", "중식당", "일식당", "분식점", "고깃집", "동남아음식점"])
                    replace_to = random.choice(entity_mentions[label])
                    
                elif label in ["식품업체"]:
                    replace_to = random.choice(entity_mentions[random.choice(["식품회사", "식품가공업", "식품유통업"])])
                
                elif label in ["매장", "매장이름","가게", "점포", "매점", "노점상", "노점"]:
                    # new label
                    label = "매장"
                    replace_to = random.choice(entity_mentions[random.choice(get_sub_categories(
                        "도소매및유통",
                        root_path=replace_subset_dataset_dir))])
                
                
                elif label in ["기업명", "주식회사", "유한회사"]:
                    label = "기업명"
                    replace_to = random.choice(entity_mentions[random.choice(["기업명", "주식회사"])])
                
                    
                elif label in ["사무실", "사무소"]:
                    replace_to = random.choice(entity_mentions[random.choice(
                        ["공인중개사", "법률사무소", "기업명", "주식회사"])])
                
                elif label in ["사업체", "사업체일반"]:
                    label = "사업체"
                    replace_to = random.choice(entity_mentions[random.choice(
                        get_sub_categories("도소매및유통", root_path=replace_subset_dataset_dir)
                        + get_sub_categories("서비스일반", root_path=replace_subset_dataset_dir)
                        + get_sub_categories("오락및스포츠", root_path=replace_subset_dataset_dir)
                        + get_sub_categories("외식업", root_path=replace_subset_dataset_dir)
                        + get_sub_categories("유흥업", root_path=replace_subset_dataset_dir)
                    )])
                    
                elif label in ["작업실"]:
                    replace_to = random.choice(entity_mentions[random.choice(
                        get_sub_categories("제조업", root_path=replace_subset_dataset_dir)
                    )])
                    
                elif label == "지부":
                    replace_to = random.choice(entity_mentions["지점"])
                    if replace_to.endswith("지점"):
                        replace_to = replace_to[:-2] + "지부"
                    else:
                        replacec_to = replace_to[:-1] + "지부"
                    
                # (10) 시설및기관
                elif label in ["사회복지법인산하기관"]:
                    label = "사회복지시설"
                    replace_to = random.choice(entity_mentions[random.choice(get_sub_categories(
                        "사회복지시설",
                        root_path=replace_subset_dataset_dir
                    ))])
                    
                elif label in ["센터"]:
                    label = random.choice(["스포츠센터", "주민센터", "안전센터", "치안센터"])
                    replace_to = random.choice(entity_mentions[label])
                
                elif label =="중고등학교":
                    label = random.choice(["중학교", "고등학교"])
                    replace_to = random.choice(entity_mentions[label])
                
                elif label =="학교":
                    label = random.choice(["초등학교", "중학교", "고등학교", "대학교"])
                    replace_to = random.choice(entity_mentions[label])
                
                
                # (11) 조직
                elif "정비조합" in label: # 주택재개발정비조합...
                    replace_to = random.choice(entity_mentions[
                        random.choice(["재개발정비조합", "재건축정비조합"])
                    ])

                elif label in entity_mentions.keys():
                    replace_to = random.choice(entity_mentions[label])
                
                    
                elif label in ["추행장소", "사기장소", "폭행장소"]:
                    candid = []
                    
                    if label == "추행장소":
                        subcat = ["외식업", "유흥업", "서비스일반",
                                 "숙박업"]
                        for p in subcat:
                            candid += get_sub_categories(p, root_path=replace_subset_dataset_dir)
                        candid += [
                            "마사지",
                            "안마시술소",
                            "안마원",
                            "왁싱샵"
                        ]
                        label = random.choice(candid)
                        replace_to = random.choice(entity_mentions[label])
                        
                    elif label == "사기장소":
                        subcat = ["외식업", "유흥업", "서비스일반", 
                                 "부동산중개및임대매매", "건설",
                                 "도소매및유통"]
                        for p in subcat:
                            candid += get_sub_categories(p, root_path=replace_subset_dataset_dir)
                        label = random.choice(candid)
                        replace_to = random.choice(entity_mentions[label])
                    
                    elif label == "폭행장소":
                        subcat = ["외식업", "유흥업", "서비스일반", 
                        ]
                        for p in subcat:
                            candid += get_sub_categories(p, root_path=replace_subset_dataset_dir)
                        candid += [
                            "공원", "산책로", 
                            "교차로", "길", "골목", "도로", "사거리",
                            "아파트", "빌라", "오피스텔", "고시원", "고급주택", "맨션", "주상복합"
                        ]
                        label = random.choice(candid)
                        replace_to = random.choice(entity_mentions[label])
                                                  
                
                if replace_to is not None:
                    # handling white space and josa
                    replace_target_text_idx = processed.find(replace_from)
                    
                    # leading white space
                    if has_leading_whitespace(processed, replace_target_text_idx):
                        leading_whitespace = True
                    else:
                        leading_whitespace = False

                    encoding_target = replace_to
                    if leading_whitespace:
                        encoding_target = ' ' + encoding_target

                    replace_to_tokenized = tokenizer.encode(encoding_target)

                    replaced_tokens.append({
                        "leading_whitespace":leading_whitespace,
                        "replace_to":replace_to,
                        "encoding_target":encoding_target,
                        "replace_to_tokenized":replace_to_tokenized,
                        "replace_from_tokenized":replace_from_tokenized,
                        "replace_target_text_idx":replace_target_text_idx,
                        "label":label
                    })


            # generate X, Y pairs while inserting entity tokens
            origin_tokenized = tokenizer.encode(processed)
            X = copy.deepcopy(origin_tokenized)  # token sequence
        
            # generate X replacing entities
            replaced_tokens.sort(key=lambda x: x['replace_target_text_idx'])
            for i_idx, item in enumerate(replaced_tokens):
                modified_X = []
                modified_Y = []
                label = item["label"]
                replace_from = item["replace_from_tokenized"]
                replace_to   = item["replace_to_tokenized"] # 조사 포함됨
                leading_whitespace = item["leading_whitespace"]

                X_str = [str(tok) for tok in X]
                X_str = ','.join(X_str)

                replace_to_str = [str(tok) for tok in replace_to]
                replace_to_str = ','.join(replace_to_str)

                replace_from_str = [str(tok) for tok in replace_from]
                replace_from_str = ','.join(replace_from_str)

                # preparation
                replace_from_str_whitespace = None

                if leading_whitespace:
                    replace_from_str_whitespace = str(tokenizer.convert_tokens_to_ids(' ')) + ',' + replace_from_str
                
                # replacement
                replaced = copy.deepcopy(X_str)
                if leading_whitespace: 
                    replaced = replaced.replace(replace_from_str_whitespace, replace_to_str)

                replaced = replaced.replace(replace_from_str, replace_to_str)

                replaced_split = replaced.split(',')
                replaced_split = [int(tok) for tok in replaced_split]

                X = replaced_split
            
            decoded = tokenizer.decode(X)
            adjusted = adjust_spacing(decoded)


            Y = ['O' for _ in range(len(X))]
            for i_idx, item in enumerate(replaced_tokens):
                # target string
                target_str = item["replace_to"].lstrip().rstrip()

                label = item["label"]
                replace_from = item["replace_from_tokenized"]
                replace_to   = item["replace_to_tokenized"] # 조사 포함됨
                leading_whitespace = item["leading_whitespace"]

                # labeling
                x_idx = 0
                width = len(replace_to)
                while x_idx + width < len(X):
                    cur = X[x_idx : x_idx + width]
                    if cur == replace_to:
                        for y_idx in range(x_idx, x_idx + width):
                            Y[y_idx] = label
                        x_idx = x_idx + width
                    else:
                        x_idx = x_idx + 1 


            if len(X) != len(Y):
                print()
                print(sample_idx, replica_idx)
                print(adjusted)
                exit()

            # 샘플 업데이트
            samples[sample_idx]["replaced"][replica_idx]["decoded"] = decoded
            samples[sample_idx]["replaced"][replica_idx]["adjusted"] = adjusted
            samples[sample_idx]["replaced"][replica_idx]["tokens"] = X
            samples[sample_idx]["replaced"][replica_idx]["labels"] = Y
    

    # 데이터셋 저장
    with open(dataset_path, 'wb') as fwb:
        pickle.dump(samples, fwb)        
        
    return samples, tokenizer
        


def parallel_generate_dataset_with_modified_tokenizer(
    # inputs
    tokenizer,
    seed=1203,
    num_replica=30,
    raw_dataset_dir="./datasets/0_raw_documents",
    replace_subset_dataset_dir="./datasets/1_replacement/subset",
    replace_addr_dataset_dir="./datasets/1_replacement/address",
    replace_map_path=f"./datasets/replace_map.json",
    
    # output
    dataset_path=f"./datasets/2_deid_dataset/.pkl",
    modified_tokenizer_path=f"./datasets/2_deid_dataset/.pkl",
    
    # misc.
    num_cpus=16,
    ):
        

    PROVINCE = [
        '서울특별시', 
        '인천광역시', '부산광역시', '대구광역시', 
        '대전광역시', '광주광역시', '울산광역시', 
        '세종특별자치시', 
        '경기도', 
        '강원도', '강원특별자치도', 
        '충청북도', '충청남도', 
        '경상북도', '경상남도', 
        '전라북도', '전북특별자치도', 
        '전라남도', 
        '제주특별자치도'
    ]
    random.seed(seed)    
    
    
    """ 1. raw documents """
    raw_documents = {
        "indecent_act_by_compulsion":sorted(glob.glob(f"{raw_dataset_dir}/indecent_act_by_compulsion/*.md")),
        "crime_of_violence":sorted(glob.glob(f"{raw_dataset_dir}/crime_of_violence/*.md")),
        "fraud":sorted(glob.glob(f"{raw_dataset_dir}/fraud/*.md")),
    }
    
    
    """ 2. lists for replacement """
    entity_mentions = dict()
    rep_subset_files = get_json_filenames(replace_subset_dataset_dir)
    for filename in rep_subset_files:
        with open(filename, 'r') as fr:
            data = json.load(fr)
        entity_mentions.update(data)
    
    # addr
    with open(f"{replace_addr_dataset_dir}/REPLACEMENT_LIST-addrs1.json", 'r') as fr1:
        data = json.load(fr1)
        data = {key:value for key, value in data.items()
                     if key in PROVINCE}
        entity_mentions["지번주소"] = data
        for province in PROVINCE:
            if province.endswith("시"):
                entity_mentions["지번주소"][province] = {k:v for k,v in entity_mentions["지번주소"][province].items()
                                              if k.endswith("구")
                                              or k.endswith("군")}
                
    with open(f"{replace_addr_dataset_dir}/REPLACEMENT_LIST-addrs2.json", 'r') as fr2:
        data = json.load(fr2)
        data = {key:value for key, value in data.items()
                     if key in PROVINCE}
        entity_mentions["도로명주소"] = data
        for province in PROVINCE:
            if province.endswith("시"):
                entity_mentions["도로명주소"][province] = {k:v for k,v in entity_mentions["도로명주소"][province].items()
                                              if k.endswith("구")
                                              or k.endswith("군")}
        
    """ 3. for processing (replace map) """
    with open(replace_map_path, 'r') as fr:
        replace_map = json.load(fr)

        
    """ samples """
    samples = list()
    
    """ preprocessing """            
    all_labels = set()
    
    
    # prepare subcategories
    sub_categories = dict()
    sub_categories["매장"] = get_sub_categories(
        "도소매및유통", root_path=replace_subset_dataset_dir)
    sub_categories["사업체"] = (
        get_sub_categories("도소매및유통", root_path=replace_subset_dataset_dir)
        + get_sub_categories("서비스일반", root_path=replace_subset_dataset_dir)
        + get_sub_categories("오락및스포츠", root_path=replace_subset_dataset_dir)
        + get_sub_categories("외식업", root_path=replace_subset_dataset_dir)
        + get_sub_categories("유흥업", root_path=replace_subset_dataset_dir)
    )
    sub_categories["작업실"] = get_sub_categories(
        "제조업", root_path=replace_subset_dataset_dir)
    sub_categories["사회복지시설"] = get_sub_categories(
        "사회복지시설", root_path=replace_subset_dataset_dir
    )

    # 추행장소
    sub_categories["추행장소"] = list()
    for p in ["외식업", "유흥업", "서비스일반", "숙박업"]:
        sub_categories["추행장소"] += get_sub_categories(p, root_path=replace_subset_dataset_dir)
    sub_categories["추행장소"] += ["마사지", "안마시술소", "안마원", "왁싱샵"]
    
    # 폭행장소
    sub_categories["폭행장소"] = list()
    for p in ["외식업", "유흥업", "서비스일반"]:
        sub_categories["폭행장소"] += get_sub_categories(p, root_path=replace_subset_dataset_dir)
    sub_categories["폭행장소"] += [
        "공원", "산책로", 
        "교차로", "길", "골목", "도로", "사거리",
        "아파트", "빌라", "오피스텔", "고시원", "고급주택", "맨션", "주상복합"
    ]
    
    # 사기장소
    sub_categories["사기장소"] = list()
    for p in ["외식업", "유흥업", "서비스일반", 
              "부동산중개및임대매매", "건설", "도소매및유통"]:
        sub_categories["사기장소"] += get_sub_categories(p, root_path=replace_subset_dataset_dir)
        
    sample_idx = 0
    for category in raw_documents.keys():        
        for filename in raw_documents[category]:
        # for f_idx, filename in enumerate(raw_documents[category]):
            # read raw document
            with open(filename, 'r') as fr:
                raw = fr.read()
                                
            # preprocess raw document
            processed, labels, alphabets = preprocessing(
                raw, 
                replace_map=replace_map,
                filename=filename
            )
            replace_link = [(a, l) for a, l in zip(alphabets, labels)]
                            
            samples.append({
                "raw":raw,
                "sample_idx":sample_idx,
                "processed":processed,
                "replace_link":replace_link,
                "filename":filename, # TODO : remove or modify this
                "category":category
            })
            for labels in labels:
                all_labels.add(labels)
                
            sample_idx += 1
            
    samples.sort(key=lambda x:x["sample_idx"])

    # add special tokens to tokenizer
    all_labels = sorted(all_labels)
    special_tokens = []
    special_tokens += [AddedToken(f"<<<{label}>>>", lstrip=False, rstrip=False, single_word=False, normalized=True, special=True) for label in all_labels]
    special_tokens += [AddedToken(f"<<</{label}>>>", lstrip=False, rstrip=False, single_word=False, normalized=True, special=True) for label in all_labels]
    
    tokenizer.add_special_tokens({"additional_special_tokens":special_tokens})
    special_token_set = set(tokenizer.additional_special_tokens)
    
    # save the modified tokenizer    
    tokenizer.save_pretrained(modified_tokenizer_path)
    
    # switch to mecab-bpe mode (can't be serialized)
    # tokenizer = custom.switch_dummy(tokenizer)
    # switch_tokenizer()

    # DEBUG
    if False:
        final_samples = []
        for sample in tqdm(samples, ncols=80):
            final_samples.append(
                process_sample(
                    sample,
                    tokenizer,
                    entity_mentions,
                    sub_categories,
                    num_replica,
                    seed
                )   
            )
    # DEBUG
            
    chunk_size = math.ceil(len(samples) / num_cpus)
    chunks = [samples[i:i + chunk_size] for i in range(0, len(samples), chunk_size)]

    partial_chunk_fn = partial(
        process_chunk,
        tokenizer=tokenizer,
        entity_mentions=entity_mentions,
        sub_categories=sub_categories,
        num_replica=num_replica,
        seed=seed
    )

    final_samples = []

    with mp.Pool(processes=num_cpus) as pool:
        with tqdm(total=len(samples), ncols=80) as pbar:
            for chunk_result in pool.imap_unordered(partial_chunk_fn, chunks):
                final_samples.extend(chunk_result)
                pbar.update(len(chunk_result))
                
    final_samples.sort(key=lambda x: x['filename'])
    
    
    # 데이터셋 저장
    with open(dataset_path, 'wb') as fwb:
        pickle.dump(final_samples, fwb)        
        
    return final_samples, tokenizer
        

def get_entities(
    tokenizer,
    entity_mentions,
    sub_categories,
    replace_link,
    processed,
    category,
    num_replica,
    seed,
    filename=None, # for debugging
):
    output_dict = dict()
    for replica_idx in range(num_replica):
        output_dict[replica_idx] = dict()
        output_dict[replica_idx]["replaced_tokens"] = list()
    
    for replica_idx in range(num_replica):
        random.seed(seed+replica_idx)
        
        # replace 된 라벨들 담아놓는 리스트
        replaced_tokens = list()
        
        # 원본 tokenized
        tokenized = tokenizer.encode(processed)
        
        origin_tokenized = copy.deepcopy(tokenized)
        
        # 모든 라벨에 대해 replacement 실행
        for label_idx, (alphabet, label) in enumerate(replace_link):
                            
            # 교체 대상
            replace_from = f"<<<{label}>>>{alphabet}<<</{label}>>>"
            replace_from_tokenized = tokenizer.encode(replace_from)
            replace_to = None
                            
            # 비식별화 해야되는 부분
            # (1) 이름
            if label == "내국인이름":
                replace_to = generator.korean_name(processed, alphabet)
            
            elif label == "외국인이름":
                replace_to = random.choice(entity_mentions[random.choice([
                    "일본인이름","중국인이름","태국인이름","베트남이름","필리핀이름","몽골인이름",
                    "캄보디아이름",
                    "영어이름"
                ])])
            
            # (2) 주민등록번호
            elif label == "주민등록번호":
                replace_to = generator.resident_number()
                
            # (3) 계좌번호
            elif label == "계좌번호":
                replace_to = generator.account_numbers()
            
            # (4) 나이, 출생연도
            elif label == "나이":
                label = "연령정보"
                replace_to = str(random.choice(list(range(18, 75))))
                
            elif label == "출생연도":
                label = "연령정보"
                replace_to = str(random.choice(list(range(1950, 2005))))
            
            # (5) 주소
            elif "아래주소" in label or label == "주소":
                replace_to = generator.address(
                    processed=processed,
                    label=label, alphabet=alphabet,
                    entity_mentions=entity_mentions,
                    filename=filename,
                )
                
                if (
                    replace_to.endswith("읍읍") 
                    or replace_to.endswith("면면") 
                    or replace_to.endswith("동동") 
                    or replace_to.endswith("리리") 
                    or replace_to.endswith("시시") 
                    or replace_to.endswith("도도") 
                    ):
                    replace_to = replace_to[:-1]
                
            # (6) 전화번호 / 휴대폰번호
            elif label in ["전화번호", "휴대폰번호"]:
                replace_to = generator.phone_numbers(label)
            
            # (7) 숫자/알파벳 정보  
            elif label == "신용카드번호":
                replace_to = generator.card_numbers()
            
            elif label == "동":
                if random.random() < 0.5:
                    candid = [i+1 for i in range(10)]
                    replace_to = str(random.choice(candid))
                    if random.random() < 0.1:
                        additional = ['A', 'B', 'C', 'D']
                        replace_to = replace_to + random.choice(additional)
                else:
                    candid = [(i+1)*100 + j for i in range(22) for j in range(0, 10)]
                    replace_to = str(random.choice(candid))
                    
            elif label in ["수용동", "수용실"]:
                replace_to = str(random.choice(list(range(1, 10))))
                    
            elif label in ["호", "실", "호실", "객실",
                    "출구", "출구번호",
                    "진료실", "생활관",
                    "룸","승강장","번", "번호", '방',
                    '호', "호실", "객실번호", "실",
                    "출구번호", "탑승장번호", "승강장번호", 
                    "버스번호", "광역버스", "노선번호",
                    "호선",
                    "층",
                    '명수']:
                replace_to = generator.bun_numbers(processed, label, alphabet)
                
            elif label == "레일":
                replace_to = random.choice(["A", "B", "C", "D", "E", "F"])
            
            elif label == "열차번호":
                replace_to = generator.train_numbers()
                
            elif label in ["차량번호", "택시", "화물차"]:
                replace_to = generator.car_numbers()
            
            elif label in ["버스", "버스번호"]:
                label = "버스번호"
                replace_to = generator.bus_numbers()
                
            elif label in ["승차장"]:
                replace_to = str(random.choice(list(range(1, 41))))
                
            elif label == "연도":
                replace_to = str(random.choice(list(range(1960, 2025))))
                
            
            # (8) 위치기준
            elif label == "위치기준":                    
                candidates = []
                # TODO
                for low_category in ["빌딩", "버스정류장", 
                                        "한식당", "중식당", "일식당", "분식점", "고깃집", "동남아음식점",
                                        "휴대전화대리점", "커피", "주점", 
                                        "기업명",
                                        "금융기관", "은행", "복합시설", "상가", "시장", "아울렛",
                                        "경찰서", "아파트", "편의점", "주유소", "마트"]:
                    candidates += entity_mentions[low_category]
                
                replace_to = random.choice(candidates)
                
            elif label == "소재":
                label = '행정동' # TODO : 행정군, 행정리, 찾기
                replace_to = random.choice(entity_mentions[label])
                    
            # (9) 사업체 random choice
            elif label in ["음식점", "식당", "프랜차이즈식당", "외식업"]:
                # new label
                label = random.choice(["한식당", "중식당", "일식당", "분식점", "고깃집", "동남아음식점"])
                replace_to = random.choice(entity_mentions[label])
                
            elif label in ["식품업체"]:
                replace_to = random.choice(entity_mentions[random.choice(["식품회사", "식품가공업", "식품유통업"])])
            
            elif label in ["매장", "매장이름","가게", "점포", "매점", "노점상", "노점"]:
                # new label
                label = "매장"
                replace_to = random.choice(entity_mentions[random.choice(sub_categories["매장"])])
            
            elif label in ["기업명", "주식회사", "유한회사"]:
                label = "기업명"
                replace_to = random.choice(entity_mentions[random.choice(["기업명", "주식회사"])])
            
                
            elif label in ["사무실", "사무소"]:
                replace_to = random.choice(entity_mentions[random.choice(
                    ["공인중개사", "법률사무소", "기업명", "주식회사"])])
            
            elif label in ["사업체", "사업체일반"]:
                label = "사업체"
                replace_to = random.choice(entity_mentions[random.choice(
                    sub_categories["사업체"]
                )])
                
            elif label in ["작업실"]:
                replace_to = random.choice(entity_mentions[random.choice(
                    sub_categories["작업실"]
                )])
                
            elif label == "지부":
                replace_to = random.choice(entity_mentions["지점"])
                if replace_to.endswith("지점"):
                    replace_to = replace_to[:-2] + "지부"
                else:
                    replacec_to = replace_to[:-1] + "지부"
                
            # (10) 시설및기관
            elif label in ["사회복지법인산하기관"]:
                label = "사회복지시설"
                replace_to = random.choice(entity_mentions[random.choice(sub_categories["사회복지시설"])])
                
            elif label in ["센터"]:
                label = random.choice(["스포츠센터", "주민센터", "안전센터", "치안센터"])
                replace_to = random.choice(entity_mentions[label])
            
            elif label =="중고등학교":
                label = random.choice(["중학교", "고등학교"])
                replace_to = random.choice(entity_mentions[label])
            
            elif label =="학교":
                label = random.choice(["초등학교", "중학교", "고등학교", "대학교"])
                replace_to = random.choice(entity_mentions[label])
            
            
            # (11) 조직
            elif "정비조합" in label: # 주택재개발정비조합...
                replace_to = random.choice(entity_mentions[
                    random.choice(["재개발정비조합", "재건축정비조합"])
                ])

            elif label in entity_mentions.keys():
                replace_to = random.choice(entity_mentions[label])
            
            elif label == "추행장소":
                label = random.choice(sub_categories["추행장소"])
                replace_to = random.choice(entity_mentions[label])
            elif label == "폭행장소":
                label = random.choice(sub_categories["폭행장소"])
                replace_to = random.choice(entity_mentions[label])
            elif label == "사기장소":
                label = random.choice(sub_categories["사기장소"])
                replace_to = random.choice(entity_mentions[label])
                                            
            
            if replace_to is not None:
                # handling white space and josa
                replace_target_text_idx = processed.find(replace_from)
                
                # leading white space
                if has_leading_whitespace(processed, replace_target_text_idx):
                    leading_whitespace = True
                else:
                    leading_whitespace = False

                encoding_target = replace_to
                if leading_whitespace:
                    encoding_target = ' ' + encoding_target

                replace_to_tokenized = tokenizer.encode(encoding_target)

                output_dict[replica_idx]["replaced_tokens"].append({
                    "leading_whitespace":leading_whitespace,
                    "replace_to":replace_to,
                    "encoding_target":encoding_target,
                    "replace_to_tokenized":replace_to_tokenized,
                    "replace_from_tokenized":replace_from_tokenized,
                    "replace_target_text_idx":replace_target_text_idx,
                    "label":label
                })
    
    return output_dict



def get_tok_label_sequence(
    tokenizer,
    processed,
    replaced_subdict,
    num_replica,
):
    output_dict = dict()
    for replica_idx in range(num_replica):
        output_dict[replica_idx] = dict()

    for replica_idx in range(num_replica):
        # generate X, Y pairs while inserting entity tokens
        origin_tokenized = tokenizer.encode(processed)
        X = copy.deepcopy(origin_tokenized)  # token sequence
        # generate X replacing entities
        replaced_tokens = replaced_subdict[replica_idx]["replaced_tokens"]
        
        replaced_tokens.sort(key=lambda x: x['replace_target_text_idx'])
        for i_idx, item in enumerate(replaced_tokens):
            modified_X = []
            modified_Y = []
            label = item["label"]
            replace_from = item["replace_from_tokenized"]
            replace_to   = item["replace_to_tokenized"] # 조사 포함됨
            leading_whitespace = item["leading_whitespace"]

            X_str = [str(tok) for tok in X]
            X_str = ','.join(X_str)

            replace_to_str = [str(tok) for tok in replace_to]
            replace_to_str = ','.join(replace_to_str)

            replace_from_str = [str(tok) for tok in replace_from]
            replace_from_str = ','.join(replace_from_str)

            # preparation
            replace_from_str_whitespace = None

            if leading_whitespace:
                replace_from_str_whitespace = str(tokenizer.convert_tokens_to_ids(' ')) + ',' + replace_from_str
            
            # replacement
            replaced = copy.deepcopy(X_str)
            if leading_whitespace: 
                replaced = replaced.replace(replace_from_str_whitespace, replace_to_str)

            replaced = replaced.replace(replace_from_str, replace_to_str)

            replaced_split = replaced.split(',')
            replaced_split = [int(tok) for tok in replaced_split]

            X = replaced_split
        
        decoded = tokenizer.decode(X)
        adjusted = adjust_spacing(decoded)

        Y = ['O' for _ in range(len(X))]
        for i_idx, item in enumerate(replaced_tokens):
            # target string
            target_str = item["replace_to"].lstrip().rstrip()

            label = item["label"]
            replace_from = item["replace_from_tokenized"]
            replace_to   = item["replace_to_tokenized"] # 조사 포함됨
            leading_whitespace = item["leading_whitespace"]

            # labeling
            x_idx = 0
            width = len(replace_to)
            while x_idx + width < len(X):
                cur = X[x_idx : x_idx + width]
                if cur == replace_to:
                    for y_idx in range(x_idx, x_idx + width):
                        Y[y_idx] = label
                    x_idx = x_idx + width
                else:
                    x_idx = x_idx + 1 
                  
        output_dict[replica_idx]["decoded"] = decoded
        output_dict[replica_idx]["adjusted"] = adjusted
        output_dict[replica_idx]["tokens"] = X
        output_dict[replica_idx]["labels"] = Y  
        
    return output_dict
    
    
    
def process_sample(
    sample,
    tokenizer,
    entity_mentions,
    sub_categories,
    num_replica,
    seed,
):
    # tokenizer = custom.switch_dummy(tokenizer)
    
    cur_row = dict()
    # n 개의 replaced replica 생성
    for replica_idx in range(num_replica):
        cur_row[replica_idx] = dict()
        
    filename = sample["filename"]
    replace_link = sample["replace_link"]
    processed = sample["processed"]
    category = sample["category"]
    
    cur_row["filename"] = filename
    cur_row["replace_link"] = replace_link
    cur_row["processed"] = processed
    cur_row["category"] = category
    
    # generate/select entities for replacements
    output_dict = get_entities(
        tokenizer=tokenizer,
        entity_mentions=entity_mentions,
        sub_categories=sub_categories,
        replace_link=replace_link, 
        processed=processed, 
        category=category,
        num_replica=num_replica,
        seed=seed,
        filename=filename
    )
    for replica_idx in range(num_replica):
        cur_row[replica_idx].update(output_dict[replica_idx])

    # generate token sequence / label sequence (token ids !)
    output_dict = get_tok_label_sequence(
        tokenizer=tokenizer,
        processed=processed,
        num_replica=num_replica,
        replaced_subdict=cur_row
    )
    for replica_idx in range(num_replica):
        cur_row[replica_idx].update(output_dict[replica_idx])

    return cur_row
    


def process_chunk(chunk, tokenizer, entity_mentions, sub_categories, num_replica, seed):
    tokenizer = custom.switch_dummy(tokenizer)

    results = []
    for sample in chunk:
        results.append(
            process_sample(sample, tokenizer, entity_mentions, sub_categories, num_replica, seed)
        )
    return results

# with parallel_backend('multiprocessing', prefer="fork"):  # 명시적 지정
#     results = Parallel(n_jobs=8)(
#         delayed(augment_n_times)(data) for data in original_data_list
#     )