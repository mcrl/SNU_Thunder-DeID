import os
import re
import random
from korean_name_generator import namer


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
PROVINCE_MAP = {
    "서울":["서울특별시"],"세종":["세종특별자치시"],
    "인천":["인천광역시"], "부산":["부산광역시"], "대구":["대구광역시"], 
    "대전":["대전광역시"], "울산":["울산광역시"], "광주":["광주광역시"],
    "강원":["강원도", "강원특별자치도"],
    "충북":["충청북도"], "충남":["충청남도"],
    "경북":["경상북도"], "경남":["경상남도"],
    "전북":["전라북도", "전북특별자치도"], "전남":["전라남도"],
    "제주":["제주특별자치도"]
}
METRO = ["서울", "세종", "인천", "부산", "대구", "대전", "울산", "광주"]

# 강원 고성군 ...
# 서울 강남구 ...




"""
    내국인이름
"""
def korean_name(document, alphabet):
    def __has_final_consonant(name):
        # 마지막 글자의 유니코드 값
        last_char = name[-1]
        if '가' <= last_char <= '힣':  # 한글인지 확인
            return (ord(last_char) - 0xAC00) % 28 != 0
        return False  # 한글이 아닌 경우

    def __generate_name_by_final_consonant(gender=True, josa_type=0):
        if josa_type == 0:  # 받침 없는 경우
            name = namer.generate(gender)
            while __has_final_consonant(name):
                name = namer.generate(gender) 
                # escape when name does not have a final consonant
        else:  # 받침 있는 경우
            name = namer.generate(gender)
            while not __has_final_consonant(name):
                name = namer.generate(gender) 
                # escape when name has a final consonant
        return name

    # pattern
    target_pat = fr"<<<내국인이름>>>{alphabet}<<</내국인이름>>>"
    
    # find gender info
    gender_pat = fr"{target_pat}\s*(?:\((.*?)\))?" # e.g., (남, 29세)
    gender_match = re.search(gender_pat, document)
    if gender_match:
        gender_info = gender_match.group(1)
        if gender_info:
            gender_info = re.search(r'(남|여)', gender_info)
            gender      = True if gender_info and gender_info.group(1) == "남" else False
        else:
            gender = random.choice([True, False])
    else:
        gender = random.choice([True, False])
    
    # 조사
    josa_type = None
    # (받침 없어야 되는 이름인지?)
    josa_pat = fr"{target_pat}({'|'.join(['와', '를', '가', '는', '로부터'])})" 
    josa_matches = re.findall(josa_pat, document)
    if len(josa_matches) > 0:
        josa_type = 0
    else:
        # (받침 있어야 되는 이름인지?)
        josa_pat = fr"{target_pat}({'|'.join(['과', '을', '이', '은', '으로부터'])})"
        josa_matches = re.findall(josa_pat, document)
        if len(josa_matches) > 0:
            josa_type = 1
        else:
            # (받침 상관 없는 경우 -> 아무거나 고름)
            josa_type = random.choice([0, 1])
            
    # set name
    person_name = __generate_name_by_final_consonant(gender=gender, josa_type=josa_type)
    return person_name



"""
    주민등록번호
"""
def resident_number():
    def __generate_korean_resident_number():
        """랜덤한 한국 주민등록번호 생성"""
        # 앞자리 (생년월일)
        year = random.randint(1955, 2023)  # 1900~2023년 출생 가정
        month = random.randint(1, 12)
        if month == 2:
            day = random.randint(1, 28)  
        elif month in [1,3,5,7,8,10,12]:
            day = random.randint(1, 31)
        else:
            day = random.randint(1, 30)
        
        yy = str(year)[2:]  # 년도 뒤 2자리
        mm = f"{month:02d}"  # 두 자리 월
        dd = f"{day:02d}"  # 두 자리 일

        # 뒷자리 첫 번째 숫자 (성별 코드)
        if year < 2000:
            gender_code = random.choice([1, 2])  # 1900~1999년생 남자(1), 여자(2)
        else:
            gender_code = random.choice([3, 4])  # 2000년 이후 남자(3), 여자(4)
        
        # 뒷자리 나머지 (지역번호 + 임의 숫자 + 검증번호)
        region_code = random.randint(0, 99)  # 출생 지역 (0~99)
        unique_number = random.randint(0, 99999)  # 5자리 랜덤 숫자

        region_str = f"{region_code:02d}"  # 두 자리 지역 코드
        unique_str = f"{unique_number:05d}"  # 다섯 자리 숫자
        
        return f"{yy}{mm}{dd}-{gender_code}{region_str}{unique_str}"
    
    replace_to = __generate_korean_resident_number()
    return replace_to    



"""
    전화번호 / 휴대폰번호
"""
def phone_numbers(tag):
    def __generate_korean_phone_number():
        """랜덤한 한국 일반 전화번호 생성"""
        area_codes = ["02", "031", "032", "051", "053", "042", "062", "052", "044",
                    "033", "043", "041", "063", "061", "054", "055", "064"]
        area_code = random.choice(area_codes)
        first_part = str(random.randint(200, 9999))  # 200~9999 사이의 번호
        second_part = str(random.randint(1000, 9999))  # 4자리 랜덤 번호
        return f"{area_code}-{first_part}-{second_part}"

    def __generate_korean_mobile_number():
        """랜덤한 한국 핸드폰 번호 생성"""
        mobile_prefixes = ["010"]
        prefix = random.choice(mobile_prefixes)
        first_part = str(random.randint(1000, 9999))  # 4자리 랜덤 번호
        second_part = str(random.randint(1000, 9999))  # 4자리 랜덤 번호
        return f"{prefix}-{first_part}-{second_part}"
    
    if tag in ["전화번호"]:
        replace_to = __generate_korean_phone_number()
        
    elif tag in ["휴대폰번호"]:
        replace_to = __generate_korean_mobile_number()
    
    return replace_to


"""
    주소
"""
def address(
    processed, 
    label, alphabet,
    entity_mentions,
    filename="debug"
):
    def subaddr_filter(subaddress_list):
        subaddress_list = [item for item in subaddress_list
                            if '*' not in item
                            and 'null' not in item
                            and '학교' not in item
                            and '교육' not in item
                            and "터미널" not in item
                            and "공장" not in item
                            and "본사" not in item
                            and "주유쇼" not in item
                            and "지점" not in item
                            and '층' not in item
                            and '필지' not in item
                            and '빌딩' not in item
                            and '건물' not in item
                            and '타워' not in item
                            and "포항시남구" not in item
                            and item != '외'
                            and len(item.lstrip().rstrip()) > 0 ]
        subaddress_list = [item for item in subaddress_list
                           if ('-' in item)
                           or ('-' not in item and len(item) < 6)]
        subaddress_list = [item for item in subaddress_list if isinstance(item, str)]
        subaddress_list = list(filter(None, subaddress_list))
        return subaddress_list    
    
    
    if label == "주소":
        # random generate
        replace_to = []
        addr_type = addr_type = "지번주소" if random.random() > 0.5 else "도로명주소"
        cur_key = random.choice(PROVINCE)
        replace_to.append(cur_key)
        
        subaddr_candid = entity_mentions[addr_type]
        while (cur_key in subaddr_candid.keys()) and (len(subaddr_filter(subaddr_candid[cur_key]))>0) :
            subaddr_candid = subaddr_candid[cur_key]
            cur_key = random.choice(subaddr_filter(list(subaddr_candid.keys())))
            if cur_key.endswith(','): break
            elif '(' in cur_key: break # (부원동) 등
            replace_to.append(cur_key)

        replace_to = " ".join(replace_to)
        return replace_to
    
    elif label == "도아래주소":
        addr_pattern = f'<<<{label}>>>{alphabet}<<</{label}>>>'
        splited = processed.split(addr_pattern)
        
        front_words = splited[0].split()
        parent = front_words[-1]
        found = False
        for p in PROVINCE_MAP:
            if parent == p:
                province = random.choice(PROVINCE_MAP[p])
                found = True
                break
        if not found:
            for p in PROVINCE:
                if parent == p:
                    province = p
                    break
        
        # random generate
        addr_type = addr_type = "지번주소" if random.random() > 0.5 else "도로명주소"
        cur_key = province
        replace_to = []
        
        subaddr_candid = entity_mentions[addr_type]
        while (cur_key in subaddr_candid.keys()) and (len(subaddr_filter(subaddr_candid[cur_key]))>0) :
            subaddr_candid = subaddr_candid[cur_key]
            cur_key = random.choice(subaddr_filter(list(subaddr_candid.keys())))
            if cur_key.endswith(','): break
            elif '(' in cur_key: break
            replace_to.append(cur_key)

        replace_to = " ".join(replace_to)
        return replace_to
    
    
    """ ~아래주소 """
    # 어절로 분리
    addr_pattern = f'<<<{label}>>>{alphabet}<<</{label}>>>'
    splited = processed.split(addr_pattern)
    front_words = splited[0].split()
    if len(splited) > 1:
        rear_words  = splited[1].split()
    
    parent_addr_words = front_words[max(0, len(front_words)-2):]
    near = front_words[-4:-2]
    if len(splited) > 1:
        near = near + rear_words[:2]
        
    # (~ 소재)
    sojae = "소재" in near

    
    province = None
    metropolitan = None
    si = None
    gun = None
    gu = None
    
    eup = None
    myun = None
    dong = None
    
    r0 = parent_addr_words[0]
    r1 = parent_addr_words[1]
    
    # r0
    if r0 in PROVINCE_MAP:
        if PROVINCE_MAP[r0][0].endswith('도'):
            province = random.choice(PROVINCE_MAP[r0])
        elif PROVINCE_MAP[r0][0].endswith('시'):
            metropolitan = random.choice(PROVINCE_MAP[r0])
    elif r0 in entity_mentions["행정시"]:
        si = r0
        pass
    elif r0 +'시' in entity_mentions["행정시"]:
        si = r0 + '시'
        
    elif r0 in entity_mentions["행정군"]:
        gun = r0
        pass
    elif r0 +'군' in entity_mentions["행정군"]:
        gun = r0 + '군'
            
    # r1
    if province is not None:
        if r1 in entity_mentions["지번주소"][province]:
            if r1.endswith("시"):
                si = r1
            elif r1.endswith("군"):
                gun = r1
    elif metropolitan is not None:
        if r1 in entity_mentions["지번주소"][metropolitan]:
            if r1.endswith("구"):
                gu = r1
            elif r1.endswith("군"):
                gun = r1
    elif (si is not None) or (gun is not None):
        for p in PROVINCE:
            if r0 in entity_mentions["지번주소"][p]:
                province = p
            
        if province is not None:
            if r1 in entity_mentions["지번주소"][province][r0]:
                if r1.endswith("구"):
                    gu = r1
                elif r1.endswith("읍"):
                    eup = r1
    
    # 시나 군은 특정되었지만 '도' 는 특정되지 않은 경우
    # (광역시, 특별시는 제외, -> e.g., 수원시 영통구 <구이하주소> 에서)
    if (province is None):
        if (si is not None) or (gun is not None):
            for p in PROVINCE:
                if (si in entity_mentions["지번주소"][p]) or (gun in entity_mentions["지번주소"][p]):
                    if p.endswith("도"):
                        province = p
        else:
            for p in PROVINCE:
                if (r0 in entity_mentions["지번주소"][p]) or (gun in entity_mentions["지번주소"][p]):
                    if p.endswith("도"):
                        province = p
                    elif p.endswith("시"):
                        metropolitan = p
                        
                    if r0.endswith("시"):
                        si = r0
                    elif r0.endswith("군"):
                        gun = r0
                    elif r0.endswith("구"):
                        gu = r0
                    break
                
            for p in PROVINCE:
                if (r1 in entity_mentions["지번주소"][p]) or (gun in entity_mentions["지번주소"][p]):
                    if p.endswith("도"):
                        province = p
                    elif p.endswith("시"):
                        metropolitan = p
                        
                    if r1.endswith("시"):
                        si = r1
                    elif r1.endswith("군"):
                        gun = r1
                    elif r1.endswith("구"):
                        gu = r1
                    break

                   
    address_list_orig = [province, metropolitan, si, gun, gu, eup, myun, dong]
    address_list = [item for item in address_list_orig if item is not None] 
    

    # random generate
    addr_type = addr_type = "지번주소" if random.random() > 0.5 else "도로명주소"
    subaddr_candid = entity_mentions[addr_type]
    for a_idx in range(len(address_list)):
        if address_list[a_idx] in subaddr_candid:
            subaddr_candid = subaddr_candid[address_list[a_idx]]
        else:
            random_key = random.choice(list(subaddr_candid.keys()))
            address_list[a_idx] = random_key  # address_list도 바꿔줌
            subaddr_candid = subaddr_candid[random_key]
        
    cur_key = random.choice(subaddr_filter(list(subaddr_candid.keys())))
    replace_to = [cur_key]
    while (cur_key in subaddr_candid.keys()) and (len(subaddr_filter(subaddr_candid[cur_key]))>0) :
        subaddr_candid = subaddr_candid[cur_key]
        cur_key = random.choice(subaddr_filter(list(subaddr_candid.keys())))
        if cur_key.endswith(','): break
        elif '(' in cur_key: break
        replace_to.append(cur_key)
        
    
    if not sojae:
        replace_to = " ".join(replace_to)
    else:
        if (replace_to[0].endswith("읍") 
            or replace_to[0].endswith("면") 
            or replace_to[0].endswith("동")):
            replace_to = replace_to[0]
        else:
            replace_to = " ".join(replace_to[:2])
        
    return replace_to




def bun_numbers(document, label, alphabet):
    target_pat = re.findall(fr"<<<{label}>>>{alphabet}<<</{label}>>>", document)
    if len(target_pat) < 1:
        return None
    
    target_pat = target_pat[0]
    index = document.find(target_pat)
    replace_to = None
    if ("호" in document[index:(min(len(document)-1, index + len(target_pat) + 1))]
        or "실" in document[index:(min(len(document)-1, index + len(target_pat) + 1))]
        or "호실" in document[index:(min(len(document)-1, index + len(target_pat) + 3))]
        or "객실" in document[index:(min(len(document)-1, index + len(target_pat) + 3))]
        ):
        # 패턴 뒤 1글자 봤을 때 "출구" 있는 경우
        if "아파트" in document[index:(max(0, index - 15))]:
            # 패턴 앞 15글자 봤을 때 "아파트" 있는 경우
            levels = [integer for integer in list(range(1,31))] # ~30층
            rooms  = [integer for integer in list(range(0, 9))] # 방번호는 00부터 8까지
            additional = ["A", "B", "C", "D"]
                        
        elif "빌라" in document[index:(max(0, index - 15))]:
            # 패턴 앞 15글자 봤을 때 "빌라" 있는 경우
            levels = [integer for integer in list(range(1,9))] # ~8층
            rooms  = [integer for integer in list(range(0, 9))] # 방번호는 00부터 8까지
            additional = ["A", "B", "C", "D"]

        else:
            # 어떤 건물의 호를 지칭하는지 알 수 없는 경우
            levels = [integer for integer in list(range(1,9))] # ~4층
            rooms  = [integer for integer in list(range(0, 9))] # 방번호는 00부터 8까지
            additional = None
            
        # random generation
        level = str(random.choice(levels)) # e.g., 8
        room  = str(random.choice(rooms)) # e.g., 10
        room = room.zfill(2) # 1층 5번 호실 -> 105
        
        if random.random() > 0.9:
            if additional is not None:
                room = room + random.choice(additional) # 105A , 304B
        
        replace_to = level + room            
        
    elif ('호선' in document[index:(min(len(document)-1, index + len(target_pat) + 3))]):
        candidates = [integer for integer in list(range(1, 9))] # ~ 8호선
        replace_to = str(random.choice(candidates))
        
    elif (
        '방' in document[index:(min(len(document)-1, index + len(target_pat) + 1))] 
            or "번방" in document[index:(min(len(document)-1, index + len(target_pat) + 3))]
            or " 방" in document[index:(min(len(document)-1, index + len(target_pat) + 2))]
            or '룸' in document[index:(min(len(document)-1, index + len(target_pat) + 1))] 
            or "번룸" in document[index:(min(len(document)-1, index + len(target_pat) + 3))]
            or " 룸" in document[index:(min(len(document)-1, index + len(target_pat) + 2))]
            ):
        # 패턴 뒤 1글자 봤을 때 "방"(룸) 있는 경우
        # 패턴 뒤 2글자 봤을 때 "번방"(번룸) 있는 경우
        # 1~ 10번 방
        candidates = [integer for integer in list(range(1, 11))]
        replace_to = str(random.choice(candidates))
        
    elif ('홀' in document[index:(min(len(document)-1, index + len(target_pat) + 2))]
            or '번홀' in document[index:(min(len(document)-1, index + len(target_pat) + 3))]
            or '번 홀' in document[index:(min(len(document)-1, index + len(target_pat) + 4))]
            ):
        # 패턴 뒤 2글자 봤을 때 "홀" 있는 경우
        # 패턴 뒤 3글자 봤을 때 "번홀" 있는 경우
        # 패턴 뒤 4글자 봤을 때 "번 홀" 있는 경우
        # 1~18홀 (모든 골프장 18홀까지 있다고 함..)
        candidates = [integer for integer in list(range(1, 19))]
        replace_to = str(random.choice(candidates))

    elif ('층' in document[index:(min(len(document)-1, index + len(target_pat) + 2))]
            ):
        # 패턴 뒤 2글자 봤을 때 "층" 있는 경우
        candidates = [integer for integer in list(range(1, 11))] # 10층까지..
        replace_to = str(random.choice(candidates))
        
    elif ("출구" in document[index:(min(len(document)-1, index + len(target_pat) + 4))]
            or "출구번호" in document[index:(min(len(document)-1, index + len(target_pat) + 4))]
            or "탑승" in document[index:(min(len(document)-1, index + len(target_pat) + 4))]
            or "승강장" in document[index:(min(len(document)-1, index + len(target_pat) + 4))]
            ):
        # 패턴 뒤 4글자 봤을 때 "출구" 있는 경우
        # 1~ 16번 출구
        candidates = [integer for integer in list(range(1, 17))]
        replace_to = str(random.choice(candidates))
        
    elif "출입구" in document[index:(min(len(document)-1, index + len(target_pat) + 4))]:
        # 패턴 뒤 4글자 봤을 때 "출입구" 있는 경우
        # <<<번>>>RA<<</번>>> 출입구 앞 노상에서
        # 건물 출입구 -> 4까지만...
        candidates = [integer for integer in list(range(1, 5))]
        replace_to = str(random.choice(candidates))
        
    elif ("버스" in document[index:(min(len(document)-1, index + len(target_pat) + 10))]
            or "노선" in document[index:(min(len(document)-1, index + len(target_pat) + 10))]
            or label == "노선번호" # 노선번호 추가
            ):
        # 패턴 뒤 10글자 봤을 때 "버스" 있는 경우
            
        candidates = [integer for integer in list(range(10, 1000))] # 2자리, 혹은 3자리
        replace_to = str(random.choice(candidates))
        
        if "광역" in document[index:(min(len(document)-1, index + len(target_pat) + 10))]:
            # 광역버스는 서울 + 수도권만 운행하는 버스체계 의미
            replace_to = '9' + replace_to.zfill(3)           
    
    elif label == "진료실":
        candidates = [f"{(i+1)}진료실" for i in range(8)]
        replace_to = str(random.choice(candidates))

    elif label == "생활관":
        candidates = [f"{(i+1)}진료실" for i in range(20)]
        replace_to = str(random.choice(candidates))
        
    elif label == "명수":
        candidate_numbers = [10, 20, 30, 40, 50, 100, 200]
        candidates = [f"{number}명" for number in candidate_numbers]
        replace_to = str(random.choice(candidates))
        
    elif label == "반":
        candidates = [f"{(i+1)}" for i in range(10)]
        replace_to = str(random.choice(candidates))
        
    return replace_to


# 차량번호
def car_numbers():
    def __generate_korean_license_plate():
        # 2019년 이후 형식
        digits_4 = random.randint(1000, 9999)  # 네 자리 숫자
        # hangul = random.choice("가나다라마바사아자차카타파하")  # 한글 중 한 글자
        hangul_list = [ # 개인용 차량 
            "가", "나", "다", "라", "마",
            "거", "너", "더", "러", "머", "버", "서", "어", "저",
            "고", "노", "도", "로", "모", "보", "소", "오", "조",
            "구", "누", "두", "루", "무", "부", "수", "우", "주"
        ]
        hangul = random.choice(hangul_list)
        digits_4_after = random.randint(1000, 9999)  # 네 자리 숫자
        
        # 2019년 이전 형식
        digits_2 = random.randint(10, 99)  # 두 자리 숫자
        digits_4_legacy = random.randint(1000, 9999)  # 네 자리 숫자

        # 랜덤 선택: 신규 또는 이전 형식
        if random.random() < 0.5:  # 50% 확률
            return f"{digits_4}{hangul}{digits_4_after}"  # 신규 형식
        else:
            return f"{digits_2}{hangul}{digits_4_legacy}"  # 이전 형식     
    
    replace_to = __generate_korean_license_plate() # "12가5342"
    
    return replace_to


def bus_numbers():
    """
    한국형 버스 번호를 랜덤하게 생성합니다.
    반환: 문자열 형태의 버스 번호 (예: '123', 'G7107', '701A', '3200-1')
    """
    # 기본 숫자 생성 (1~9999)
    base_number = random.randint(1, 9999)
    
    # 버스 번호 형식 선택 (가중치로 확률 조정)
    format_choice = random.choices(
        ['simple', 'prefix', 'suffix', 'special_suffix'],
        weights=[0.4, 0.2, 0.2, 0.2],  # 단순 숫자 40%, 나머지 20%씩
        k=1
    )[0]
    
    if format_choice == 'simple':
        # 단순 숫자 (예: '123', '701', '3200')
        return str(base_number)
    
    elif format_choice == 'prefix':
        # 접두사 포함 (예: 'G123', 'M7107')
        prefixes = ['G', 'M', 'N', 'D', 'S']  # 경기, 마을, 노원 등
        prefix = random.choice(prefixes)
        return f"{prefix}{base_number}"
    
    elif format_choice == 'suffix':
        # 접미사 포함 (예: '123A', '701B')
        suffixes = ['A', 'B', 'C']
        suffix = random.choice(suffixes)
        return f"{base_number}{suffix}"
    
    elif format_choice == 'special_suffix':
        # 특수 접미사 (예: '123-1', '701급행')
        special_suffixes = ['-1', '-2', '급행', '지선', '간선']
        suffix = random.choice(special_suffixes)
        return f"{base_number}{suffix}"


# 철도번호
def train_numbers():
    """
    완전히 랜덤한 열차번호 생성 함수
    Parameters: 없음
    Returns:
        str: 랜덤하게 생성된 열차번호
    """
    # 운영 기관 무작위 선택
    operator = random.choice(['korail', 'seoul', 'incheon'])
    
    if operator == 'korail':
        # KORAIL 열차 종류 무작위 선택
        train_type = random.choice(['commuter', 'freight', 'ktx', 'itx', 'mugunghwa'])
        if train_type == 'commuter':
            # 광역전철: 1xxx, 3xxx, 4xxx 중 무작위
            line_prefix = random.choice(['1', '3', '4'])  # 1호선, 경의중앙, 수인선 예시
            return f"{line_prefix}{random.randint(100, 999)}"
        elif train_type == 'freight':
            # 화물열차: 6xxxx 또는 7xxxx
            prefix = random.choice(['6', '7'])
            return f"{prefix}{random.randint(1000, 9999)}"
        elif train_type == 'ktx':
            # KTX: 0xx
            return f"0{random.randint(10, 99)}"
        elif train_type == 'itx':
            # ITX-새마을: 08xx
            return f"08{random.randint(10, 99)}"
        else:  # mugunghwa
            # 무궁화: 1xxx
            return f"1{random.randint(100, 999)}"
    
    elif operator == 'seoul':
        # 서울교통공사 노선 무작위 선택
        line = random.choice(['2', '5', '6', '7', '8', '9'])
        if line == '2':
            # 2호선: S2xxx
            return f"S2{random.randint(100, 999)}"
        elif line in ['5', '6', '7', '8']:
            # 5~8호선: Lxxx
            return f"L{random.randint(100, 999)}"
        else:  # 9호선
            # 9호선: 9xxx
            return f"9{random.randint(100, 999)}"
    
    else:  # incheon
        # 인천교통공사 노선 무작위 선택
        line = random.choice(['1', '2'])
        if line == '1':
            # 인천 1호선: I1xx
            return f"I1{random.randint(10, 99)}"
        else:
            # 인천 2호선: I2xx
            return f"I2{random.randint(10, 99)}"


"""
    계좌번호
"""
def account_numbers():
    def __generate_account_number(pattern):
        account = ""
        for i, c in enumerate(pattern):
            if c == '-':
                account += '-'
            if i == 0 or i == len(pattern)-1:
                account += str(random.randint(1,9))
            else:
                account += str(random.randint(0,9))
        return account


    patterns = [
        "nnn-nn-nnnnn-n",
        "nnnn-nn-nnnnn-n",
        "nnn-nnnn-nnnn-n-n",

        "nnn-nn-nnnnn-n",
        "nnn-nnnnnnnn-n",
        "nnn-nn-nnnnnnnn-n",

        "nnnn-nnn-nnnnnn",

        "nnn-nn-nnnnn-n",
        "nnn-nnnnnnn-n-nnn",
        "nnn-nnnnnnnn-nnn",

        "nnn-nn-nnnn-nnn",
        "nnnn-nn-nnnnnnn-n",

        "nnn-nn-nnnnn-n",
        "nnn-nnnnnn-nnn",

        "nnn-nnnnn-nn-n-nn",
        "n-nnnnnn-n-nn-nn",
        "nn-nn-nnnnn-n",

        "nnn-nn-nnnnnn",
        "nnn-nn-nnnnnn-n",
        "nnn-nn-nnnnnn-n",
        "nnn-nn-nnnnnn-nnn",

        "nnn-nn-nnnnnn-n",
        "nnn-nnnn-nnnn-nn",

        "nnn-nnn-nnnnn-n",
        "nnn-nnn-nnnnnn",

        "nn-nn-nnnnnn",

        "nnn-nn-nnnnnnn",
        "n-nnn-nn-nnnnnnn",

        "nnn-nn-nnnnnnn",
        "nnn-nnnnnnnnn-n",

        "nnnn-nn-nnnnnn-n",
        "nnnn-nnn-nnnnnn-n",
        "nnnn-nnnnnnnn-n",

        "nnnnn-nn-nnnnn-n",
        "nnnnn-nn-nnnnnn-n",
        "nnn-nnn-nnnn",
        "nnn-nnnn-nnnn",
        "nnn-nnn-nnnnn-n",

        "nnnnnn-nn-nnnnnn",

        "nnn-nnnnnnnn-n",
        "nnn-nnn-nnnnnnn-n",]


    # 계좌번호 패턴
    account_pat = random.choice(patterns)
    
    # 무작위 계좌번호
    number = __generate_account_number(account_pat)
    
    return number


# 카드번호
def card_numbers():
    return ' '.join(f"{random.randint(0, 9999):04}" for _ in range(4))