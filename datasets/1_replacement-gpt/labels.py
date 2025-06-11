import os
import json

# 탐색할 루트 폴더 경로
root_folder = '.'

# 결과를 저장할 딕셔너리
json_keys = {}

# 모든 하위 폴더 및 파일 탐색
for dirpath, _, filenames in os.walk(root_folder):
    for filename in filenames:
        if filename.endswith('.json'):
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 파일 경로를 기준으로 키 저장
                    if isinstance(data, dict):
                        keys = list(data.keys())
                    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                        keys = list(data[0].keys())
                    else:
                        keys = ['[알 수 없는 구조]']
                    json_keys[file_path] = keys
            except Exception as e:
                json_keys[file_path] = [f'[오류: {str(e)}]']

# 결과 출력
L = []
for path, keys in json_keys.items():
    for key in keys:
        L.append(key)
L = sorted(L)
print(L)
print(len(L))
L = "\n".join(L)
with open("gpt-labels.txt", "w") as fw:
    fw.write(L)
