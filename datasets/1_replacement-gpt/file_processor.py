import os
import json

def find_leaf_dirs(root_dir):
    leaf_dirs = []
    for current_dir, subdirs, files in os.walk(root_dir):
        if 'all_labels.json' in files:
            leaf_dirs.append(current_dir)
    return leaf_dirs

def split_nested_labels_to_json(root_dir):
    leaf_dirs = find_leaf_dirs(root_dir)

    for leaf_dir in leaf_dirs:
        all_labels_path = os.path.join(leaf_dir, 'all_labels.json')
        try:
            with open(all_labels_path, 'r', encoding='utf-8') as f:
                all_labels = json.load(f)

            for top_key, subdict in all_labels.items():
                if isinstance(subdict, dict):
                    for sub_key, value_list in subdict.items():
                        out_path = os.path.join(leaf_dir, f"{sub_key}.json")
                        with open(out_path, 'w', encoding='utf-8') as out_f:
                            json.dump({sub_key: value_list}, out_f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"⚠️ {leaf_dir} 처리 중 오류 발생: {e}")

# 실행
root = '/shared/s1/lab08/sungeun/deid2/se_replacement_lists/plausible_labels'
split_nested_labels_to_json(root)
