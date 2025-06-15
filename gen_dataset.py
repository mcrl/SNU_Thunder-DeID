import argparse
import os
import json
from transformers import AutoTokenizer

from data_generator import generate_dataset_with_modified_tokenizer

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1203, type=int)
    parser.add_argument("--num_replica", default=30, type=int)
    parser.add_argument("--vocab_size", default=32000, type=int)
    parser.add_argument("--tokenizer_path", default="./tokenizer/default_tokenizers/mecab_bpe_deid_32k")

    parser.add_argument("--raw_dataset_dir", 
                        default="./datasets/0_raw_documents")
    parser.add_argument("--replace_subset_dataset_dir", 
                        default="./datasets/1_replacement/subset")
    parser.add_argument("--replace_addr_dataset_dir", 
                        default="./datasets/1_replacement/address")
    parser.add_argument("--replace_map_path", 
                        default="./datasets/replace_map.json")
    # (output of dataset generation)
    parser.add_argument("--dataset_save_root", 
                        default=f"./datasets/generated")
    parser.add_argument("--modified_tokenizer_save_root", 
                        default=f"./tokenizer")
    args = parser.parse_args()    



    def get_all_json_keys(dataset_root):
        directory = f"{dataset_root}/1_replacement/subset"
        all_keys = set()  # Using a set to avoid duplicate keys
        
        # Walk through directory and subdirectories
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.endswith('.json') and filename != 'all_categories.json':
                    file_path = os.path.join(root, filename)
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            # Assuming JSON contains a dictionary at the root
                            all_keys.update(data.keys())
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"Error reading {file_path}: {e}")
        
        return list(all_keys)

    # replacement labels    
    replace_labels = get_all_json_keys("./datasets")

    # get_replace_labels(labels, replace_labels)
    generated_labels = ["내국인이름", 
                        "주민등록번호", "계좌번호", "신용카드번호",
                        "연령정보",
                        "도아래주소", "시아래주소", "군아래주소", "구아래주소", '읍아래주소', "주소",
                        "전화번호", "휴대폰번호",
                        "동", "수용동", "수용실", "호", "실", "호실", "객실",
                        "출구", "출구번호", "진료실", "생활관", "룸", "승강장", "번", "번호",
                        "방", "호", "호실", "객실번호", "실", 
                        "탑승장번호", "승강장번호", "승차장",
                        "버스", "버스번호", "광역버스", "노선번호",
                        "호선", "층", "명수", "레일",
                        "열차번호", "차량번호", "택시", "화물차",
                        "연도"
                        ]
    choice_labels = [
        '매장', '사회복지시설', '식품업체', '사무소', '위치기준', 
        '외국인이름', '해외하천', '지부', '작업실', '사무실', '사업체'
    ]
    all_labels = sorted(set(replace_labels + generated_labels + choice_labels + ['O']))
    print(f"[INFO] number of all labels : {len(all_labels)}")


    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    # save directory
    vocab_size_str = f"{int(args.vocab_size/1000)}"
    dataset_path = f"{args.dataset_save_root}/{vocab_size_str}k-n{args.num_replica}-seed{args.seed}"
    modified_tokenizer_path = f"{args.modified_tokenizer_save_root}/modified-tokenizer-{vocab_size_str}k"

    if not os.path.exists(args.dataset_save_root):
        os.makedirs(args.dataset_save_root)   
    if not os.path.exists(args.modified_tokenizer_save_root):
        os.makedirs(args.modified_tokenizer_save_root)   

    # processing
    raw_dataset, tokenizer = generate_dataset_with_modified_tokenizer(
        tokenizer=tokenizer,
        seed=args.seed,
        num_replica=args.num_replica,
        raw_dataset_dir=args.raw_dataset_dir,
        replace_subset_dataset_dir=args.replace_subset_dataset_dir,
        replace_addr_dataset_dir=args.replace_addr_dataset_dir,
        replace_map_path=args.replace_map_path,
        dataset_path=dataset_path, # save path
        modified_tokenizer_path=modified_tokenizer_path # save path
    )    
    return





if __name__ == "__main__":
    main()