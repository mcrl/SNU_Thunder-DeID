import json
import os
import glob

root = "/shared/s1/lab08/sungeun/deid2/ks-workspace/1_replacement/subset"
def main():
    
    all_categories = dict()
    
    files1 = os.listdir(root)
    cur_dir = root
    cur_dict = all_categories
    files = os.listdir(root)
    append_keys(cur_dir, cur_dict)
    
    print(all_categories)
    with open("all_categories.json", "w") as fw:
        json.dump(all_categories, fw, ensure_ascii=False, indent=4)
            
    return


def append_keys(cur_dir, cur_dict):
    files = os.listdir(cur_dir)
    # base
    if len(files) < 1:
        return
    
    # recursive
    for filename in files:
        if filename.endswith(".json"):
            if "목록" not in cur_dict.keys():
                cur_dict["목록"] = list()
            with open(f"{cur_dir}/{filename}", 'r') as fr:
                data = json.load(fr)
            keys = data.keys()
            for key in data.keys():
                cur_dict["목록"].append(key)
        
        elif filename.endswith(".py"):
            continue
        
        else:                
            if filename not in cur_dict:
                cur_dict[filename] = dict()
            new_dir = f"{cur_dir}/{filename}"
            append_keys(
                cur_dir = new_dir,
                cur_dict = cur_dict[filename])



if __name__ == "__main__":
    main()