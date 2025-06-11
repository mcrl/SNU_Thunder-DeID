import json

def main():
    
    name1 = "군청"
    name2 = "행정군"
    with open(f"{name1}.json", 'r') as fr:
        data = json.load(fr)
      
    data = data[name1]
    new_data =  {name2:[item.replace(name1, '') for item in data]}
    
    with open(f"{name2}.json", 'w') as fw:
        json.dump(new_data, fw, ensure_ascii=False, indent=4)
        
    
    
    return


if __name__ == "__main__":
    main()