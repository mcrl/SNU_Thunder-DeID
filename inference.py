import argparse
import os
import re
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig, PreTrainedTokenizerFast
from datasets import load_dataset

# test
import pickle

# private
import custom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", default="340M", choices=["340M", "750M", "1.5B"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--input", default="""피고인 이규성은 서울대학교 데이터사이언스대학원 박사과정에 재학 중이며, 같은 연구실 소속 함성은, 박현지와 함께 AI 모델 비식별화와 관련된 연구를 진행 중이다. 그는 해당 기술이 이미 여러 공공기관 및 대기업으로부터 상용화 제안을 받고 있다고 허위로 주장하며, 커뮤니티 사이트 ‘에브리타임’에 “비식별화 기술 투자자 모집”이라는 제목의 글을 게시하였다. 해당 글에는 “이미 검증된 알고리즘, 선점 투자 시 지분 우선 배정”, “특허 수익 배분 예정” 등의 문구와 함께 자신 명의의 우리은행 계좌 (9429-424-343942)를 기재하고, 1인당 10만 원의 초기 투자금을 요구하였다. 이에 따라 이규성은 손영준, 조경제, 이동영, 소연경, 석지헌 등 5명으로부터 총 50만 원을 송금받아 편취하였다.""")
    parser.add_argument("--output_file", default="./output.csv")
    parser.add_argument("--print_output", action="store_true")

    parser.add_argument("--input_type", default="text", choices=["text", "file"])
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_replica", default=30, type=int)
    parser.add_argument("--seed", default=1203, type=int)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Cuda device is not available: CPU inference")
        args.device = "cpu"    
    
    if args.model_size == "1.5B":
        tokenizer_path = "./tokenizer/default_tokenizers/mecab_bpe_deid_128k"
    else:
        tokenizer_path = "./tokenizer/default_tokenizers/mecab_bpe_deid_32k"
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    tokenizer = custom.switch_dummy(tokenizer)
    
    model = AutoModelForTokenClassification.from_pretrained(
        f"thunder-research-group/SNU_Thunder-DeID-{args.model_size}",
        trust_remote_code=True,
    ).to(args.device)
    
    # input type
    if args.input_type == "file":
        if args.input.endswith(".csv"):
            input_data = pd.read_csv(args.input)[0] # no column names
        
        with open(args.input, 'r') as fr:
            input_data = fr.read()
            input_data = [input_data]
        
    else:
        input_data = args.input
        input_data = [input_data]
    
    # model inputs
    model_inputs = [
        torch.tensor(
            tokenizer.encode(
                text,
                padding='max_length',
                truncation=True,
                max_length=2048,
                return_tensors="pt",
                add_special_tokens=True  
            )
        ) for text in input_data
    ]
    model_inputs = torch.stack(model_inputs).squeeze(1)

    # model outputs
    model_outputs = []
    for idx in range(0, len(model_inputs), args.batch_size):
        if args.batch_size == -1:
            args.batch_size = len(model_outputs)

        cur_input = model_inputs[idx:idx+args.batch_size].to(args.device)
        attention_mask = (cur_input != tokenizer.pad_token_id).long().to(args.device)
        
        with torch.no_grad():
            logits = model(cur_input, attention_mask=attention_mask).logits
        pred_labels = torch.argmax(logits, dim=-1).detach().cpu().tolist()
        model_outputs += pred_labels


    # de-identification
    deidentified = []
    all_seen_alphabets = set()

    for document_idx in range(len(model_inputs)):
        seen_alphabets = set()
        entity_begins = False

        cur_document = [tok for tok in model_inputs[document_idx]]
        entity_dict = dict()
        current_entity_tok = []
        current_entity_pos = []

        for pos_idx, (tok, pred_tok) in enumerate(zip(model_inputs[document_idx], model_outputs[document_idx])):
            # 1 : entity 아님. 'O' 라벨
            if pred_tok == 1:
                entity_begins = False
                if len(current_entity_tok) > 0:
                    cur_entity = adjust_spacing(tokenizer.decode(current_entity_tok)).strip()
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


    deidentified = [tokenizer.decode(doc, skip_special_tokens=True, clean_up_tokenization_spaces=True) for doc in deidentified]
    deidentified = [fix_josa_with_entity_dict(
        text=post_process_fix_spacing(clean_repeated_alphabets(adjust_spacing(deidentified_doc))), 
        entity_dict=entity_dict)
        for deidentified_doc in deidentified]

    # output files
    output_file = pd.DataFrame(deidentified)
    output_file.to_csv(args.output_file, index=False, encoding="cp949")

    if args.print_output:
        print()
        for idx, doc in enumerate(deidentified):
            print(f"Document {idx} deidentified: \n{doc}\n")

    return        


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
            if char not in seen_alphabets:
                return char

def clean_repeated_alphabets(text):
    text = re.sub(r'([A-Za-z]{2})\1+', r' \1', text)
    text = re.sub(r'([A-Za-z])\1+', r' \1', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text

def post_process_fix_spacing(text):
    text = re.sub(r"([\'\"\(\[\{\‘\“])\s+([A-Za-z0-9가-힣])", r"\1\2", text)
    text = re.sub(r"\s+([\'\"\)\]\}\’\”])", r"\1", text)
    return text

def fix_josa_with_entity_dict(text, entity_dict):
    batchim_alphabets = ["L", "M", "N", "R"]  

    josa_type_A = {'은':'는', '이':'가', '을':'를'} 
    josa_type_B = {'는':'은', '가':'이', '를':'을'} 
    all_josa = list(josa_type_A.keys()) + list(josa_type_B.keys())

    for alphabet, value in entity_dict.items():
        next_chars = value.get("next_chars", "")
        if not next_chars:
            continue

        next_josa = next_chars[0]  
        if next_josa not in all_josa:
            continue
        if value.get("josa_adjusted", False):
            continue

        replace_from = f"{alphabet}{next_josa}"

        if alphabet[-1] in batchim_alphabets:
            if next_josa in josa_type_B:
                replace_to = f"{alphabet}{josa_type_B[next_josa]}"
            else:
                continue
        else:
            if next_josa in josa_type_A:
                replace_to = f"{alphabet}{josa_type_A[next_josa]}"
            else:
                continue

        text = text.replace(replace_from, replace_to, 1)
        value["josa_adjusted"] = True

    return text

if __name__ == "__main__":
    main()
