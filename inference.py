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


    # de-identify input data using model outputs
    deidentified = []
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
                    # entity -> tokens to text
                    cur_entity = adjust_spacing(
                            tokenizer.decode(current_entity_tok)).lstrip().rstrip()
                    entity_exists = False

                    for key in entity_dict.keys(): # alphabet
                        if entity_dict[key]["entity"] == cur_entity:
                            # same entity
                            for pos in current_entity_pos:
                                cur_document[pos] = tokenizer.convert_tokens_to_ids(key) # same alphabet
                            entity_exists = True
                            break
                    
                    if not entity_exists:
                        entity_dict[alphabet] = dict()
                        entity_dict[alphabet]["tokens"] = current_entity_tok
                        entity_dict[alphabet]["entity"] = cur_entity
                        # entity_dict[alphabet]["label"] = current_label

                        for pos in current_entity_pos:
                            cur_document[pos] = tokenizer.convert_tokens_to_ids(alphabet)

                    # reset
                    current_entity_tok = []
                    current_entity_pos = []

            else:
                if entity_begins == False:
                    alphabet = get_next_alphabet(seen_alphabets)
                    seen_alphabets.add(alphabet)
                    entity_begins = True
                    # current_label = tokenizer.id2label[pred_tok]

                current_entity_pos.append(pos_idx)
                current_entity_tok.append(tok)
        

        deidentified.append(cur_document)

    deidentified = [tokenizer.decode(
        doc, 
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    ) for doc in deidentified]
    deidentified = [
        post_process_fix_spacing(
            clean_repeated_alphabets(
                adjust_spacing(doc)))
                        for doc in deidentified]

    
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


if __name__ == "__main__":
    main()
