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


    deidentified = [tokenizer.decode(doc, skip_special_tokens=True, clean_up_tokenization_spaces=True) 
                    for doc in deidentified]
    deidentified = [adjust_spacing(d) for d in deidentified]
    deidentified = [clean_repeated_alphabets(d) for d in deidentified]
    deidentified = [post_process_fix_spacing(d) for d in deidentified]
    deidentified = [fix_josa_with_entity_dict(text=d,entity_dict=entity_dict) for d in deidentified]
    deidentified = [handling_address(d, phase="output") for d in deidentified]

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
    target_dict = {
        '( ':'(', ' )':')',
        '[ ':'[', ' ]':']',
        '{ ':'{', ' }':'}',
        '" ':'"', ' "':'"',
        "' ":"'", " '":"'",
        '‘ ':'‘', ' ’':'’',
        '“ ':'“', ' ”':'”',
    }
    for replace_from, replace_to in target_dict.items():
        text = text.replace(replace_from, replace_to)
    
    return text


def fix_josa_with_entity_dict(text, entity_dict):
    batchim_alphabets = ["L", "M", "N", "R"]  # 받침 있는 알파벳

    # 받침 유무에 따라 바뀌는 조사들 + 뒤에 반드시 띄어쓰기 또는 종결을 포함
    josa_type_A = {'이라는 ': '라는 ', '은 ': '는 ', '이 ': '가 ', '을 ': '를 ', '과 ': '와 '}
    josa_type_B = {'라는 ': '이라는 ', '는 ': '은 ', '가 ': '이 ', '를 ': '을 ', '와 ': '과 '}
    all_josa = sorted(set(josa_type_A.keys()) | set(josa_type_B.keys()), key=len, reverse=True)

    for alphabet, value in entity_dict.items():
        next_chars = value.get("next_chars", "")
        if not next_chars or value.get("josa_adjusted", False):
            continue

        next_josa = None
        for josa in all_josa:
            if next_chars.startswith(josa):
                next_josa = josa
                break

        if not next_josa:
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


def handling_address(text, phase="input"):
    target_dict = {
        "서울특별시 ":"서울시 ",
        "인천광역시 ":"인천시 ",
        "부산광역시 ":"부산시 ",
        "대구광역시 ":"대구시 ",
        "대전광역시 ":"대전시 ",
        "울산광역시 ":"울산시 ",
        "광주광역시 ":"광주시 ",
        "세종특별자치시 ":"세종시 ",
    }
        
    if phase == "input":
        for replace_from, replace_to in target_dict.items():
            text = text.replace(replace_from, replace_to)   
    
    elif phase == "output":
        for replace_to, replace_from in target_dict.items():
            text = text.replace(replace_from, replace_to)
    
    return text
            
            
        


def load_local_moel(model_size, tokenizer, num_labels=595):
    cfg = AutoConfig.from_pretrained(f"./config/models/ours-{model_size.lower().replace('.', '_')}.json")
    cfg.max_position_embeddings = 2048
    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.vocab_size = len(tokenizer)
    cfg.num_labels = num_labels
    
    model = AutoModelForTokenClassification.from_config(cfg)
        
    # 상대 위치 임베딩 크기 조정
    desired_rel_pos_size = 2048
    current_rel_pos_size = model.deberta.encoder.rel_embeddings.weight.shape[0]
    embedding_dim = model.deberta.encoder.rel_embeddings.weight.shape[1]
    
    if current_rel_pos_size != desired_rel_pos_size:
        # 새로운 rel_embeddings 레이어 생성
        new_rel_embeddings = torch.nn.Embedding(desired_rel_pos_size, embedding_dim)
        with torch.no_grad():
            # 기존 가중치 복사
            new_rel_embeddings.weight[:current_rel_pos_size] = model.deberta.encoder.rel_embeddings.weight
            # 새로운 위치에 대해 초기화
            new_rel_embeddings.weight[current_rel_pos_size:] = torch.randn(
                desired_rel_pos_size - current_rel_pos_size, embedding_dim
            ) * 0.02  # 작은 분산으로 초기화
        model.deberta.encoder.rel_embeddings = new_rel_embeddings
    
    # 모델 설정에서 max_relative_positions 업데이트
    if hasattr(model.config, 'max_relative_positions'):
        model.config.max_relative_positions = desired_rel_pos_size
    
    # 최대 시퀀스 길이 확인 및 업데이트
    if model.config.max_position_embeddings < desired_rel_pos_size:
        model.config.max_position_embeddings = desired_rel_pos_size


    if model_size == "340M":
        model_weight_path = NEW340M
    elif model_size == "1.5B":
        model_weight_path = NEW1_5B
        
    
    model_weight = torch.load(model_weight_path,
                            map_location="cpu",
                            weights_only=False)["module"]
    model.load_state_dict(model_weight, strict=False)        
    print(f"[INFO] model weight successfully loadded from {model_weight_path}")
    
    return model



if __name__ == "__main__":
    main()
