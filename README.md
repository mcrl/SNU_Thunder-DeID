# SNU Thunder-DeID

**SNU Thunder-DeID** is a de-identification project that includes models, high-quality datasets, and an inference tool for Named Entity Recognition (NER)-based anonymization of Korean court judgments.  
**This repository** provides a standalone inference tool that runs **SNU Thunder-DeID** models on raw text inputs.  
It detects named entities at the token level and replaces them with anonymized placeholders (e.g., A, B, ...) to ensure consistency across mentions.

---

## Installation 

```bash
# install mecab-ko tokenizer
cd ~
mkdir mecab

wget https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
tar zxfv mecab-0.996-ko-0.9.2.tar.gz
cd mecab-0.996-ko-0.9.2/
./configure --prefix=$HOME/mecab
make
make install

# install mecab-ko-dictionary
wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
tar zxvf mecab-ko-dic-2.1.1-20180720.tar.gz
cd mecab-ko-dic-2.1.1-20180720/
./configure --prefix=$HOME/mecab --with-mecab-config=$HOME/mecab/bin/mecab-config
./autogen.sh
make
make install

pip install mecab-python3 konlpy

# clone this repository
cd ~
git clone https://github.com/mcrl/SNU_Thunder-DeID

# create new environment
conda create -n snu_thunder_deid python=3.13 -y
conda activate snu_thunder_deid

# install required packages
pip3 install torch torchvision torchaudio
pip3 install transformers datasets pandas

# install package for our custom tokenizer (mecab-ko + bpe)
cd ./SNU_Thunder-DeID/tokenizer/mecab_bpe
pip install .

# return to SNU_Thunder-DeID directory
cd ../..
```

## How to Use : Inference
2. Run inference:

```bash
python inference.py \
  --model_size 340M \
  --device cuda \
  --input_type text \
  --input "피고인 이규성은 서울대학교 데이터사이언스대학원에 박사과정으로 재학 중인 대학원생이며..." \
  --output_file ./output.csv \
  --print_output
```

or simply test with,

```bash
python inference.py --print_output
```

---

## How to Use : Training dataset generation
2. Run inference:

```bash
python gen_dataset.py \
  --seed 1203 \
  --num_replica 30
```

---


## Example

### Input
```text
피고인 **이규성**은 **서울대학교 데이터사이언스**대학원 박사과정에 재학 중이며, 같은 연구실 소속 **함성은**, **박현지**와 함께 AI 모델 비식별화와 관련된 연구를 진행 중이다. 그는 해당 기술이 이미 여러 공공기관 및 대기업으로부터 상용화 제안을 받고 있다고 허위로 주장하며, 커뮤니티 사이트 ‘**에브리타임**’에 “비식별화 기술 투자자 모집”이라는 제목의 글을 게시하였다. 해당 글에는 “이미 검증된 알고리즘, 선점 투자 시 지분 우선 배정”, “특허 수익 배분 예정” 등의 문구와 함께 자신 명의의 **우리은행** 계좌 (**9429-424-343942**)를 기재하고, 1인당 10만 원의 초기 투자금을 요구하였다. 이에 따라 **이규성**은 **손영준**, **조경제**, **이동영**, **소연경**, **석지헌** 등 5명으로부터 총 50만 원을 송금받아 편취하였다.
```

### Output
```text
피고인 **A**은 **B**대학원 박사과정에 재학 중이며, 같은 연구실 소속 **C**, **D**와 함께 AI 모델 비식별화와 관련된 연구를 진행 중이다. 그는 해당 기술이 이미 여러 공공기관 및 대기업으로부터 상용화 제안을 받고 있다고 허위로 주장하며, 커뮤니티 사이트 ‘**E**’에 “비식별화 기술 투자자 모집”이라는 제목의 글을 게시하였다. 해당 글에는 “이미 검증된 알고리즘, 선점 투자 시 지분 우선 배정”, “특허 수익 배분 예정” 등의 문구와 함께 자신 명의의 **F** 계좌 (**G**)를 기재하고, 1인당 10만 원의 초기 투자금을 요구하였다. 이에 따라 **A**은 **I**, **J**, **K**, **L**, **M** 등 5명으로부터 총 50만 원을 송금받아 편취하였다.
```

---

## 🔗 Related Resources

- **Models**:
  - [SNU_Thunder-DeID-340M](https://huggingface.co/thunder-research-group/SNU_Thunder-DeID-340M)
  - [SNU_Thunder-DeID-750M](https://huggingface.co/thunder-research-group/SNU_Thunder-DeID-750M)
  - [SNU_Thunder-DeID-1.5B](https://huggingface.co/thunder-research-group/SNU_Thunder-DeID-1.5B)

- **Training Datasets**:
  - [snu_deid_annotated_court judgments](https://huggingface.co/datasets/thunder-research-group/SNU_Thunder-DeID-annotated_court_judgments)  
    – NER-annotated court judgment data with placeholders
  - [snu_deid_list of entity_mentions](https://huggingface.co/datasets/thunder-research-group/SNU_Thunder-DeID-list_of_entity_mentions)  
    – Entity span and label mappings for full supervision dataset generation

---


## Citation

If you use this repository or the associated models/datasets, please cite the following:

to be updated


## License

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]  

Unless otherwise stated, this repository is licensed under a  
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].  
[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

### Attribution Notice

This dataset includes or is derived from content originally published in the  
[`lbox-open`](https://huggingface.co/datasets/lbox/lbox-open) dataset,  
available under the [CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/).  

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/  
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png  
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg  



## Contact

If you have any questions, issues, or collaboration inquiries, please contact: [snullm@aces.snu.ac.kr](mailto:snullm@aces.snu.ac.kr)
