import os
from tokenizers import Tokenizer, NormalizedString, Regex
from tokenizers.pre_tokenizers import *
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from konlpy.tag import Mecab as MECAB
import json
import regex as re

class MecabPreTokenizer:
    def __init__(self, dic_path='', rc_path=''):
        HOME=os.environ['HOME']
        if not dic_path:
            dic_path = f"{HOME}/mecab/lib/mecab/dic/mecab-ko-dic"
        if not rc_path:
            rc_path = f"{HOME}/mecab/etc/mecabrc"

        self.mecab = MECAB(f"{dic_path} -r {rc_path}")
        self.pattern = r"(?: ?\p{L}+)+(?:[.?!])?"

    def mecab_split(self, i, normalized_string):
        normalized_str = str(normalized_string)
        splits = []
        
        if re.fullmatch(self.pattern, normalized_str) is not None:
            for word in re.findall(r'\s?[\p{L}.?!]+', normalized_str):
                morphs = self.mecab.morphs(word)
                
                if word.startswith(' '):
                    morphs[0] = ' ' + morphs[0]
                splits.extend(morphs)

            output = [NormalizedString(s) for s in splits]
            return output
        else:
            return [normalized_string]

    def pre_tokenize(self, pretok):
        return pretok.split(self.mecab_split)

    def pre_tokenize_str(self, string):
        output = self.mecab_split(0, string)
        return str(output)

    def to_json(self):
        return json.dumps({"type":"Whitespace"})

PRETOK_MAP = {
    'BertPreTokenizer': BertPreTokenizer,
    'ByteLevel': ByteLevel,
    'Whitespace': Whitespace,
    'CharDelimiterSplit': CharDelimiterSplit,
    'Digits': Digits,
    'Metaspace': Metaspace,
    'Punctuation': Punctuation,
    'UnicodeScripts': UnicodeScripts,
    'Split': Split,
    'WhitespaceSplit': WhitespaceSplit,
}

def convert_pretok(pre_tokenizer):
    state = pre_tokenizer.__getstate__()
    state = json.loads(state)
    results = []
    if state['type'] == 'Sequence':
        for substate in state['pretokenizers']:
            ptype = substate.pop('type')
            if ptype == "ByteLevel":
                substate['add_prefix_space'] = True
            if ptype == "Metaspace":
                substate['prepend_scheme'] = 'first'
            results.append(PRETOK_MAP[ptype](**substate))
    else:
        ptype = state.pop('type')
        if ptype == "ByteLevel":
            state['add_prefix_space'] = True
        if ptype == "Metaspace":
            state['prepend_scheme'] = 'first'
        results.append(PRETOK_MAP[ptype](**state))

    return results

def setup_for_mecab():
    def __new__(self):
        print('new')
        return PreTokenizer.custom(MecabPreTokenizer())

    def __setstate__(self, state):
        print('setstate')
        self.__dict__.update(state)
        st = self._tokenizer.pre_tokenizer.__getstate__()
        def load_pretok(pt_state):
            def get_pretok(ptype, p):
                if ptype == "Dummy":
                    return PreTokenizer.custom(MecabPreTokenizer())
                elif ptype in PRETOK_MAP:
                    if 'pattern' in p and 'Regex' in p['pattern']:
                        p['pattern'] = Regex(p['pattern']['Regex'])
                    if 'behavior' in p:
                        p['behavior'] = p['behavior'].lower()
                    return PRETOK_MAP[ptype](**p)
                else:
                    raise ValueError()

            data = json.loads(pt_state)
            pt_type = data.pop('type')
            if pt_type == "Sequence":
                pretoks = []
                for p in data['pretokenizers']:
                    ptype = p.pop('type')
                    pretoks.append(get_pretok(ptype, p))
                return Sequence(pretoks)
            else:
                return get_pretok(pt_type, data)

        self._tokenizer.pre_tokenizer = load_pretok(st)


    setattr(PreTokenizer, '__new__', __new__)
    setattr(PreTokenizer, '__setstate__', lambda self, state: None)
    setattr(PreTrainedTokenizerBase, '__setstate__', __setstate__)

def add_mecab_pretok(tokenizer, dic_path='', rc_path=''):
    if isinstance(tokenizer, Tokenizer):
        orig_pre_tokenizer = tokenizer.pre_tokenizer
        tokenizer.pre_tokenizer = Sequence([PreTokenizer.custom(MecabPreTokenizer()), *convert_pretok(orig_pre_tokenizer)])
    elif isinstance(tokenizer, PreTrainedTokenizerBase):
        orig_pre_tokenizer = tokenizer._tokenizer.pre_tokenizer
        tokenizer._tokenizer.pre_tokenizer = Sequence([PreTokenizer.custom(MecabPreTokenizer()), *convert_pretok(orig_pre_tokenizer)])
    else:
        raise ValueError("Not Support this type of tokenizer")

    return tokenizer

def switch_dummy(tok):
    if not hasattr(tok._tokenizer.pre_tokenizer, '__getstate__'):
        return tok
    st = tok._tokenizer.pre_tokenizer.__getstate__()
    def load_pretok(pt_state):
        def get_pretok(ptype, p):
            if ptype == "Whitespace":
                return PreTokenizer.custom(MecabPreTokenizer())
            elif ptype in PRETOK_MAP:
                # print(ptype, p)
                if 'pattern' in p and 'Regex' in p['pattern']:
                    p['pattern'] = Regex(p['pattern']['Regex'])
                if 'behavior' in p:
                    p['behavior'] = p['behavior'].lower()
                # print(ptype, p)
                return PRETOK_MAP[ptype](**p)
            else:
                raise ValueError(ptype)

        data = json.loads(pt_state)
        pt_type = data.pop('type')
        if pt_type == "Sequence":
            pretoks = []
            for p in data['pretokenizers']:
                ptype = p.pop('type')
                pretoks.append(get_pretok(ptype, p))
            return Sequence(pretoks)
        else:
            return get_pretok(pt_type, data)

    tok._tokenizer.pre_tokenizer = load_pretok(st)
    return tok
