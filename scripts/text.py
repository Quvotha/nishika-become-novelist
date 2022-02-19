import hashlib

import re
from typing import Tuple
import unicodedata

import MeCab


def hashfxn(x):
    # Reproducibility for Doc2Vec
    return int(hashlib.md5(str(x).encode()).hexdigest(), 16)


def normalize(text: str) -> str:
    text = re.sub(r'[\n|\r|\t]+', ' ', text)
    text = re.sub(r'[ |　]+', ' ', text)
    text = unicodedata.normalize('NFKC', text)
    text = text.lower()
    text = text.strip()
    return text


def tokenize_v1(
        tagger: MeCab.Tagger,
        text: str,
        normalized: bool = False,
        part_of_speech: Tuple[str] = ('動詞', '名詞', '形容詞', '助動詞', '副詞')) -> str:
    if not normalized:
        text = normalize(text)
    tokens = []
    parsed = tagger.parse(text)
    for line in parsed.splitlines()[:-1]:
        if not line:
            continue
        word, attributes = line.split('\t')
        p = attributes.split(',')[0]
        if p in part_of_speech:
            tokens.append(word)
    return ' '.join(tokens)


if __name__ == '__main__':
    tagger = MeCab.Tagger('-u dictionary/NEologd.20200910-u.dic')
    text = 'やっぱり幼馴染がいいそうです。 〜二年付き合った彼氏に振られたら、彼のライバルが迫って来て恋人の振りをする事になりました〜'
    print(tokenize_v1(tagger, text))
