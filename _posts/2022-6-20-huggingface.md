---
layout:     post
title:      Huggingface 库的一些细节用法
subtitle:    "经常用，放在这查方便"
date:       2022-6-20
author:     Chen Zhuo
header-img: img/2022-6-20.png
catalog: true
tags:
    - Notes
---



# Tokenizer

~~~python
from transformer import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # 这里用 bert-base-uncase 举例
~~~

有时候需要对输入句子进行修改，然后再送进 BERT，就需要这个东西。

## tokenizer.tokenize()

~~~python
tokenizer.tokenize('i love nlp')
~~~

会把句子 tokenize 成 bert 的词表表示序列：

~~~python
['i', 'love', 'nl', '##p']
~~~

## tokenizer.encode()

~~~python
tokenizer.encode(['i', 'love', 'nl', '##p'])
~~~

会把序列变为 bert 词表中 token 的 index，同时加入 101-[CLS]，103-[SEP] 的 index

~~~python
[101, 1045, 2293, 17953, 2361, 102]
~~~

## tokenizer.decode()

~~~python
tokenizer.decode([101, 1045, 2293, 17953, 2361, 102])
~~~

把 index 序列变回句子，这个变化是前后一致的，即使句子中有 sub-token，比如 ##p 这种

~~~python
'[CLS] i love nlp [SEP]'
~~~

如果不想要 [CLS] [SEP]，就把 index 列表掐头去尾。