# Graph Guided Attention for Relation Extraction
This code is for ...
## Requirements
1. Pytorch 1.0
2. Python 3.7
3. unzip, wget (for downloading glove only)
## TAC RED Dataset
The TACRED dataset: Details on the TAC Relation Extraction Dataset can be found on this [dataset website.](https://nlp.stanford.edu/projects/tacred/)

To respect the copyright of the underlying TAC KBP corpus, TACRED is released via the Linguistic Data Consortium (LDC). Therefore, you can download TACRED from the [LDC TACRED webpage](https://catalog.ldc.upenn.edu/LDC2018T24). If you are an LDC member, the access will be free; otherwise, an access fee of $25 is needed.

## Preparation
first, download and unzip GloVe word embedding with:
```
chmod +x download.sh; ./download.sh
```
Then prepare vocabulary and initial word vectors with:
```
python prepare_vocab.py dataset/tacred dataset/vocab 
```
This will write vocabulary and word vectors as a numpy matrix into the dir `dataset/vocab`.

## Training
```
chmod +x run.sh; ./run.sh
```

## Evaluation
```
python eval.py saved_model/00
```


