import transformers
import numpy as np
import pandas as pd
import os
import time
import re
import nltk
nltk.download('punkt')
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=2, progress_bar=False)

LANGS = {
    'en': 'english',
    'it': 'italian', 
    'fr': 'french', 
    'es': 'spanish',
    'tr': 'turkish', 
    'ru': 'russian',
    'pt': 'portuguese'
}


def get_sentences(text, lang='en'):
    return nltk.sent_tokenize(text, LANGS.get(lang, 'english'))


def exclude_duplicate_sentences(text, lang='en'):
    """https://www.kaggle.com/shonenkov/tpu-inference-super-fast-xlmroberta"""
    sentences = []
    for sentence in get_sentences(text, lang):
        sentence = sentence.strip()
        if sentence not in sentences:
            sentences.append(sentence)
    return ' '.join(sentences)


def clean_text(text, lang='en'):
    """https://www.kaggle.com/shonenkov/tpu-inference-super-fast-xlmroberta"""
    text = str(text)
    text = re.sub(r'[0-9"]', '', text)
    text = re.sub(r'#[\S]+\b', '', text)
    text = re.sub(r'@[\S]+\b', '', text)
    text = re.sub(r'https?\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = exclude_duplicate_sentences(text, lang)
    return text.strip()


def tokenize(texts, tokenizer, max_length=512):
    tokenized = tokenizer.batch_encode_plus(texts, max_length=max_length, return_attention_masks=True, pad_to_max_length=True, add_special_tokens=True)
    return np.array(tokenized['input_ids']), np.array(tokenized['attention_mask'])


def read_tok_save(fn: str, tokenizer):
    """
    Read, tokenize and save as .npz tokenized texts and labels
    """
    # Read and clean
    df = pd.read_csv(fn)
    columns = df.columns
    text_column = 'comment_text' if 'comment_text' in columns else 'content'
    df[text_column] = df.parallel_apply(lambda x: clean_text(x[text_column], x['lang'] if 'lang' in columns else 'en'), axis=1)
    
    # Process and combine
    input_ids, attention_mask = tokenize(df[text_column].tolist(), tokenizer)
    labels = np.array(df['toxic'] > 0.5, dtype=np.uint8) if 'toxic' in df.columns else np.empty(len(df))  # Threshold toxicity for unintended bias
    to_save = dict(x=input_ids, y=labels, attention_mask=attention_mask)
    if 'lang' in columns:
        to_save['lang'] = df.lang
    
    # Save
    fn = fn.rstrip('csv')+'npz'
    print(f'Saving to {fn} dict with {to_save.keys()}')
    np.savez(fn, **to_save)

    
def read_tok_save_all_roberta(files: list = ['jigsaw-toxic-comment-train.csv',
                                             'jigsaw-unintended-bias-train.csv',
                                             'validation.csv',
                                             'test.csv',
                                            ],
                              path: str = '../input/',
                              tokenizer_name='xlm-roberta-large'):
    """
    Read, tokenize and save all the files with XLM Roberta tokenizer
    """
    tokenizer = transformers.XLMRobertaTokenizer.from_pretrained(tokenizer_name)
    for fn in files:
        s = time.time()
        print('Processing', fn)
        read_tok_save(os.path.join(path, fn), tokenizer)
        print(f'Finished in {time.time()-s} s\n')