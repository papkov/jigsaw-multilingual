import transformers
import numpy as np
import pandas as pd
import os
import time
import re
import nltk
nltk.download('punkt')

try:
    from pandarallel import pandarallel
    pandarallel.initialize(nb_workers=2, progress_bar=False)
    use_pandarallel = True
except ModuleNotFoundError:
    print('padnarallel not installed')    
    use_pandarallel = False



from util import tqdm_loader
from copy import deepcopy
import dataset
import torch.utils.data as D
from tqdm import tqdm
import torch

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

    # Some hand-crafted rules based on manual inspection
    # text = re.sub(r'n00b', r'noob', text)
    # text = re.sub(r'l33t', r'leet', text)
    # text = re.sub(r'1337', r'leet', text)
    # text = re.sub(r'[Ff][uU\*][cC\*][Kk\*]', r'fuck', text)
    # text = re.sub(r'[Jj][eE\*][sS\*][uU\*][sS\*]', r'jesus', text)
    # text = re.sub(r'[Bb][iI\*][tT\*][cC\*][hH\*]', r'bitch', text)
    # text = re.sub(r'[Cc][uU\*][nN\*][tT\*]', r'cunt', text)
    # text = re.sub(r'[Ss][hH\*][iI\*][tT\*]', r'shit', text)
    # text = re.sub(r'[Dd][aA\*][mM\*][nN\*]', r'damn', text)
    # text = re.sub(r'[aA][sS\*][sS\*]', r'ass', text)
    # text = re.sub(r'[aA][sS\*][sS\*][hH\*][oO\*][lL\*][eE\*]', r'asshole', text)
    # text = re.sub(r'[\@\#\$\%\^]{3,}', r'fuck', text) # questionable
    # text = re.sub(r' \*\*\*\* ', r' fuck ', text) # questionable
    # text = re.sub(r'\*{4,}', r'', text) # weird stars
    # text = re.sub(r'(\w)\1{3,}', r'\1', text) # letters repeating 3+ times

    # text = re.sub(r'[0-9"]', '', text)
    text = re.sub(r'#[\S]+\b', '', text)
    text = re.sub(r'@[\S]+\b', '', text)
    text = re.sub(r'https?\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = exclude_duplicate_sentences(text, lang)
    return text.strip()


def tokenize(texts, tokenizer, max_length=512):
    tokenized = tokenizer.batch_encode_plus(texts, max_length=max_length, return_attention_masks=True, pad_to_max_length=True, add_special_tokens=True)
    return np.array(tokenized['input_ids']), np.array(tokenized['attention_mask'])


def read_tok_save(fn: str, tokenizer, save_here=False):
    """
    Read, tokenize and save as .npz tokenized texts and labels
    """
    # Read and clean
    df = pd.read_csv(fn)
    columns = df.columns
    text_column = 'comment_text' if 'comment_text' in columns else 'content'
    if use_pandarallel:
        df[text_column] = df.parallel_apply(lambda x: clean_text(x[text_column], x['lang'] if 'lang' in columns else 'en'), axis=1)
    else:
        df[text_column] = df.apply(lambda x: clean_text(x[text_column], x['lang'] if 'lang' in columns else 'en'), axis=1)
    
    # Process and combine
    input_ids, attention_mask = tokenize(df[text_column].tolist(), tokenizer)
    labels = np.array(df['toxic'] > 0.5, dtype=np.uint8) if 'toxic' in df.columns else np.empty(len(df))  # Threshold toxicity for unintended bias
    lang = df.lang.tolist() if 'lang' in columns else 'en'

    # Free up memory
    del df
    to_save = dict(x=input_ids, y=labels, attention_mask=attention_mask, lang=lang)
    
    # Save
    fn = fn.replace('csv','npz')
    if save_here:
        fn = fn.split('/')[-1]
    print(f'Saving to {fn} dict with {to_save.keys()}')
    np.savez(fn, **to_save)

    
def read_tok_save_all_roberta(files: list = ['validation.csv',
                                             'test.csv',
                                             'jigsaw-toxic-comment-train.csv',
                                             'jigsaw-unintended-bias-train.csv',
                                            ],
                              path: str = '../input/',
                              tokenizer_name='xlm-roberta-large',
                              save_here=False,
                              ):
    """
    Read, tokenize and save all the files with XLM Roberta tokenizer
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    for fn in files:
        s = time.time()
        print('Processing', fn)
        read_tok_save(os.path.join(path, fn), tokenizer, save_here=save_here)
        print(f'Finished in {time.time()-s} s\n')


def extract_features(backbone, loader, device):
    """
    Extract pre-pooling features from `model` from `loader` data
    """
    features = []
    backbone = backbone.to(device)
    try:
        with torch.no_grad():
            for i, batch in tqdm_loader(loader, desc='feature extraction'):
                x, _ = backbone(input_ids=batch[0].to(device), attention_mask=batch[2].to(device))
                # two poolings for feature extraction
                pool_avg = torch.mean(x, 1)
                pool_max, _ = torch.max(x, 1)
                cls_token = x[:,0,:]
                x = torch.cat((cls_token, pool_avg, pool_max), 1)
                x = x.cpu().numpy()
                # append features
                features.append(deepcopy(x))
    except KeyboardInterrupt:
        tqdm.write('Interrupted')
    features = np.concatenate(features)
    return features


def extract_roberta_features_to_file(fn, device, backbone=None, batch_size=128, num_workers=8):
    """
    Run `extract_features` for file `fn` and saves the result in identical .npz under different name
    """
    # Load dataset
    ds = dataset.Dataset(fn)
    # Extract features
    loader = D.DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    backbone = transformers.AutoModel.from_pretrained('xlm-roberta-large') if backbone is None else backbone
    features = extract_features(backbone, loader, device)
    # Replace sentenses with respective features in the dataset .npz, save
    np.save(fn.replace('.npz', '')+'_roberta_features.npy', features)

