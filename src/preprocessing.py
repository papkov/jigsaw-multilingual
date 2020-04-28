import transformers
import numpy as np
import pandas as pd
import os
import time

def tokenize(texts, tokenizer, max_length=512):
    return np.array(tokenizer.batch_encode_plus(texts, max_length=max_length, return_attention_masks=False, pad_to_max_length=True)['input_ids'])


def read_tok_save(fn: str, tokenizer):
    """
    Read, tokenize and save as .npz tokenized texts and labels
    """
    df = pd.read_csv(fn)
    tokens = np.array(tokenize(df.iloc[:,1].tolist(), tokenizer))
    labels = np.array(df.toxic > 0.5, dtype=np.uint8) if 'toxic' in df.columns else np.empty(len(df))  # Threshold toxicity for unintended bias
    to_save = dict(x=tokens, y=labels)
    if 'lang' in df.columns:
        to_save['lang'] = df.lang
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