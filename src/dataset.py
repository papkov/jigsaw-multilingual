import os
import sys
import time
import pickle
import logging
from filelock import FileLock

from tqdm.auto import tqdm

import torch.utils.data as D
import torch

import pandas as pd
import numpy as np
import random

from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from preprocessing import tokenize, clean_text
from transforms import Compose, compose_transforms

from functools import reduce
from transformers import AutoTokenizer, XLMRobertaTokenizer, PreTrainedTokenizer, DataCollatorForLanguageModeling
from typing import Dict, List

logger = logging.getLogger(__name__)


def weighted_sampler(y, scale_factor=1, weights=None, tot_samples=None, replacement=False):
    """
    Returns WeightedRandomSampler which will sample
    approximately equal amount of positive and negative samples from y
    total number of samples = min(class count) * 2 * scale_factor
    (e.g. scale factor 0.5 will reduce epoch twice)
    """
    # if weights not provided, weight with inverse count
    labels, counts = np.unique(y, return_counts=True)
    if weights is None:
        weights = counts[::-1] / counts.sum()
    # create weight for each sample
    weights = np.array([weights[i] for i in y])
    if tot_samples is not None:
        scale_factor = tot_samples / counts.min() * 2
        print(f'tot_samples={tot_samples} overrides scale_factor={scale_factor}')
    return WeightedRandomSampler(weights, int(counts.min() * 2 * scale_factor), replacement=replacement)


class Dataset(D.Dataset):
    def __init__(self, fn, use_features=False, feature_slice=slice(0,3072), transforms=None):
        super().__init__()
        self.fn = fn
        self.use_features = use_features
        self.feature_slice = feature_slice
        self.dataset = np.load(fn, allow_pickle=True)

        self.features = None
        # TODO customize features?
        if use_features:
            try:
                features_fn = fn.replace('.npz', '')+'_roberta_features.npy'
                self.features = np.load(features_fn)[:,self.feature_slice]
            except:
                print(features_fn, 'loading failed. Set `self.features = None`, use tokenized input')

        self.x = self.features if (use_features and self.features is not None) else self.dataset['x']
        self.y = (self.dataset['y'] > 0.5).astype(np.uint8)
        self.n_classes = 2
        self.attention_mask = self.dataset['attention_mask']
        
        self.transforms = compose_transforms(transforms)
        
    def process_x(self, x):
        """What to do with x before batching (for augmentations)"""
        if self.transforms is not None:
            x = self.transforms(x)
        return x

    def process_y(self, y):
        """What to do with y before batching (for encoding)"""
        # one-hot encoding
        y = np.eye(self.n_classes, dtype=np.uint8)[y]
        return y
        
    def __getitem__(self, i):
        x, y, am = self.x[i], self.y[i], self.attention_mask[i]
        x = self.process_x(x)
        y = self.process_y(y)
        
        # typization
        x, y, am = map(torch.tensor, [x, y, am])
        x = x.float() if self.use_features else x.long()
        y = y.long()
        am = am.long()

        return x, y, am
    
    def __len__(self):
        return len(self.y)
    
    def weighted_sampler(self, *args, **kwargs):
        """
        Returns WeightedRandomSampler which will sample with repetition 
        approximately equal amount of positive and negative samples from self.y
        total number of samples = min(class count) * 2 * scale_factor
        (e.g. scale factor 0.5 will reduce epoch twice)
        """
        return weighted_sampler(self.y, *args, **kwargs)
    
    
def make_debug(fn, n=32):
    ds = Dataset(fn)
    to_save = {}
    keys = [k for k in ds.dataset.keys()]
    print(f'Loaded dataset from {fn} with keys {keys}')
    for k in keys:
        to_save[k] = ds.dataset[k][:n]
    np.savez(fn.replace('.npz', '')+f'_debug_{n}.npz', **to_save)


class TokenizerDataset(Dataset):
    def __init__(self, fn, tokenizer_name='xlm-roberta-large', tokenizer=None, max_length=512, clean=True, text_column='comment_text', transforms=None):
        # Data
        self.dataset = pd.read_csv(fn)
        self.columns = self.dataset.columns
        # Handle different column structure
        assert text_column in self.columns, f'{text_column} is not among columns {self.columns}'
        self.text_column = text_column
        self.y = np.array(self.dataset['toxic'] > 0.5 if 'toxic' in self.columns else np.zeros(len(self.dataset)), dtype=np.uint8)

        self.n_classes = 2
        self.clean = clean

        # Tokenizer
        self.tokenizer_name = tokenizer_name
        self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer_name) if tokenizer is None else tokenizer
        self.max_length = max_length
        self.transforms = compose_transforms(transforms)


    def __getitem__(self, i):
        x = self.dataset[self.text_column][i]
        y = self.y[i]

        if self.clean:
            x = clean_text(x, lang=self.dataset.lang[i] if 'lang' in self.dataset.columns else 'en')
            # x = clean_text(x)

        # Process x and y inherited from base (tokenized) class
        x = self.process_x(x)
        y = self.process_y(y)

        # Get tokenized input and attention mask from tokenizer
        x, am = tokenize([x], self.tokenizer, max_length=self.max_length)
        x, am = x[0], am[0]

        # typization
        x, y, am = map(lambda t: torch.tensor(t).long(), [x, y, am])
        return x, y, am


class ConcatDataset(D.ConcatDataset):
    """Concat dataset implementing weighted sampler"""
    def weighted_sampler(self, *args, **kwargs):
        y = np.concatenate([d.y for d in self.datasets])
        return weighted_sampler(y, *args, **kwargs) 


def read_multilingual_dataset(path_pattern='../input/multilingual-train-ludovick/jigsaw-toxic-comment-train_{}_clean.csv', 
                              langs=['es', 'tr', 'ru', 'fr', 'it', 'pt']):

    multilingual_df = [pd.read_csv(path_pattern.format(lang))[['id', 'comment_text', 'toxic']] for lang in langs]
    # merge
    multilingual_df = reduce(lambda x, y: pd.merge(x, y, on='id'), multilingual_df)
    # clean
    multilingual_df = multilingual_df.assign(toxic=lambda df: df.iloc[:,-1]).drop(['toxic_x', 'toxic_y'], 1).reset_index(drop=True)
    multilingual_df.columns = ['id'] + langs + ['toxic']
    return multilingual_df



class MultilingualTrainTokenizerDataset(TokenizerDataset):
    def __init__(self, path_pattern='../input/multilingual-train-ludovick/jigsaw-toxic-comment-train_{}_clean.csv',
                 filter_bad=False,
                 filter_short=0,
                 filter_long=9999,
                 lang_batch_sample=0,
                 langs=['es', 'tr', 'ru', 'fr', 'it', 'pt', 'en'],
                 remove_lang=None,
                 p_en=1/7,
                 tokenizer='xlm-roberta-large', max_length=256, clean=True, transforms=None):
        
        self.langs = langs
        self.remove_lang = remove_lang
        self.lang_batch_sample = lang_batch_sample
        self.sampled = 0
        self.dataset = read_multilingual_dataset(path_pattern, langs)

        if filter_bad:
            print('Filter confusing (0.25) and assused (0.97) examples')
            to_filter = np.load('../output/filter_confusing_025_assured_097.npy')
            self.dataset = self.dataset[~to_filter].reset_index(drop=False)

        self.filter_short = filter_short
        if filter_short > 0:
            to_filter = self.dataset.en.apply(len) < filter_short
            print(f'Filter sentences shorter than {filter_short} symbols: {np.sum(to_filter)}')
            self.dataset = self.dataset[~to_filter].reset_index(drop=False)

        self.filter_long = filter_long
        if filter_long < 9999:
            to_filter = self.dataset.en.apply(len) > filter_long
            print(f'Filter sentences longer than {filter_long} symbols: {np.sum(to_filter)}')
            self.dataset = self.dataset[~to_filter].reset_index(drop=False)

        self.y = np.array(self.dataset['toxic'] > 0.5, dtype=np.uint8)

        # randomly select lang
        self.p_en = p_en
        # self.p_oth = (1-p_en) / (len(langs)-1)
        self.text_column = 'en'

        self.n_classes = 2
        self.clean = clean

        # Tokenizer
        self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer) if isinstance(tokenizer, str) else tokenizer
        self.max_length = max_length
        self.transforms = compose_transforms(transforms)

    def __getitem__(self, i):
        # randomly select lang if it's time
        # if self.lang_batch_sample == 0 or (self.lang_batch_sample > 0 and self.sampled % self.lang_batch_sample == 0):
        #     # if we want to use only translations (remove source lang from selection)
        #     if self.remove_lang is not None:
        #         langs = [lang for lang in self.langs if lang != self.remove_lang[i]]
        #     else:
        #         langs = self.langs
        #     p_oth = (1-self.p_en) / (len(langs)-1)
        #     p = [p_oth if lang != 'en' else self.p_en for lang in langs]
        #     self.text_column = np.random.choice(langs, p=p)
        #     # print(self.text_column)
        # TODO rewrite this clause properly
        if self.lang_batch_sample == 0:
            self.text_column = np.random.choice(self.langs)
        elif self.sampled % self.lang_batch_sample == 0:
            self.text_column = np.random.choice(self.langs)
        self.sampled += 1
        # print(self.sampled)
        return super().__getitem__(i)


class PseudolabelDataset(TokenizerDataset):
    def __init__(self, scores, threshold=0.95, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.scores = scores
        self.mask = (scores > threshold) | (scores < (1-threshold))

        print(f'Use {np.sum(self.mask)} rows for pseudolabels from {len(self.y)}')

        self.dataset = self.dataset[self.mask].reset_index()
        self.y = (scores > threshold).astype(np.uint8)[self.mask]

        print('Pseudolabeling accuracy: ', np.mean(self.y == self.dataset['toxic']))


class PseudolabelMultilingualDataset(MultilingualTrainTokenizerDataset):
    def __init__(self, scores, path_pattern='../input/multilingual-train-ludovick/validation_{}.csv', threshold=0.95, *args, **kwargs):
        super().__init__(path_pattern=path_pattern, *args, **kwargs)
        self.threshold = threshold
        self.scores = scores
        self.mask = (scores > threshold) | (scores < (1-threshold))

        print(f'Use {np.sum(self.mask)} rows for pseudolabels from {len(self.y)}')

        self.dataset = self.dataset[self.mask].reset_index()
        self.y = (scores > threshold).astype(np.uint8)[self.mask]

        print('Pseudolabeling accuracy: ', np.mean(self.y == self.dataset['toxic']))
        return weighted_sampler(y)


class ToxicDFDataset(D.Dataset):
    """ tokenizes the comments from the pandas DataFrame
        adapted from transformers TextDataset
    """

    def __init__(
            self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache=False, debug=False,
    ):
        logger.info("ToxicDFDataSet: file_path=%s", file_path)
        assert os.path.isfile(file_path)

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename, ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples, self.toxicity_levels = pickle.load(handle)
                    assert len(self.examples) == len(self.toxicity_levels)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                # with open(file_path, encoding="utf-8") as f:
                #     text = f.read()
                if debug:
                    # to be able to run it on CPU and low resources for debugging purposes
                    toxic_df = pd.read_pickle(file_path).loc[:100]
                else:
                    toxic_df = pd.read_pickle(file_path)

                logger.info("Read the text.")

                def convert_tokens_to_ids(row):
                    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(row['comment_text']))

                # tokenize the comments
                tqdm.pandas(desc='Tokenize')
                toxic_df['token_ids'] = toxic_df.progress_apply(convert_tokens_to_ids, axis=1)

                logger.info("Summary of the token lengths: \n", toxic_df['token_ids'].apply(lambda x: len(x)).describe())

                # create a list of examples from the toxic comments
                self.examples = []
                self.toxicity_levels = []
                for i, row in toxic_df.iterrows():
                    tokenized_text = row['token_ids']
                    if len(tokenized_text) <= block_size:
                        self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text))
                        self.toxicity_levels.append(row['toxic'])
                    else:  # chunk up bigger entries
                        for i in range(0, len(tokenized_text) - block_size + 1,
                                       block_size):  # Truncate in block of block_size
                            self.examples.append(
                                tokenizer.build_inputs_with_special_tokens(tokenized_text[i: i + block_size])
                            )
                            self.toxicity_levels.append(row['toxic'])

                del toxic_df

                assert len(self.examples) == len(self.toxicity_levels)

                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump([self.examples, self.toxicity_levels], handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        assert len(self.examples) == len(self.toxicity_levels)
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        """ returns tuple of tensors, first the tokenized toxic comment, second the toxicity level """
        # return torch.tensor(self.examples[i], dtype=torch.long)
        x, y = map(lambda t: torch.tensor(t, dtype=torch.long), [self.examples[i], self.toxicity_levels[i]])
        return x, y


class DataCollatorForConditionedLanguageModeling(DataCollatorForLanguageModeling):
    def collate_batch(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        examples_0 = [e[0] for e in examples]
        examples_1 = [e[1] for e in examples]
        batch = self._tensorize_batch(examples_0)
        batch_condition = torch.tensor(examples_1, dtype=torch.float)
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs, "masked_lm_labels": labels, "conditional_var": batch_condition}
        else:
            return {"input_ids": batch, "labels": batch, "conditional_var": batch_condition}
