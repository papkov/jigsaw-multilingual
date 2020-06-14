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

from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from preprocessing import tokenize, clean_text
from transforms import Compose, compose_transforms

from transformers import AutoTokenizer, XLMRobertaTokenizer, PreTrainedTokenizer, DataCollatorForLanguageModeling
from typing import Dict, List

logger = logging.getLogger(__name__)


def weighted_sampler(y):
    labels, counts = np.unique(y, return_counts=True)
    weights = counts[::-1] / counts.sum()
    weights = np.array([weights[i] for i in y])
    return WeightedRandomSampler(weights, int(counts.min() * 2))


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
    
    def weighted_sampler(self):
        return weighted_sampler(self.y)
    
    
def make_debug(fn, n=32):
    ds = Dataset(fn)
    to_save = {}
    keys = [k for k in ds.dataset.keys()]
    print(f'Loaded dataset from {fn} with keys {keys}')
    for k in keys:
        to_save[k] = ds.dataset[k][:n]
    np.savez(fn.replace('.npz', '')+f'_debug_{n}.npz', **to_save)


class TokenizerDataset(Dataset):
    def __init__(self, fn, tokenizer_name='xlm-roberta-large', max_length=512, clean=True, text_column='comment_text', transforms=None):
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
        self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.transforms = compose_transforms(transforms)


    def __getitem__(self, i):
        x = self.dataset[self.text_column][i]
        y = self.y[i]

        if self.clean:
            x = clean_text(x)

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
    def weighted_sampler(self):
        y = np.concatenate([d.y for d in self.datasets])
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