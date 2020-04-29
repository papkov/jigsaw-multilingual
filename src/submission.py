import numpy as np
import pandas as pd


def scale_min_max_submission(submission):
    min_, max_ = submission['toxic'].min(), submission['toxic'].max()
    submission['toxic'] = (submission['toxic'] - min_) / (max_ - min_)
    return submission


def make_submission(predictions):
    submission = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
    submission['toxic'] = predictions
    submission.to_csv('submission.csv', index=False)