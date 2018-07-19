import pandas as pd
import numpy as np
def gen_expand(gen,num):
    for i,(x,y) in enumerate(gen):
    #     print(i)
        print(x.shape,y.shape)
        if i>num:
            break
    #     print(torch.tensor(x))
    #     print(torch.tensor(y))


def encode_page_features(df):
    """
    Applies one-hot encoding to page features and normalises result
    :param df: page features DataFrame (one column per feature)
    :return: dictionary feature_name:encoded_values. Encoded values is [n_pages,n_values] array
    """
    def encode(column):
        one_hot = pd.get_dummies(df[column], drop_first=False)
        # noinspection PyUnresolvedReferences
        return (one_hot - one_hot.mean()) / one_hot.std()
        # return one_hot

    return {str(column): encode(column) for column in df}

def min_float(type_):
    return np.finfo(type_).eps
def normalize(rewards):
    return (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

