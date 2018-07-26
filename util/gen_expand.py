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

def mplot(env):
    for i in range(6):
        yield pd.Series(env.matrix[i]).plot()

def mmplot(env):
    for i,keys in zip(range(6),['cpu','mem','disk','p','m','pm']):
        print(i,keys)
        yield pd.Series(env.matrix[i][9339:9339+6000]/env.mn[keys]).plot()

def dplot(env):
    for i,keys in zip(range(6),['c','m','d','p_pm','m_pm','pm']):
        print(i,keys)
        yield pd.Series(env.deploy_state[keys]).plot()

def numplot(env):
    pd.Series(env.a_i).plot()

def unit_max(env):
    for key in ['c','m','d','p_pm','m_pm','pm']:
        print(key,(np.argmax(env.unit[key]),env.unit[key].max()))

def app_view(env,aid):
    env.app[env.app.aid==aid].head()

def elegant_stack():

    xn=np.empty_like(np.identity(5)[i])
    xn_li=[]
    [xn_li.append(np.identity(5)[i]) for i in range(3)]
    xn_li.append(xn)
    # xn=np.sum([xn ,np.sum(xn_li,axis=0)],axis=0)
    xn=np.sum(xn_li,axis=0)


