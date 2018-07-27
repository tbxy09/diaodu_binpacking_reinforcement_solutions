import pandas as pd
import numpy as np
import torch
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
# %load -r 140-144 /opt/playground/diaodu/train.py

def get_frame(env):
    #     return torch.tensor(env.matrix,dtype=torch.float).view(24,107,-1)
    return torch.tensor(env.matrix,dtype=torch.float).view(6,4,107,-1)
        # return torch.randn(4,36)

def mrun(env,m,inp):
    loss_prob_li=[]
    rewards=[]
    o=m(inp)
    # print(m.get_logprob(o),o)
    # loss_prob_li.append(m.get_logrob(o))
    # rewards=env.evaluate(o)

def ck_parser(fn,m):
    # ck=torch.load('./run/a/policy2_inst_43727.pth.tar')
    ck=torch.load(fn)

    # ck=torch.load('policy8.pth.tar')

    # env_stat.matrix=ck['env_dic']['matrix']
    # env_stat.deploy_state=ck['env_dic']['deploy_state']
    # r=ck['rewards']
    log_prob=ck['saved_log_probs']
    mid=ck['mid']
    iid=ck['iid']
    # aid=ck['aid']
    # id_=ck['id_']
    state_dict=ck['state_dict']
    m.load_state_dict(state_dict)
    # env_dic=ck['env_dic']
    print(len([mid[i] for i in range(68219) if mid[i]!=-1]))
    print(len([iid[i] for i in range(68219) if mid[i]!=-1]))
    [mid[i] for i in range(68219) if mid[i]!=-1]
    return log_prob,[mid[i] for i in range(68219) if mid[i]!=-1], [iid[i] for i in range(68219) if iid[i]!='']
