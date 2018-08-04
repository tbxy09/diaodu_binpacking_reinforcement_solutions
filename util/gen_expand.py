import pandas as pd
import numpy as np
import torch
def proc_split(app_inter,unit,num):
# 4405
    li=[]
    unit=int(unit)
    # li.append(app_inter.iloc[:1*unit,:])
    # li.append(app_inter.iloc[1*unit:2*unit,:])
    # li.append(app_inter.iloc[2*unit:3*unit,:])
    # li.append(app_inter.iloc[3*unit:4*unit,:])
    # li.append(app_inter.iloc[4*unit:5*unit,:])
    # li.append(app_inter.iloc[5*unit:6*unit,:])
    # li.append(app_inter.iloc[6*unit:7*unit,:])
    # li.append(app_inter.iloc[7*unit:,:])
    for i in range(num-1):
        li.append(app_inter.iloc[i*unit:(i+1)*unit])
    li.append(app_inter.iloc[(num-1)*unit:,:])
    # print('\nlen{}'.format(len(li)))
    return li
def some_code(re_find,text):
        result_a1=proc.apply_async(re_find,(text,vi_li_a1))
        if result_a1.get()==1:
            print('\ninfer end')
            return 1,1
        result_a2=proc.apply_async(re_find,(text,vi_li_a2))
        if result_a2.get()==1:
            print('\ninfer end')
            return 1,1
        result_b1=proc.apply_async(re_find,(text,vi_li_b1))
        if result_b1.get()==1:
            print('\ninfer end')
            return 1,1
        result_b2=proc.apply_async(re_find,(text,vi_li_b2))
        if result_b2.get()==1:
            print('\ninfer end')
            return 1,1
        result_c1=proc.apply_async(re_find,(text,vi_li_c1))
        if result_c1.get()==1:
            print('\ninfer end')
            return 1,1
        result_c2=proc.apply_async(re_find,(text,vi_li_c2))
        if result_c2.get()==1:
            print('\ninfer end')
            return 1,1
        result_d1=proc.apply_async(re_find,(text,vi_li_d1))
        if result_d1.get()==1:
            print('\ninfer end')
            return 1,1
        result_d2=proc.apply_async(re_find,(text,vi_li_d2))
        if result_d2.get()==1:
            print('\ninfer end')
            return 1,1

def re_find(text,v_li):
    import time
    if len(v_li):
        pat=')|('.join(v_li)
        pat=r'('+pat+')'
        print(len(pat))
        # p=re.compile(pat)
        start=time.time()
        if re.findall(pat,text):
            return 1
        end=time.time()
        print('\ninside infer end:{}'.format(end-start))
    return 0

def re_find_y(text,app_inter):
    import re
    p=re.compile('(\s+)')
    text=p.sub(' ',text)

    for g,v in app_inter.groupby('aid') :
        if re.findall(g,text):
            for each in v.ab:
                if re.findall(each,text):
                    print(each)
                    return 1
    return 0

# def re_find_y(li,app_inter):
#     print(li[0])
#     for m in li[:1]:
#     end=app_inter[['v','ab_encode']].apply(
#                                     lambda x: [m.count(each) for each in x.ab_encode]==[1,x.v],axis=1).sum()
#     if end:
#         return 1
def re_find_x(text,a=1):
    import re
    p=re.compile('(\s+)')
    text=p.sub(' ',text)
    if a==0:
        for g,v in self.app_inter.groupby('aid') :
            if re.findall(g,text):
            # if re_findall(g,text):
                pat=')|('.join(v.bid.tolist())
                pat=r'('+pat+')'
                if re.findall(pat.bid,text):
                    for each in v.ab:
                        if re.findall(each,text):
                        # if re_findall(each,text):
                            print(each)
                            return 1
    if a==1:
        v_li=[]
        for g,v in self.app_inter.groupby('aid') :
            if re.findall(g,text):
                for each in v.ab:
                    v_li.append(each)
        if len(v_li):
            pat=')|('.join(v_li)
            pat=r'('+pat+')'
            print(len(pat))
            # p=re.compile(pat)
            start=time.time()
            if re.findall(pat,text):
                return 1
            end=time.time()
            print('\ninside infer end:{}'.format(end-start))
            # if re_findall(g,text):
    if a==2:
        self.p.findall(text)
    return 0

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
    o=m.netc(inp)
    # print(m.get_logprob(o),o)
    # loss_prob_li.append(m.get_logrob(o))
    # rewards=env.evaluate(o)

def ck_parser(fn,m,env_stat=None):
    # ck=torch.load('./run/a/policy2_inst_43727.pth.tar')
    ck=torch.load(fn)

    # ck=torch.load('policy8.pth.tar')

    # r=ck['rewards']
    log_prob=ck['saved_log_probs']
    mid=ck['mid']
    iid=ck['iid']
    if env_stat:
        env_stat.matrix=ck['env_dic']['matrix']
        env_stat.deploy_state=ck['env_dic']['deploy_state']
        return log_prob,[mid[i] for i in range(68219) if mid[i]!=-1], [iid[i] for i in range(68219) if iid[i]!='']
    # aid=ck['aid']
    # id_=ck['id_']
    state_dict=ck['state_dict']
    m.load_state_dict(state_dict)
    # env_dic=ck['env_dic']
    print(len([mid[i] for i in range(68219) if mid[i]!=-1]))
    print(len([iid[i] for i in range(68219) if mid[i]!=-1]))
    [mid[i] for i in range(68219) if mid[i]!=-1]
    return log_prob,[mid[i] for i in range(68219) if mid[i]!=-1], [iid[i] for i in range(68219) if iid[i]!='']

