import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gc
from funtest.test_pathlib import first_try
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('/opt/playground/diaodu/')
from dst import *
from model import *
sys.path.append('/opt/playground/diaodu/util')
from gen_expand import *
from threed_view import *
from meter import AverageMeter
import random
from preprocess import Preprocess
from preprocess import en_vec
from env_stat import Env_stat
from keras.utils import Progbar
from env_stat import MA_NUM,APP_NUM

parser = argparse.ArgumentParser(description='')
parser.add_argument('--seed',type=int,default=553)
parser.add_argument('--fn',type=str,default='')
parser.add_argument('--log-interval', type=int, default=8, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--dump-interval', type=int, default=100, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--gamma', type=float, default=0.89, metavar='G',
                    help='discount factor (default: 0.99)')
args = parser.parse_args()
print(args.gamma)

print('train')
torch.manual_seed(args.seed)

pre_processor=Preprocess()
df_app_res,df_machine,df_ins,df_app_inter,df_ins_sum=pre_processor.run_pre()
del pre_processor
gc.collect()
env=Env_stat(df_machine,df_app_res,df_app_inter,df_ins_sum,verbose=0)
del df_machine,df_app_res,df_app_inter
gc.collect()


from policymodel import Policy
m=Policy()
# m.weight_action(init='normal')
optimizer=optim.Adam(m.parameters(),lr=0.02)
dic={}
def dic_init(fn=None):
    if fn==None:
        dic['saved_log_probs']=[]
        dic['rewards'] =[]
        dic['iid']=''
        dic['id_']=-1
        dic['mid']=np.ones(MA_NUM)*-1
    else:
        load(fn)

    # m=Policy()
    # optimizer=optim.Adam(m.parameters(),lr=0.2)

def quick_roll_save(e='quick_roll'):
    fn='policy{}.pth.tar'.format(e)
    # dic['saved_log_probs'].append(log_probs)
    # dic['rewards'].append(rewards)
    dic['env_dic']=env.dic
    # dic['id_']=id_
    # dic['policy_rewards']=[]
    # dic['policy_rewards'].append(policy_rewards)
    torch.save(dic,fn)

def save_checkpoints(rewards,log_probs,e,iid,id_):
    fn='policy{}.pth.tar'.format(e)
    # dic['saved_log_probs'].append(log_probs)
    # dic['rewards'].append(rewards)
    # dic['env_dic']=env.dic
    dic['state_dict']=m.state_dict()
    dic['iid']=iid
    dic['id_']=id_
    # dic['policy_rewards']=[]
    # dic['policy_rewards'].append(policy_rewards)
    # dic['len'].append(len(self.rewards))
    torch.save(dic,fn)

def load_checkpoints(fn):
    checkpoint=torch.load(fn)
    env.load_checkpoints(checkpoint['env_dic'])
    dic['id_']=checkpoint['id_']
    dic['iid']=checkpoint['iid']
    m.load_state_dict(checkpoint['state_dic'])

def gen_id(inp):

    digits=m(inp)
    # o=F.softmax(m(inp),dim=1)
#     o_=Categorical(o)
#     digits=o_.sample()
    id_=digits.dot(torch.tensor([10**3,10**2,10,1]))
    id_=id_.tolist()
    return id_

def run_game(mid,step,not_quick_roll):

    # dic['mid'].append(mid)
    # id_=id_.tolist()
    # print(id_)

    reward,end=env.evaluate(step,mid)
#     loss=digits.to(torch.float).dot(m.get_logprob(digits)*-1)
    if not_quick_roll:
        loss=m.get_logprob(mid)
        # print(loss)

        ret_=torch.tensor([0],dtype=torch.float)

        for each in loss:
            ret_=ret_+each

    #     print(ret_)
        m.save_logprob(ret_,reward)
    return end

#     loss=digits.to(torch.float).dot(m.get_logprob(digits)*-1)*rewards

def get_frame():
    return torch.tensor(env.matrix,dtype=torch.float).view(24,107,-1)
    # return torch.randn(4,36)

# m.save_logprob(2,2)
def load(fn):
    ck=torch.load(fn)
    # env.matrix=ck['env_dic']['matrix']
    dic['mid']=ck['mid']
    dic['id_']=ck['id_']
    # m.load_state_dict(ck['state_dict'])

def train():
    print('train')
    bar=Progbar(target=len(env.i_a),width=30,interval=0.05)

    log_rewards = []
    log_saved = []
    dic_init()

    loss_old=0
    for epoch in range(1000):
        m.logprob_history=[]
        m.rewards=[]
        env.reset()
        # load_checkpoints(fn)
        for id_,(iid,aid) in enumerate(env.i_a.items()):
        # foirwarding
            log_mid=dic['mid'][id_]
            inp=get_frame()
            # cur=env.app[app.aid==aid]

            cur=env.a_idx[aid]

            a=env.df_a_i.iloc[cur].cpu.split('|')
            assert pd.Series(a,dtype=float).max()==env.unit['c'][cur]

            if log_mid==-1:
                not_quick_roll=1
                mid=m(inp)
                # digits=234
            #     print(digits)
                # id_=digits.dot(torch.tensor([10**3,10**2,10,1]))
                dic['mid'][id_]=mid
            ## into a quick_roll
            else:
                print('in quick roll')
                not_quick_roll=0

                mid=dic['mid'][id_]

            end=run_game(mid,cur,not_quick_roll)
            bar.update(id_)

            for each in ['c','m','d','p_pm','m_pm','pm','cm','a']:
                assert env.deploy_state[each].shape==(MA_NUM,)

            e='_'.join([str(epoch),str(iid)])
            if id_%args.dump_interval==0:
                save_checkpoints(log_rewards,log_saved,e,iid,id_)

            if end:
                break
    #     print('---------------------------')
        rewards = []
        # log_rewards = []
        # log_saved = []
        R=0
        for r in m.rewards[::-1]:
            R = r + args.gamma * R
            rewards.insert(0, R)
        rewards = torch.Tensor(rewards)
        log_rewards.append(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + torch.tensor(np.finfo(np.float32).eps,dtype=torch.float))
        log_rewards.append(rewards)
    #     print('the overall reward{}'.format(rewards))
    #     rewards = (rewards - rewards.mean()) / (rewards.std() + torch.tensor(np.finfo(np.float32).eps,dtype=torch.float))
    #     print(rewards)
        loss_li=[]
        log_saved.append(m.logprob_history)
        for log_prob,r in zip(m.logprob_history,rewards):
            loss_li.append(-log_prob*r)
    #     print(m.logprob_history)
    #     print(loss_li)
        log_saved.append(loss_li)
        optimizer.zero_grad()
        loss = torch.cat(loss_li).sum()
        log_saved.append(loss)
        if loss_old==loss:
            print('loss stay the same')
            torch.manual_seed(args.seed+100)
            # m=Policy()
            # optimizer=Adam(m.parameters(),lr=0.015)
        loss_old=loss
        if epoch%args.log_interval==0:
            save_checkpoints(log_rewards,log_saved,epoch,0,0)
        print('---------------------------')
        print(loss)
        print('---------------------------')
        loss.backward()
        optimizer.step()
        del m.logprob_history[:]
        del m.rewards[:]


def quick_roll():
    print('train')
    bar=Progbar(target=len(env.i_a),width=30,interval=0.05)

    log_rewards = []
    log_saved = []
    dic_init(args.fn)

    loss_old=0
    for epoch in range(1):
        m.logprob_history=[]
        m.rewards=[]
        env.reset()
        print(len(dic['mid']))
        # load_checkpoints(fn)
        for id_,(iid,aid) in enumerate(env.i_a.items()):
        # foirwarding
            log_mid=dic['mid'][id_+1]
            inp=get_frame()
            # cur=env.app[app.aid==aid]

            cur=env.a_idx[aid]

            a=env.df_a_i.iloc[cur].cpu.split('|')
            assert pd.Series(a,dtype=float).max()==env.unit['c'][cur]

            if log_mid==-1:
                break
            ## into a quick_roll
            else:
                not_quick_roll=0

                mid=dic['mid'][id_+1]

            end=run_game(mid,cur,not_quick_roll)
            show_mid=[['mid',mid.data.numpy()]]
            bar.update(id_)
load(args.fn)
quick_roll()
quick_roll_save()
# train()
