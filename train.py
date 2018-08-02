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
sys.path.append('./')
from dst import *
from model import *
sys.path.append('./util')
from gen_expand import re_find_y
from threed_view import *
from meter import AverageMeter
import random
from preprocess import Preprocess
from preprocess import en_vec
from env_stat import Env_stat
from keras.utils import Progbar
from env_stat import MA_NUM,APP_NUM,INST_NUM,NUM_PROC
from collections import OrderedDict
from itertools import count
from multiprocessing import Pool
import os
import ipdb

parser = argparse.ArgumentParser(description='')
parser.add_argument('--seed',type=int,default=553)
parser.add_argument('--verbose',type=int,default=1)
parser.add_argument('--fn',type=str,default='')
parser.add_argument('--base',type=str,default='')
parser.add_argument('--use-cache',type=int,default=0)
parser.add_argument('--run-id',type=str,default='')
parser.add_argument('--log-interval', type=int, default=8, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--dump-interval', type=int, default=500, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--gamma', type=float, default=0.89, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--non-roll', type=int, default=1,metavar='N')
args = parser.parse_args()
verbose=args.verbose
print(args.gamma)

print('train')
torch.manual_seed(args.seed)

pre_processor=Preprocess(args.base)
df_app_res,df_machine,df_ins,df_app_inter,df_ins_sum=pre_processor.run_pre()
del pre_processor
gc.collect()
proc=Pool(NUM_PROC)


env=Env_stat(df_machine,df_app_res,df_app_inter,df_ins_sum,verbose=0)
del df_machine,df_app_res,df_app_inter
gc.collect()


from policymodel import Policy
m=Policy()
# m.weight_action(init='normal')
optimizer=optim.Adam(m.parameters(),lr=0.001)
dic={}

use_cuda = torch.cuda.is_available()


def dic_init(fn=None):
    if fn==None:
        dic['saved_log_probs']=[]
        dic['rewards'] =[]
        dic['iid']=pd.Series(np.array(['']*INST_NUM))
        dic['id_']=-1
        dic['mid']=pd.Series(np.ones(INST_NUM,dtype=int)*-1)
        dic['aid']=pd.Series(np.array(['']*INST_NUM))
        dic['im']={}
        dic['ia']={}
        dic['im']={i:-1 for i,_ in env.i_a.items()}
        [op_(i,m,'im')for i,m in zip(dic['iid'],dic['mid']) if i!='']
        dic['im']=OrderedDict(sorted(dic['im'].items(), key=lambda t: t[1],reverse=True))
    else:
        load(fn)
        # load_checkpoints(fn)

    # m=Policy()
    # optimizer=optim.Adam(m.parameters(),lr=0.2)

def quick_roll_save(e='quick_roll'):
    fn='policy{}.pth.tar'.format(e)
    # dic['saved_log_probs'].append(log_probs)
    # dic['rewards'].append(rewards)
    env.save_checkpoints()
    dic['env_dic']=env.dic
    # dic['id_']=id_
    # dic['policy_rewards']=[]
    # dic['policy_rewards'].append(policy_rewards)
    torch.save(dic,fn)

def save_checkpoints(rewards,log_probs,e,iid,id_,run_id):
    if run_id not in os.listdir('/data2/run/'):os.mkdir('/data2/run/'+run_id)
    fn='/data2/run/{0}/policy{1}.pth.tar'.format(run_id,e)
    dic['saved_log_probs']=log_probs
    # dic['rewards']=rewards
    # dic['env_dic']=env.dic
    dic['state_dict']=m.state_dict()
    # dic['iid']=iid
    # dic['id_']=id_
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

    reward,end=env.evaluate(step,mid,proc,re_find_y)
#     loss=digits.to(torch.float).dot(m.get_logprob(digits)*-1)
    if not_quick_roll:
        loss=m.get_logprob(mid)
        # print(loss)

        # ret_=torch.tensor([0],dtype=torch.float)

        # for each in loss:
            # ret_=ret_+each

    #     print(ret_)
        m.save_logprob(loss,reward)
    return end

#     loss=digits.to(torch.float).dot(m.get_logprob(digits)*-1)*rewards

def get_frame():
    return torch.tensor(env.matrix,dtype=torch.float).view(6,4,107,-1)
    # return torch.randn(4,36)

# m.save_logprob(2,2)
def load(fn):
    ck=torch.load(fn)
    # env.matrix=ck['env_dic']['matrix']
    dic['mid']=ck['mid'].copy()
    dic['id_']=ck['id_']
    dic['aid']=ck['aid']
    dic['iid']=ck['iid'].copy()
    ck={}
    dic['im']={i:m for i,m in zip(dic['iid'],dic['mid']) if i!=''}

    dic['ia']={i:env.i_a[i] for i in dic['iid'] if i!=''}
    # [dic['ia']=env.i_a[i] for i in dic['iid'] if i=='']
    # [dic['ia'][i]=a for i,a in env.i_a.items() if i not in dic['iid']]
    # m.load_state_dict(ck['state_dict'])

def load_checkpoints(fn):
    print('load_checkpoints')
    def op_(i,a,key):
        dic[key][i]=a
    # fn=['./bk/policy0_inst_61107.pth.tar',
    # fn= ['policy2_inst_14294.pth.tar']
    # fn= ['policy2_inst_70527.pth.tar']
    # fn= ['policy2_inst_74916.pth.tar']
    # fn= ['policy2_inst_40927.pth.tar']
    if fn==None:
        fn= ['./bk/policy2_inst_21132.pth.tar']
        fn= ['./run/k/policy1_inst_61285.pth.tar','./run/l/policy1.pth.tar']
    else:
        fn=[fn]

    # fn= ['policy2_inst_40927.pth.tar']
        # ,'./bk/policy5_inst_48614.pth.tar','./bk/policy7_inst_22575.pth.tar' ]
        # './bk/policy0.pth.tar']
    ck=torch.load(fn[0])
    # env.matrix=ck['env_dic']['matrix']
    dic['mid']=ck['mid'].copy()
    dic['id_']=ck['id_']
    dic['aid']=ck['aid']
    dic['iid']=ck['iid'].copy()
    m.logprob_history=ck['saved_log_probs']
    m.load_state_dict(ck['state_dict'])
    m.rewards=ck['rewards']
    ck={}

    dic['im']={i:-1 for i,_ in env.i_a.items()}
    [op_(i,m,'im')for i,m in zip(dic['iid'],dic['mid']) if i!='']
    dic['im']=OrderedDict(sorted(dic['im'].items(), key=lambda t: t[1],reverse=True))
    # [op_(i,a,'ia') for i,a in env.i_a.items() if i not in dic['iid']]

    for i in range(1,len(fn)):
        ck=torch.load(fn[i])
        m.logprob_history=ck['saved_log_probs']
        # m.rewards=np.ones_like(ck['saved_log_probs']).tolist()
        m.load_state_dict(ck['state_dict'])
        m.rewards=ck['rewards']
        dic['mid']=ck['mid'].copy()
        dic['id_']=ck['id_']
        dic['aid']=ck['aid']
        dic['iid']=ck['iid'].copy()
        ck={}


        [op_(i,m,'im') for i,m in zip(dic['iid'],dic['mid']) if i!='']
        # [op_(i,env.i_a[i],'ia') for i in dic['iid'] if i!='']
        dic['im']=OrderedDict(sorted(dic['im'].items(), key=lambda t: t[1],reverse=True))
        # [op_(i,a,'ia') for i,a in env.i_a.items() if i not in dic['iid']]
    dic['saved_log_probs']=[]
    dic['rewards'] =[]
    dic['iid']=pd.Series(np.array(['']*INST_NUM))
    dic['mid']=pd.Series(np.ones(INST_NUM,dtype=int)*-1)

def train(m):
    print('train')
    print('non_roll_mode:{}'.format(args.non_roll))
    # bar=Progbar(target=len(env.i_a),width=30,interval=0.05)

    log_rewards = []
    log_saved = []
    dic_init()
    # env.not_quick_roll=0

    loss_old=0

    if args.use_cache:

        load_checkpoints()

    for epoch in count(1):
        # m.logprob_history=[]
        # m.rewards=[]
        env.reset()
        # this is the place to load the policy checkpoint

        bar=Progbar(target=len(env.i_a),width=30,interval=0.05)

        for id_,(iid,mid) in enumerate(dic['im'].items()):
            if id_==0:
                print(iid)

            if epoch==3:
                if id_==300:
                    # ipdb.set_trace()
                    pass
        # foirwarding
            # mid=dic['im'][iid]
            inp=get_frame()
            if use_cuda:
                inp=inp.cuda()
            # cur=env.app[app.aid==aid]
            aid=env.i_a[iid]

            cur=env.a_idx[aid]

            a=env.df_a_i.iloc[cur].cpu.split('|')
            assert pd.Series(a,dtype=float).max()==env.unit['c'][cur]

            if mid==-1|args.non_roll:
                env.not_quick_roll=1
                env.not_quick_roll=1
                env.not_quick_roll=1
                mid=m(inp)
                # mid=torch.randint(6000,size=(1,))
                mid=mid.data[0]
                # digits=234
            #     print(digits)
                # id_=digits.dot(torch.tensor([10**3,10**2,10,1]))
                dic['mid'][id_]=mid
                # dic['aid'][id_]=aid
                dic['iid'][id_]=iid
            ## into a quick_roll
            else:
                # print('in quick roll')
                env.not_quick_roll=0
                # mid=dic['im'][iid]

            end=run_game(mid,cur,env.not_quick_roll)
            bar.update(id_)

            # for each in ['c','m','d','p_pm','m_pm','pm','cm','a']:
                # assert env.deploy_state[each].shape==(MA_NUM,)

            e='_'.join([str(epoch),str(iid)])
            if verbose:
                if id_%args.dump_interval==0:
                    save_checkpoints(m.rewards,m.logprob_history,e,iid,id_,args.run_id)

            if end:
                break

            # del m.logprob_history[:]
            # del m.rewards[:]

    #     print('---------------------------')
        if epoch%1==0:

            rewards = []
            # ipdb.set_trace()
            # log_rewards = []
            # log_saved = []
            R=0
            for r in m.rewards[::-1]:
                R = r + args.gamma * R
                rewards.insert(0, R)
            rewards = torch.Tensor(rewards)

            # log_rewards.append(rewards.data.numpy())

            rewards = (rewards - rewards.mean()) / (rewards.std() + torch.tensor(np.finfo(np.float32).eps,dtype=torch.float))
            # print('\nrewards:{}'.format(rewards))
            rewards = rewards/10
            # print('\nrewards:{}'.format(rewards))

            # log_rewards.append(rewards.data.numpy())

        #     print('the overall reward{}'.format(rewards))
        #     rewards = (rewards - rewards.mean()) / (rewards.std() + torch.tensor(np.finfo(np.float32).eps,dtype=torch.float))
        #     print(rewards)
            loss_li=[]
            # log_saved.append(m.logprob_history.data.numpy())
            # log_saved.append(torch.cat(m.logprob_history).data.numpy())
            for log_prob,r in zip(m.logprob_history,rewards):
                loss_li.append(-log_prob*r)
        #     print(m.logprob_history)
        #     print(loss_li)

            # log_saved.append(torch.cat(loss_li).data.numpy())

            loss = torch.cat(loss_li).sum()

            log_saved.append(loss.data.numpy())

            # if loss_old==loss:
                # print('loss stay the same')
                # torch.manual_seed(args.seed+100)
                # m=Policy()
                # optimizer=Adam(m.parameters(),lr=0.015)
            # if epoch%args.log_interval==0:
            if verbose:
                if epoch%3==0:
                    save_checkpoints(log_rewards,log_saved,epoch,0,0,args.run_id)
            print('\n---------------------------')
            print(loss)
            print('---------------------------')
            if epoch%1==0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                del m.logprob_history[:]
                del m.rewards[:]
                dic_init()


def quick_roll():
    print('quick_roll')
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
        # for id_,(iid,aid) in enumerate(env.i_a.items()):
        # for id_,(iid,mid) in enumerate(dic['im'].items()):
        for id_,iid in enumerate(dic['iid']):
        # foirwarding
            # cur=env.app[app.aid==aid]
            mid=dic['mid'][id_]
            # print(iid,mid)

            # if (aid+iid)=='':
            if mid==-1:
                break
            else:
                if id_==0:
                    print(iid)
                if id_==300:
                    print(iid)
                aid=env.i_a[iid]
                cur=env.a_idx[aid]
                a=env.df_a_i.iloc[cur].cpu.split('|')
                assert pd.Series(a,dtype=float).max()==env.unit['c'][cur]
                not_quick_roll=0
                # mid=dic['im'][iid]

            end=run_game(mid,cur,not_quick_roll)
            # show_mid=[['mid',int(mid.data.numpy())]]
            # show_mid+=[['iid',iid]]
            # show_mid+=[['aid',aid]]
            bar.update(id_)
            if end:
                break
# load(args.fn)
import torch.multiprocessing as mp
from gen_expand import re_find_y
if use_cuda:
    m=m.cuda()
if args.non_roll:
    train(m)
#     if __name__ == '__main__':
#       num_processes = 4
#       m = Policy()
#       # NOTE: this is required for the ``fork`` method to work
#       m.share_memory()
#       processes = []
#       for rank in range(num_processes):
#           p = mp.Process(target=train, args=(m,))
#           p.start()
#           processes.append(p)
#       for p in processes:
#           p.join()
else:
    quick_roll()
    quick_roll_save()
# train()
