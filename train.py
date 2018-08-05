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
from env_stat import MA_NUM,APP_NUM,NUM_PROC
from collections import OrderedDict
from itertools import count
from multiprocessing import Pool
import os
import ipdb

parser = argparse.ArgumentParser(description='')
parser.add_argument('--seed',type=int,default=553)
parser.add_argument('--inst-num',type=int,default=0)
parser.add_argument('--verbose',type=int,default=1)
parser.add_argument('--fn',type=str,default='')
parser.add_argument('--base',type=str,default='')
parser.add_argument('--use-cache',type=int,default=0)
parser.add_argument('--run-id',type=str,default='')
parser.add_argument('--ab',type=str,default='')
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
inst_num_dic={'a':68219,'b':68224}
num_life={'b':30000,'a':15000}
        # checkpoint=torch.load('./policyquick_roll.pth.tar')
roll_file_dic={'a':'/data2/a_/policyquick_roll.pth.tar','b':'/data2/b_/policyquick_roll.pth.tar'}
base_dic={'a':'/mnt/osstb/tianchi/diaodu','b':'/mnt/osstb/diaodu'}
INST_NUM=inst_num_dic[args.ab]
NUM_LIFE=num_life[args.ab]
torch.manual_seed(args.seed)

pre_processor=Preprocess(base_dic[args.ab])
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
    def op_(i,a,key):
        dic[key][i]=a
    if fn==None:
        dic['saved_log_probs']=[]
        dic['rewards'] =[]
        dic['iid']=pd.Series(np.array(['']*INST_NUM))
        dic['id_']=-1
        dic['step']=pd.Series(np.ones(INST_NUM,dtype=int)*-1)
        dic['aid']=pd.Series(np.array(['']*INST_NUM))
        dic['istep']={}
        dic['ia']={}
        mid_copy=df_ins.mid.fillna('ff',inplace=False)
        dic['imid']={i:mid for i,mid in zip(df_ins.iid,mid_copy)}
        del mid_copy
        gc.collect()
        dic['istep']={i:-1 for i,_ in env.i_a.items()}
        # [op_(i,m,'im') for i,m in zip(dic['iid'],dic['mid']) if i!='']
        # [print(op_) for i,m in zip(dic['iid'],dic['mid']) if i!='']
        # dic['im']=OrderedDict(sorted(dic['im'].items(), key=lambda t: t[1],reverse=True))
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

def run_game(choice,cur,not_quick_roll,mid_real='ff'):

    # dic['mid'].append(mid)
    # id_=id_.tolist()
    # print(id_)

    reward,end=env.evaluate(cur,choice,proc,re_find_y,mid_real)
#     loss=digits.to(torch.float).dot(m.get_logprob(digits)*-1)
    if not_quick_roll:
        loss=m.get_logprob(choice)
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
    dic['step']=ck['step'].copy()
    dic['id_']=ck['id_']
    dic['aid']=ck['aid']
    dic['iid']=ck['iid'].copy()
    ck={}
    dic['istep']={i:m for i,m in zip(dic['iid'],dic['step']) if i!=''}

    dic['ia']={i:env.i_a[i] for i in dic['iid'] if i!=''}
    # [dic['ia']=env.i_a[i] for i in dic['iid'] if i=='']
    # [dic['ia'][i]=a for i,a in env.i_a.items() if i not in dic['iid']]
    # m.load_state_dict(ck['state_dict'])

def load_checkpoints(fn=None):
    print('load_checkpoints')
    def op_(i,a,key):
        dic[key][i]=a
    # fn=['./bk/policy0_inst_61107.pth.tar',
    # fn= ['policy2_inst_14294.pth.tar']
    # fn= ['policy2_inst_70527.pth.tar']
    # fn= ['policy2_inst_74916.pth.tar']
    # fn= ['policy2_inst_40927.pth.tar']
    if fn==None:
        fn= ['/opt/playground/diaodu/bk/policy0.pth.tar']
        # fn= ['./run/k/policy1_inst_61285.pth.tar','./run/l/policy1.pth.tar']
    else:
        fn=[fn]

    # fn= ['policy2_inst_40927.pth.tar']
        # ,'./bk/policy5_inst_48614.pth.tar','./bk/policy7_inst_22575.pth.tar' ]
        # './bk/policy0.pth.tar']
    ck=torch.load(fn[0])
    # env.matrix=ck['env_dic']['matrix']
    dic['step']=ck['step'].copy()
    dic['id_']=ck['id_']
    dic['aid']=ck['aid']
    dic['iid']=ck['iid'].copy()
    m.logprob_history=ck['saved_log_probs']
    # m.load_state_dict(ck['state_dict'])
    m.rewards=ck['rewards']
    ck={}

    dic['istep']={i:-1 for i,_ in env.i_a.items()}
    [op_(i,m,'istep') for i,m in zip(dic['iid'],dic['step']) if i!='']
    # dic['istep']=OrderedDict(sorted(dic['istep'].items(), key=lambda t: t[1],reverse=True))
    # [op_(i,a,'ia') for i,a in env.i_a.items() if i not in dic['iid']]

    for i in range(1,len(fn)):
        ck=torch.load(fn[i])
        m.logprob_history=ck['saved_log_probs']
        # m.rewards=np.ones_like(ck['saved_log_probs']).tolist()
        m.load_state_dict(ck['state_dict'])
        m.rewards=ck['rewards']
        dic['step']=ck['step'].copy()
        dic['id_']=ck['id_']
        dic['aid']=ck['aid']
        dic['iid']=ck['iid'].copy()
        ck={}


        [op_(i,m,'step') for i,m in zip(dic['iid'],dic['step']) if i!='']
        # [op_(i,env.i_a[i],'ia') for i in dic['iid'] if i!='']
        # dic['istep']=OrderedDict(sorted(dic['istep'].items(), key=lambda t: t[1],reverse=True))
        # [op_(i,a,'ia') for i,a in env.i_a.items() if i not in dic['iid']]
    dic['saved_log_probs']=[]
    dic['rewards'] =[]
    dic['iid']=pd.Series(np.array(['']*INST_NUM))
    dic['step']=pd.Series(np.ones(INST_NUM,dtype=int)*-1)

def train(m):
    print('train')
    print('non_roll_mode:{}'.format(args.non_roll))
    # bar=Progbar(target=len(env.i_a),width=30,interval=0.05)

    log_rewards = []
    log_saved = []
    dic_init()
    # env.not_quick_roll=0

    loss_old=0
    playing_len=0

    if args.use_cache:

        load_checkpoints()

    for epoch in count(1):

        m.logprob_history=[]
        m.rewards=[]
        env.reset()

        # checkpoint=torch.load('./policyquick_roll.pth.tar')
        checkpoint=torch.load(roll_file_dic[args.ab])
        env.load_checkpoints(checkpoint['env_dic'])
        if args.fn:
            checkpoint=torch.load(args.fn)
            m.load_state_dict(checkpoint['state_dict'])
        # this is the place to load the policy checkpoint


        # _,df_ins_sum['iid_num']=df_ins_sum.iid.str.split('_').str
        # df_ins_sum['iid_num']=df_ins_sum['iid_num'].astype(int)

        # df_ins_sum.iid_num.sort_values()
        # for id_,(iid,mid) in enumerate(dic['im'].items()):

        iid_li=df_ins[df_ins.mid.notnull()].iid.tolist()
        origin_len=len(iid_li)
        df_ins_copy=df_ins.copy()
        df_ins_copy.mid.fillna('',inplace=True)
        add_=df_ins_copy.sort_values(ascending=False,by='mid').iid.tolist()

        del df_ins_copy
        gc.collect()

        bar=Progbar(target=len(iid_li)+100,width=30,interval=0.05)
        update_id=0
        log_iid='ff'
        for id_,iid in enumerate(iid_li):
        # for id_,iid in enumerate(df_ins_sum.iid_num.sort_values()):

            # del m.logprob_history[:]
            # del m.rewards[:]
            step=dic['istep'][iid]
            mid_real=dic['imid'][iid]
            # print(mid_real)
            if id_==0:
                print(mid_real)
                print(iid)
                print('train_roll')
            if iid==log_iid:
                print('\n')
                print(mid_real,iid)

            if epoch==1:
                if id_==300:
                    print('xy')
                    # ipdb.set_trace()
                    # pass
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

            if step==-1|args.non_roll:
                env.not_quick_roll=1
                step=m(inp)
                # mid=torch.randint(6000,size=(1,))
                step=step.data[0]
                # digits=234
            #     print(digits)
                # id_=digits.dot(torch.tensor([10**3,10**2,10,1]))
                # dic['mid'][id_]=mid
                # dic['aid'][id_]=aid
                # dic['iid'][id_]=iid
            ## into a quick_roll
            else:
                print('in quick roll')
                # break
                env.not_quick_roll=0
                # mid=dic['im'][iid]

            end=run_game(step,cur,env.not_quick_roll,mid_real)
            # bar.update(id_+1)

            # for each in ['c','m','d','p_pm','m_pm','pm','cm','a']:
                # assert env.deploy_state[each].shape==(MA_NUM,)

            e='_'.join([str(epoch),str(iid)])
            if verbose:
                if (update_id+1)%args.dump_interval==0:
                    save_checkpoints(m.rewards,m.logprob_history,e,iid,id_,args.run_id)
                if (update_id+1)%100==0:
                    print(len(iid_li))
            if (len(iid_li)-origin_len) >NUM_LIFE:
                print('\nlen break')
                break

            if end:
                # break
                # iid_li.remove(iid)
                # if dic['imid'][iid]!=-1:
                #     iid_li.insert(id_+1,iid)
                if log_iid=='ff':
                    print(dic['imid'][iid])
                dic['imid'][iid]=env.mn.mid[env.n]
                if log_iid=='ff':
                    log_iid=iid
                    print(dic['imid'][iid])
                iid_li.append(iid)
                bar.update(id_)
                # print(iid)
                # print(dic['im'][iid])
                # print(dic['im'][iid])
            else:

                dic['step'][id_]=step
                dic['imid'][iid]=env.mn.mid[env.n]
                dic['iid'][id_]=iid
                update_id=update_id+1
                # bar.update(update_id)

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

            if len(rewards)>playing_len:
                playing_len=len(rewards)

            if playing_len>len(iid_li)-1:
                print('\n---------------------------')
                print('\nGame Win')
                print('\n---------------------------')
                break

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
            print('\n---------------------------')
            print(playing_len,len(rewards),update_id,env.counter[0],env.counter[1])
            print('---------------------------')
            if epoch%1==0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                del m.logprob_history[:]
                del m.rewards[:]
                dic_init()
def deploy_roll():
    iid_li=df_ins[df_ins.mid.notnull()].iid.tolist()
    df_ins_copy=df_ins.copy()
    df_ins_copy.mid.fillna('',inplace=True)
    #shuffle the add
    # add_=df_ins_copy.sort_values(ascending=False,by='mid').iid.tolist()
    del df_ins_copy
    gc.collect()

    bar=Progbar(target=len(iid_li),width=30,interval=0.05)
    for id_,iid in enumerate(iid_li):

        mid_real=dic['imid'][iid]
        if id_==0:
            print(mid_real)
            print(iid)
            print('deployed_roll')

        aid=env.i_a[iid]
        cur=env.a_idx[aid]
        end=run_game(-1,cur,env.not_quick_roll,mid_real)
        bar.update(id_+1)
    quick_roll_save()

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

        print(len(dic['step']))
        # load_checkpoints(fn)
        # for id_,(iid,aid) in enumerate(env.i_a.items()):
        # for id_,(iid,mid) in enumerate(dic['im'].items()):
        for id_,iid in enumerate(dic['iid']):
        # foirwarding
            # cur=env.app[app.aid==aid]
            step=dic['step'][id_]
            # print(iid,mid)

            # if (aid+iid)=='':
            if step==-1:
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

            end=run_game(step,cur,not_quick_roll)
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
