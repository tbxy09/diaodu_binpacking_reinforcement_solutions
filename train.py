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
from gen_expand import re_find_y,evaluate_whole,re_find_whole
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
parser.add_argument('--epoch',type=int,default=553)
parser.add_argument('--inst-num',type=int,default=0)
parser.add_argument('--verbose',type=int,default=1)
parser.add_argument('--fn',type=str,default='')
parser.add_argument('--base',type=str,default='')
parser.add_argument('--use-cache',type=int,default=0)
parser.add_argument('--only-backward',type=int,default=0)
parser.add_argument('--run-id',type=str,default='')
parser.add_argument('--ab',type=str,default='')
parser.add_argument('--log-interval', type=int, default=8, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--dump-interval', type=int, default=2000, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--gamma', type=float, default=0.89, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--non-roll', type=int, default=1,metavar='N')
args = parser.parse_args()
verbose=args.verbose
print(args.gamma)

print('train')
inst_num_dic={'a':68219,'b':68224}
num_life={'b':40000,'a':15000}
        # checkpoint=torch.load('./policyquick_roll.pth.tar')
roll_file_dic={'a':'/data2/a_/policyquick_roll.pth.tar','b':'/data2/b_/policyquick_roll.pth.tar'}
base_dic={'a':'/mnt/osstb/tianchi/diaodu','b':'/mnt/osstb/diaodu'}
INST_NUM=inst_num_dic[args.ab]
NUM_LIFE=num_life[args.ab]
torch.manual_seed(args.seed)

if args.run_id not in os.listdir('/data2/run/'):os.mkdir('/data2/run/'+args.run_id)
# if not args.only_backward:
if not args.epoch:
    assert os.listdir('/data2/run/'+args.run_id)==[]

pre_processor=Preprocess(base_dic[args.ab])
df_app_res,df_machine,df_ins,df_app_inter,df_ins_sum=pre_processor.run_pre()
del pre_processor
gc.collect()
proc=Pool(NUM_PROC)


env=Env_stat(df_machine,df_app_res,df_app_inter,df_ins_sum,verbose=0)
print(env.matrix.shape)
del df_machine,df_app_res,df_app_inter,df_ins_sum
gc.collect()


from policymodel import Policy
m=Policy()
# m.weight_action(init='normal')
optimizer=optim.Adam(m.parameters(),lr=0.001)
dic={}

use_cuda = torch.cuda.is_available()

log_loss=[]
log_reward=[]

def dic_init(fn=None):
    def op_(i,a,key):
        dic[key][i]=a
    if fn==None:
        dic['saved_log_probs']=[]
        dic['rewards'] =[]
        dic['iid']=pd.Series(np.array(['']*INST_NUM))
        dic['mid']=pd.Series(np.array(['']*INST_NUM))
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
    fn='/data2/run/{0}/policy{1}.pth.tar'.format(run_id,e)
    dic['saved_log_probs']=log_probs
    env.save_checkpoints()
    # dic['rewards']=rewards
    dic['env_dic']=env.dic
    dic['rewards']=rewards
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
        # logprob=m.get_logprob(choice)
        # print(loss)

        # ret_=torch.tensor([0],dtype=torch.float)

        # for each in loss:
            # ret_=ret_+each

    #     print(ret_)
        if not reward:
            # m.save_logprob(choice,reward)
            m.save_logprob(choice,-1)
            # log_loss.append(m.logprob_history[-1])
            # log_reward.append(m.rewards[-1])
            # print(log_loss,log_reward)
    return end

#     loss=digits.to(torch.float).dot(m.get_logprob(digits)*-1)*rewards

def get_frame():
    # print(env.matrix.shape)
    return torch.tensor(env.matrix,dtype=torch.float).view(7,4,107,-1)
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

def train():
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

    # for epoch in count(1):
    # epoch=args.epoch

    for epoch in range(args.epoch,args.epoch+100):

        m.logprob_history=[]
        m.rewards=[]
        env.reset()

        # checkpoint=torch.load('./policyquick_roll.pth.tar')
        print('load game')
        checkpoint=torch.load(roll_file_dic[args.ab])
        env.load_checkpoints(checkpoint['env_dic'])
        li_a_encode=[]
        li_a_encode.append(np.zeros(APP_NUM))
        li_a_encode.append(env.deploy_state['a_encode'])
        li_a_encode.append(np.zeros(14*5))
        env.matrix= np.vstack([env.matrix,np.hstack(li_a_encode)])
        print(env.matrix.shape)
        print(env.deploy_state['a_encode'].max())
        del checkpoint
        gc.collect()
        if epoch==args.epoch:
            if args.fn:
                print('load policy')
                checkpoint=torch.load(args.fn)
                m.load_state_dict(checkpoint['state_dict'])
                del checkpoint
                gc.collect()
        if epoch!=args.epoch:
            print('load MID')
            checkpoint=torch.load('/data2/run/{}/policy{}_only_dic.pth.tar'.format(args.run_id,epoch-1))
            iid_li=checkpoint['iid']
            del checkpoint
            gc.collect()
        # this is the place to load the policy checkpoint


        # _,df_ins_sum['iid_num']=df_ins_sum.iid.str.split('_').str
        # df_ins_sum['iid_num']=df_ins_sum['iid_num'].astype(int)

        # df_ins_sum.iid_num.sort_values()
        # for id_,(iid,mid) in enumerate(dic['im'].items()):
        if epoch==args.epoch:
            iid_li=top_level_batch()

        origin_len=len(iid_li)


        bar=Progbar(target=len(iid_li)+NUM_LIFE,width=30,interval=0.05)
        update_id=0
        log_iid='ff'
        # np.random.shuffle(iid_li)

        for id_,iid in enumerate(iid_li):
            if epoch==args.epoch:
                if args.only_backward:
                    print('only_backward in {}'.format(epoch))
                    break
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

            if epoch==2:
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

            # e='_'.join([str(epoch),str(iid)])
            if verbose:
                if (len(m.rewards)+1)%args.dump_interval==0:
                # if (len(m.rewards)+1)%100==0:
                    e='_'.join([str(epoch),iid])
                    # save_checkpoints(m.rewards,m.logprob_history,e,iid,id_,args.run_id)
                    # if log_loss:
                    if m.logprob_history:
                        # save_checkpoints(log_reward,log_loss,e,iid,id_,args.run_id)
                        save_checkpoints(m.rewards,m.logprob_history,e,iid,id_,args.run_id)
                    # print(m.rewards[0],m.logprob_history[0])
                    # print(m.rewards[-1],m.logprob_history[-1])
                    del m.rewards[:]
                    del m.logprob_history[:]
                    del log_loss[:]
                    del log_reward[:]

            # if (len(iid_li)-origin_len) >NUM_LIFE:
            #     print('\nlen break')
            #     break

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
                update_id=update_id+1
                # bar.update(id_)
                # print(iid)
                # print(dic['im'][iid])
                # print(dic['im'][iid])
            else:

                # dic['step'][id_]=step
                dic['imid'][iid]=env.mn.mid[env.n]
                # dic['iid'][id_]=iid
                # update_id=update_id+1
                # bar.update(update_id)

            dic['iid'][id_]=iid
            dic['mid'][id_]=env.mn.mid[env.n]
            # dic['imid'][iid]=env.mn.mid[env.n]

            bar.update(id_+1)

            if (len(iid_li)-origin_len) >NUM_LIFE:
                print('\nlen break')
                break

        fn=first_try('/data2/run/{}'.format(args.run_id),'policy{}_*'.format(epoch))
        print(fn)
        if epoch%1==0:
            if epoch==args.epoch:
                if args.only_backward:
                    print('only_backward in {}'.format(epoch))
                    update_id =(1000-1)*len(fn)
                    print(update_id)
            assert (update_id-len(m.rewards))%(args.dump_interval-1)==0
            rewards = []
            # m.rewards=[1]*(update_id+1)
            temp=[-1]*(update_id)
            # print(len(m.rewards),m.rewards[0],m.rewards[-1])
            # ipdb.set_trace()
            # log_rewards = []
            # log_saved = []
            R=0
            # for r in m.rewards[::-1]:
            for r in temp:
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
            rewards = rewards/10
            print(rewards.shape)
            # print(rewards[:10])

            loss_li=[]

            # log_saved.append(m.logprob_history.data.numpy())
            # log_saved.append(torch.cat(m.logprob_history).data.numpy())


            # if epoch==0&args.only_backward==1
                    # fn=first_try('/data2/run/{}'.format(args.run_id),'policy{}_*'.format(args.epoch))
            fn=first_try('/data2/run/{}'.format(args.run_id),'policy{}_*'.format(epoch))
            fn.sort(key=lambda x:x.stat().st_mtime)
            k_acc=0
            for i_,each in enumerate(fn):
                print(each)
                loader= torch.load(str(each))
                # for (log_prob,r_value) in zip(loader['saved_log_probs'],rewards[i_:(i_+1)*len(loader['saved_log_probs'])]):

                for k,log_prob in enumerate(loader['saved_log_probs']):
                    loss_li.append(-log_prob*rewards[k_acc+k])
                k_acc=k_acc+(k+1)

            print(len(loss_li))


            loss = torch.cat(loss_li).sum()

            loss_li=[]

            for log_prob,r_value in zip(m.logprob_history,rewards[-len(m.logprob_history):]):
            # for log_prob,r_value in zip(log_loss,rewards[-len(log_loss):]):
                loss_li.append(-log_prob*r_value)

            if loss_li:
                loss=torch.add(loss,torch.cat(loss_li).sum())
            else:
                print('the second loss_li is empty')

            print('\n---------------------------')
            print(loss)
            print('\n---------------------------')

            # log_saved.append(loss.data.numpy())

            del loss_li[:]

            # if loss_old==loss:
                # print('loss stay the same')
                # torch.manual_seed(args.seed+100)
                # m=Policy()
                # optimizer=Adam(m.parameters(),lr=0.015)
            # if epoch%args.log_interval==0:
            # if verbose:
            #     if epoch%1==0:
            #,         save_checkpoints(log_rewards,log_saved,epoch,0,0,args.run_id)

            env.evaluate_at_the_end(re_find_whole,proc,NUM_PROC)

            print('\n---------------------------')
            print(playing_len,len(rewards),update_id,env.counter[0],env.counter[1])
            print(env.evl_counter[0],env.evl_counter[1])
            print('---------------------------')
            if epoch%1==0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                save_checkpoints(log_rewards,log_saved,epoch,0,0,args.run_id)
                ck=torch.load('/data2/run/n/policy_cont/policy1_inst_62788.pth.tar')
                submit(ck,args.run_id,args.epoch)

                del m.logprob_history[:]
                del m.rewards[:]
                del log_loss[:]
                del log_reward[:]
                dic_init()

def deploy_roll():
    dic_init()
    m.logprob_history=[]
    m.rewards=[]
    env.reset()
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
    print(env.counter)
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
def top_level_batch():
    df_ins_copy=df_ins.copy()

    df_ins_copy['mid_null']=df_ins_copy.mid.isnull()
    df_ins_copy.groupby('mid_null').get_group(True)

    gp=df_ins_copy.groupby('mid_null')

    ba=gp.get_group(False).iid.tolist()
    np.random.shuffle(ba)

    del df_ins_copy
    gc.collect()

    return ba+gp.get_group(True).iid.tolist()

def reput(df):
    up_limit=list(set(df.iid.value_counts()))
    up_limit.sort(reverse=False)
    print(up_limit)
    stack=[]
    for each in up_limit:
        f1=df.groupby('iid').filter(lambda x: len(x) == each)
        ret=f1.groupby('iid').apply(lambda x: x.mid.values[-1])
    #     print(ret.shape)
        if ret.shape[0]:
            print(ret.shape)
            stack.append(ret)
    return pd.concat(stack).reset_index()

def submit(ck,run_id,epoch):
    fn.append('/data2/{}/policy{}.pth.tar'.format(run_id,epoch))
    su_path={'a':'./a_/su_{}.csv'.format(run_id),
         'b':'./b_/su_{}.csv'.format(run_id),
         'ab':'./ab_/su_{}.csv'.format(run_id)
        }

    # log_prob,mid,iid=ck_parser(fn[-1],m,env_stat)
    su=pd.DataFrame(np.vstack([ck['iid'],ck['mid']]).T,columns=['iid','mid'])

    su.mid.replace('',float('NaN'),inplace=True)
    su.iid.replace('',float('NaN'),inplace=True)
#     assert su.shape==(ab_s[ab],2)
    print(su[su.mid.notnull()].shape,df_ins[df_ins.mid.notnull()].shape,df_ins.shape)

    reput_df=reput(su)

    assert reput_df.shape[0]==df_ins[df_ins.mid.notnull()].shape[0]

    # to_csv
    reput_df.rename(columns={0:'mid'},inplace=True)
    reput_df.to_csv(su_path[args.ab], sep=",", index=False, header=None,line_terminator='\n')

    # to_torch_dic
    dic['iid']=reput_df.iid.values.tolist()
    dic['mid']=reput_df.mid.values.tolist()
    torch.save(dic,'/data2/run/{}/policy{}_only_dic.pth.tar'.format(run_id,epoch))


import torch.multiprocessing as mp
from gen_expand import re_find_y
if use_cuda:
    m=m.cuda()
if args.non_roll:
    fn=[]
    train()
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
    # quick_roll()
    # quick_roll_save()
    deploy_roll()
# train()
