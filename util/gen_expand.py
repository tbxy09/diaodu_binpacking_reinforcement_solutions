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

def submit(ab,fig=0,i=-1):
    fn={'a':fn_a,'b':fn_b}
    ab_s={'a':df_ins_a.shape,'b':df_ins_b.shape}
    abc_s={'a':df_ins_a[df_ins_a.mid.notnull()].shape,'b':df_ins_b[df_ins_b.mid.notnull()].shape}
    log_prob,mid,iid=show_matrix(str(fn[ab][i]),fig)
    su=pd.DataFrame(np.vstack([iid,mid]).T,columns=['iid','mid'])

    su.mid.replace('',float('NaN'),inplace=True)
    su.iid.replace('',float('NaN'),inplace=True)
#     assert su.shape==(ab_s[ab],2)
    print(su[su.mid.notnull()].shape,ab_s[ab],abc_s[ab])
    return su

def evaluate_whole(f,deploy_state,splits,app_inter,proc,num_proc,counter):
    # def myfun(deploy_state,app_inter):
    #     import re
    #     text=(deploy_state['a']+' ').sum()
    #     for g,v in app_inter.groupby('aid') :
    #         if re.findall(g,text):
    #             for each in v.ab:
    #                 if re.findall(each,text):
    #                     counter[1]=counter[1]+1
    #                     print(each)
    for k in  ['c','m','d','p_pm','m_pm','pm']:
        counter[0]=counter[0]+sum(deploy_state[k]>1)

    result=[]

    text=(deploy_state['a']+' ').sum()
    for i in range(num_proc):
        # result.append(proc.apply_async(f,(text,splits[i])))
        result.append(proc.apply_async(f,(text,splits[i])))

    for i in range(num_proc):
        # end,v=result[i].get()
        if result[i].get()==1:
            counter[1] =counter[1]+1
    # myfun(deploy_state,app_inter)

def re_find_y(text,app_inter):
    import re
    p=re.compile('(\s+)')
    text=p.sub(' ',text)

    for g,v in app_inter.groupby('aid') :
        if re.findall(g,text):
            for each in v.ab:
                if re.findall(each,text):
                    # print(each)
                    return 1
    return 0

# def re_find_y(li,app_inter):
#     m=li
#     end=app_inter[['v','ab_encode']].apply(
#                                     lambda x: [m.count(each) for each in set(x.ab_encode)]==[1,x.v],axis=1).sum()
#     if end:
#         return 1

# def re_find_y(text,m,app_inter,n,deploy_state):
def re_find_y(text,m,n,deploy_state,app_inter):
    import re
    # p=re.compile('(\s+)')
    # text=p.sub(' ',text)

    ret=app_inter[['v','ab_encode']].apply(
        lambda x: sum([m.count(each) for each in set(x.ab_encode)]),axis=1).max()

    for g,v in app_inter.groupby('aid') :
        if re.findall(g,text):
            for each in v.ab:
                if re.findall(each,text):
                    # print(each)
                    return 1,ret
    return 0,ret
# def re_find_y(m,app_inter):
#     return app_inter[['v','ab_encode']].apply(
#         lambda x: sum([m.count(each) for each in set(x.ab_encode)]),axis=1).max()

def re_find_whole(text,app_inter):
    import re
    # p=re.compile('(\s+)')
    # text=p.sub(' ',text)

    for g,v in app_inter.groupby('aid') :
        if re.findall(g,text):
            for each in v.ab:
                if re.findall(each,text):
                    # print(each)
                    return 1
    return 0

def li_gen(i,deploy_state):
    li_=deploy_state['a'][i].split('app_')[1:]
    li_=pd.Series(li_,dtype=int).tolist()
    return li_

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
    # mid=ck['step']
    mid=ck['imid']
    iid=ck['iid']
    if env_stat:
        env_stat.matrix=ck['env_dic']['matrix']
        env_stat.deploy_state=ck['env_dic']['deploy_state']
        # return log_prob,[mid[i] for i in range(68219) if mid[i]!=-1], [iid[i] for i in range(68219) if iid[i]!='']
        return mid,iid
    # aid=ck['aid']
    # id_=ck['id_']
    state_dict=ck['state_dict']
    m.load_state_dict(state_dict)
    # env_dic=ck['env_dic']
    print(len([mid[i] for i in range(68219) if mid[i]!=-1]))
    print(len([iid[i] for i in range(68219) if mid[i]!=-1]))
    [mid[i] for i in range(68219) if mid[i]!=-1]
    return log_prob,[mid[i] for i in range(68219) if mid[i]!=-1], [iid[i] for i in range(68219) if iid[i]!='']

def update_history(self,cur,choice):

    def get_cpu_usage_history(cur,res_cpu):
    #     res_v=res[index]
        ps=self.app[self.col_req].iloc[cur,self.col_cpu_req].astype(float)
        if self.verbose==1:
            ps.apply(lambda x: x/res_cpu).plot()
        return ps.apply(lambda x: x/res_cpu)
        # return ps.apply(f)

    def get_mem_usage_history(cur,res_mem):
    #     res_v=res[index]
        ps=self.app[self.col_req].iloc[cur,self.col_mem_req].astype(float)
        if self.verbose==1:
            ps.apply(lambda x: x/res_mem).plot()
        return ps.apply(lambda x: x/res_mem)

    def get_fe_nt_usage(cur,key,res_total):
        return self.unit[key][cur].astype(float)/res_total

    # def get_pm_usage(cur,res_pm):
    # #     res_v=res[index]
    #     ps=self.app['p'].iloc[cur].astype(float)
    #     if self.verbose==1:
    #         ps.apply(lambda x: x/res_pm).plot()
    #     return ps.apply(lambda x: x/res_pm)
    # s=df_machine.iloc[cur-3:cur+3][['index','cpu']]
#         choice=op_policy(cur)
    # s=df_machine.iloc[choice-3:choice+3][['index','cpu']]

    z98=pd.Series(np.zeros(98))
    # v=get_cpu_usage(cur,res_cpu)
    # v=get_mem_usage(cur,res_mem)
    # map_cpu={0:z,1:v}
    # map_mem={0:z,1:v}

    s=self.mn.iloc[:][['mid','cpu','mem','disk','p','m','pm']]

    self.ret_v_cpu.append(s.apply(lambda x:get_cpu_usage(cur,x['cpu']) if choice==x['mid'] else z98,axis=1).values)
    self.ret_v_cpu_abs.append(s.apply(lambda x:get_cpu_usage(cur,x['cpu'])*x['cpu'] if choice==x['mid'] else z98,axis=1).values)
    self.ret_v_mem.append(s.apply(lambda x:get_mem_usage(cur,x['mem']) if choice==x['mid'] else z98,axis=1).values)
    self.ret_v_mem_abs.append(s.apply(lambda x:get_mem_usage(cur,x['mem'])*x['mem'] if choice==x['mid'] else z98,axis=1).values)

    self.ret_v_disk.append(s.apply(lambda x:get_fe_nt_usage(cur,'disk',x['disk']) if choice==x['mid'] else 0,axis=1).values)

    self.ret_v_p.append(s.apply(lambda x:get_fe_nt_usage(cur,'p_pm',x['p']) if choice==x['mid'] else 0,axis=1).values)
    self.ret_v_m.append(s.apply(lambda x:get_fe_nt_usage(cur,'m_pm',x['m']) if choice==x['mid'] else 0,axis=1).values)
    self.ret_v_pm.append(s.apply(lambda x:get_fe_nt_usage(cur,'pm',x['pm']) if choice==x['mid'] else 0,axis=1).values)

    self.ret_app_infer.append(s['mid'].apply(lambda x:self.app.aid[cur] if choice==x else '').values)


    if self.verbose==1:
        threed_view(np.sum(self.ret_v_cpu,0)[int(choice.split('_')[-1])-10:int(choice.split('_')[-1])+10,:].T,end=100)
        threed_view(np.sum(self.ret_v_mem,0)[int(choice.split('_')[-1])-10:int(choice.split('_')[-1])+10,:].T,end=100)
        threed_view(np.sum(self.ret_v_disk,0)[int(choice.split('_')[-1])-10:int(choice.split('_')[-1])+10,:].T,end=100)
        threed_view(np.sum(self.ret_v_p,0)[int(choice.split('_')[-1])-10:int(choice.split('_')[-1])+10,:].T,end=100)
        threed_view(np.sum(self.ret_v_m,0)[int(choice.split('_')[-1])-10:int(choice.split('_')[-1])+10,:].T,end=100)
        threed_view(np.sum(self.ret_v_pm,0)[int(choice.split('_')[-1])-10:int(choice.split('_')[-1])+10,:].T,end=100)
    # df_machine['cpu_deploy'].sum(axis=1).plot()

    return {'c':np.sum(self.ret_v_cpu,0)
            ,'m':np.sum(self.ret_v_mem,0)
            ,'a':np.sum(self.ret_app_infer,0)
            ,'ca':np.sum(self.ret_v_cpu_abs,0)
            ,'ma':np.sum(self.ret_v_mem_abs,0)
            ,'d':np.sum(self.ret_v_disk,0)
            ,'p_pm':np.sum(self.ret_v_p,0)
            ,'m_pm':np.sum(self.ret_v_m,0)
            ,'pm':np.sum(self.ret_v_pm,0)
            }

def update_history2(self,cur,choice):
    def get_usage(cur,key,res_total):
        return self.unit[key][cur].astype(float)/res_total

    s=self.mn.iloc[:][['mid','cpu','mem','disk','p','m','pm']]

    self.ret_v_cpu.append(s.apply(lambda x:get_usage(cur,'c',x['cpu']) if choice==x['mid'] else 0,axis=1).values)
    self.ret_v_cpu_mean.append(s.apply(lambda x:get_usage(cur,'cm',1) if choice==x['mid'] else 0,axis=1).values)
    self.ret_v_mem.append(s.apply(lambda x:get_usage(cur,'m',x['mem']) if choice==x['mid'] else 0,axis=1).values)

    self.ret_v_disk.append(s.apply(lambda x:get_usage(cur,'d',x['disk']) if choice==x['mid'] else 0,axis=1).values)

    self.ret_v_p.append(s.apply(lambda x:get_usage(cur,'p_pm',x['p']) if choice==x['mid'] else 0,axis=1).values)
    self.ret_v_m.append(s.apply(lambda x:get_usage(cur,'m_pm',x['m']) if choice==x['mid'] else 0,axis=1).values)
    self.ret_v_pm.append(s.apply(lambda x:get_usage(cur,'pm',x['pm']) if choice==x['mid'] else 0,axis=1).values)

    self.ret_app_infer.append(s['mid'].apply(lambda x:self.df_a_i.aid[cur] if choice==x else '').values)


    if self.verbose==1:
        threed_view(np.sum(self.ret_v_cpu,0)[int(choice.split('_')[-1])-10:int(choice.split('_')[-1])+10,:].T,end=100)
        threed_view(np.sum(self.ret_v_mem,0)[int(choice.split('_')[-1])-10:int(choice.split('_')[-1])+10,:].T,end=100)
        threed_view(np.sum(self.ret_v_disk,0)[int(choice.split('_')[-1])-10:int(choice.split('_')[-1])+10,:].T,end=100)
        threed_view(np.sum(self.ret_v_p,0)[int(choice.split('_')[-1])-10:int(choice.split('_')[-1])+10,:].T,end=100)
        threed_view(np.sum(self.ret_v_m,0)[int(choice.split('_')[-1])-10:int(choice.split('_')[-1])+10,:].T,end=100)
        threed_view(np.sum(self.ret_v_pm,0)[int(choice.split('_')[-1])-10:int(choice.split('_')[-1])+10,:].T,end=100)
    # df_machine['cpu_deploy'].sum(axis=1).plot()

    return {'c':np.sum(self.ret_v_cpu,0)
            ,'m':np.sum(self.ret_v_mem,0)
            ,'a':np.sum(self.ret_app_infer,0)
            ,'cm':np.sum(self.ret_v_cpu_mean,0)
            ,'d':np.sum(self.ret_v_disk,0)
            ,'p_pm':np.sum(self.ret_v_p,0)
            ,'m_pm':np.sum(self.ret_v_m,0)
            ,'pm':np.sum(self.ret_v_pm,0)
            }

def update_history(self,cur,choice):
    def get_usage(cur,key,res_total):
        return self.unit[key][cur].astype(float)/res_total

    s=self.mn.iloc[:][['mid','cpu','mem','disk','p','m','pm']]

    self.ret_v_cpu.append(s.apply(lambda x:get_usage(cur,'c',x['cpu']) if choice==x['mid'] else 0,axis=1).values)
    self.ret_v_cpu.append(self.deploy_state['c'])

    self.ret_v_cpu_mean.append(s.apply(lambda x:get_usage(cur,'cm',1) if choice==x['mid'] else 0,axis=1).values)
    self.ret_v_cpu_mean.append(self.deploy_state['cm'])

    self.ret_v_mem.append(s.apply(lambda x:get_usage(cur,'m',x['mem']) if choice==x['mid'] else 0,axis=1).values)
    self.ret_v_mem.append(self.deploy_state['m'])

    self.ret_v_disk.append(s.apply(lambda x:get_usage(cur,'d',x['disk']) if choice==x['mid'] else 0,axis=1).values)
    self.ret_v_disk.append(self.deploy_state['d'])

    self.ret_v_p.append(s.apply(lambda x:get_usage(cur,'p_pm',x['p']) if choice==x['mid'] else 0,axis=1).values)
    self.ret_v_p.append(self.deploy_state['p_pm'])

    self.ret_v_m.append(s.apply(lambda x:get_usage(cur,'m_pm',x['m']) if choice==x['mid'] else 0,axis=1).values)
    self.ret_v_m.append(self.deploy_state['m_pm'])

    self.ret_v_pm.append(s.apply(lambda x:get_usage(cur,'pm',x['pm']) if choice==x['mid'] else 0,axis=1).values)
    self.ret_v_pm.append(self.deploy_state['pm'])

    self.ret_app_infer.append(s['mid'].apply(lambda x:self.df_a_i.aid[cur] if choice==x else '').values)
    self.ret_app_infer.append(self.deploy_state['a'])


    if self.verbose==1:
        threed_view(np.sum(self.ret_v_cpu,0)[int(choice.split('_')[-1])-10:int(choice.split('_')[-1])+10,:].T,end=100)
        threed_view(np.sum(self.ret_v_mem,0)[int(choice.split('_')[-1])-10:int(choice.split('_')[-1])+10,:].T,end=100)
        threed_view(np.sum(self.ret_v_disk,0)[int(choice.split('_')[-1])-10:int(choice.split('_')[-1])+10,:].T,end=100)
        threed_view(np.sum(self.ret_v_p,0)[int(choice.split('_')[-1])-10:int(choice.split('_')[-1])+10,:].T,end=100)
        threed_view(np.sum(self.ret_v_m,0)[int(choice.split('_')[-1])-10:int(choice.split('_')[-1])+10,:].T,end=100)
        threed_view(np.sum(self.ret_v_pm,0)[int(choice.split('_')[-1])-10:int(choice.split('_')[-1])+10,:].T,end=100)
    # df_machine['cpu_deploy'].sum(axis=1).plot()

    return {'c':np.sum(self.ret_v_cpu,0)
            ,'m':np.sum(self.ret_v_mem,0)
            ,'a':np.sum(self.ret_app_infer,0)
            ,'cm':np.sum(self.ret_v_cpu_mean,0)
            ,'d':np.sum(self.ret_v_disk,0)
            ,'p_pm':np.sum(self.ret_v_p,0)
            ,'m_pm':np.sum(self.ret_v_m,0)
            ,'pm':np.sum(self.ret_v_pm,0)
            }


def evaluate_copy(self,cur,choice):

    def re_find(text,a):
        import re
        p=re.compile('(\s+)')
        text=p.sub(' ',text)
        if a==0:
            for g,v in self.app_inter.groupby('aid') :
                if re.findall(g,text):
                # if re_findall(g,text):
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
            pat=')|('.join(v_li)
            pat=r'('+pat+')'
            p=re.compile(pat)
            if p.findall(text):
                return 1
                # if re_findall(g,text):
        return 0

    self.ret_init()
    self.li_init()

    # a=np.ones(250)
    # b=np.zeros(250)
    # b[-1]=1

    # self.i=self.i+1
    # return a[self.i-1],b[self.i-1]

    if choice>self.mn.shape[0]-1:
        # return self.env_cpu.sum(0).sum(0),1
        return 1,1

    self.n=(self.n+choice)%MA_NUM
    # print('\n')
    # print(self.n,choice)
    choice=self.mn.mid[self.n]

    # self.env_cpu,self.env_mem,self.env_app,env_cpu_abs,env_mem_abs,self.env_disk=self.update(cur,choice)

    self.deploy_state=self.update(cur,choice,self.n)

    # if any(self.update(cur,choice).max(0)>threshold):
    # print('{0},{1}'.format(self.env_cpu.max(1).argmax(),self.env_cpu.max(1).max()))
    # print('{0}'.format(env_cpu_abs.sum(1).max()))
    # print('{0},{1}'.format(self.env_mem.max(1).argmax(),self.env_mem.max(1).max()))
    # print('{0}'.format(self.env_mem.sum(1).max()))

    self.env_matrix(cur)

    # self.dic['matrix']=self.matrix
    # self.dic['deploy_state']=self.deploy_state

    for k in  ['c','m','d','p_pm','m_pm','pm']:
        if any(self.deploy_state[k]>1):
            pass
            # print('\n')
            # print(k ,'end')
            # return self.env_cpu.sum(0).sum(0),1
            # return 1,1

    # for k in  ['c','m']:
    #     if any(self.deploy_state[k].max(1)>1):
    #         print(k ,'end')
    #         # return self.env_cpu.sum(0).sum(0),1
    #         return 1,1

    # text=(self.env_app+' ').sum()

    text=(self.deploy_state['a']+' ').sum()


    # ab=self.app_inter.ab.sort_values()
    # r=self.app_inter[['ab']].apply(lambda x: re.findall(x.ab,text),axis=1)
    # r=[re.findall(each,text) for each in self.ab.ab]
                                   # ,axis=1)
    # [re.findall('(app_3432).*?(app_7652).*?(app_8618).*?(app_1300).*?(app_4663).*?(app_8324)',text) for each in ab]
    # r=pd.Series(r).apply(lambda x: len(x)!=0)
    # r=[[ for each in v.ab if re.findall(each,text) ] for g,v in self.app_inter.groupby('aid') if re.findall(g,text)]

    a=np.ones(69000)
    b=np.zeros(69000)
    b[-1]=1

    self.i=self.i+1
    return a[self.i-1],b[self.i-1]

    if self.not_quick_roll==1:
        end=re_find(text,1)
        # assert re_find(text,0)==re_find(text,1)
        if end==1:
            print('\ninfer end')
        # return 1,end

    a=np.ones(250)
    b=np.zeros(250)
    b[-1]=1

    self.i=self.i+1
    return a[self.i-1],b[self.i-1]

    # if any(r)==True:
        # print('end')

    # return self.env_cpu.sum(0).sum(0),end

    return 1,0
