import numpy as np
import pandas as pd
from util.threed_view import *
from util.gen_expand import proc_split
import gc
import time

MA_NUM=6000
APP_NUM=9338

# INST_NUM=68219
# INST_NUM=68224

NUM_PROC=8

class Env_stat():
    def __init__(self,df_machine,df_app_res,df_app_inter,df_ins_sum,verbose):

        def gen_ai_map(df):
            grouped=df[['iid','aid']].groupby('aid')
            a_i={}
            i_a={}
            a_idx={}
            def sum_():
                return {
                    'iid':np.sum,
                    'num':lambda x: len(x),
                }
            self.df_a_i=grouped.iid.agg(sum_()).reset_index()
            self.df_a_i=pd.merge(self.app[['aid','cpu']],self.df_a_i,on='aid',how='inner')
            for a,i in zip(self.df_a_i.aid,self.df_a_i.num):
                a_i[a]=i

            for idx,a in zip(self.df_a_i.index,self.df_a_i.aid):
                a_idx[a]=idx
            # a_i['app_7189']
            for i,a in zip(df.iid,df.aid):
            #     print(i,a)
                i_a[i]=a
            return a_i,a_idx,i_a

        def get_unit_app_resource(app):

            app_cpu=app['cpu'].str.split('|',expand=True)
            app_mem=app['mem'].str.split('|',expand=True)
            app_disk=app['disk']
            return {
                    'c':app_cpu.astype(float).max(axis=1).values
                    ,'cm':app_cpu.astype(float).mean(axis=1).values
                    ,'m':app_mem.astype(float).max(axis=1).values
                    ,'d':app['disk'].astype(float).values
                    ,'p_pm':app['p'].astype(float).values
                    ,'m_pm':app['m'].astype(float).values
                    ,'pm':app['pm'].astype(float).values
                   }
        # def init_deploy_state():
        #     def init_str():
        #             x=np.empty_like(np.zeros(shape=(MA_NUM,)),dtype='O')
        #             x.fill('')
        #             # x=np.empty_like(self.app_inter['ab_encode'])
        #             # x.fill([])
        #             # x=x[:MA_NUM]

        #             return x

        #     return {'c':np.zeros(shape=(MA_NUM,))
        #             ,'m':np.zeros(shape=(MA_NUM,))
        #             ,'a':init_str()
        #             ,'a_encode':np.zeros(shape=(MA_NUM,))
        #             ,'d':np.zeros(shape=(MA_NUM,))
        #             ,'cm':np.zeros(shape=(MA_NUM,))
        #             ,'p_pm':np.zeros(shape=(MA_NUM,))
        #             ,'m_pm':np.zeros(shape=(MA_NUM,))
        #             ,'pm':np.zeros(shape=(MA_NUM,))
        #             }
        self.n=0
        self.i=0
        # print(self.p)
        self.ret_v_cpu=[]
        self.ret_v_mem=[]
        self.ret_v_disk=[]
        self.ret_v_p=[]
        self.ret_v_m=[]
        self.ret_v_pm=[]
        self.ret_v_cpu_mean=[]
        self.ret_app_infer=[]

        self.counter=[0,0]


        # self.deploy_state=init_deploy_state()

        self.mn=df_machine.copy().reset_index()
        self.app=df_app_res.copy()
        self.app_state=df_app_res.copy()
        self.app_inter=df_app_inter.copy()


        self.verbose=verbose
        self.cur=0
#         self.cpu_limit=self.mn.cpu.values
#         self.mem_limit=self.mn.mem.values
        # self.res_cpu=self.mn.cpu.values
        # self.res_mem=self.mn.mem.values

        self.col_req=np.arange(98)
        self.col_cpu_req=np.arange(98)*2
        self.col_mem_req=np.arange(98)*2+1
        # self.env_mem=np.zeros(shape=(self.mn.shape[0],98))
        # self.env_cpu=np.zeros(shape=(self.mn.shape[0],98))
        # self.env_disk=np.zeros(shape=(self.mn.shape[0]))

        self.a_i={}
        self.i_a={}
        self.a_idx={}
        self.a_i,self.a_idx,self.i_a=gen_ai_map(df_ins_sum)
        del df_ins_sum
        gc.collect()

        li=[]
        a=self.app_inter.aid.apply(lambda x: [self.a_idx[x]]).tolist()
        b=self.app_inter[['bid','v']].apply(lambda x: [self.a_idx[x.bid]]*(x.v+1),axis=1).tolist()
        b=pd.Series(b)
        a=pd.Series(a)
        self.app_inter['ab_encode']=pd.Series(np.sum([a,b],axis=0))

        self.deploy_state=self.init_deploy_state()
        # this is base block
        # self.unit_c,self.unit_m,self.unit_d,self.unit_p,self.unit_m,self.unit_pm=get_unit_app_resource(self.app)
        self.unit=get_unit_app_resource(self.app)


        self.li_cpu=[]
        self.li_mem=[]
        self.li_disk=[]
        self.li_p=[]
        self.li_m=[]
        self.li_pm=[]
        self.init_matrix()
        self._not_quick_roll =0

        # used for log
        self.dic={}
        self.clean4room()
        # self.matrix=np.zeros([])
        # self.ab=self.app_inter.groupby('aid').apply(lambda x:x.sort_values(by='v'))
#         pass
        self.mn_identity={each:np.identity(MA_NUM)*(1/(self.mn[each].values + np.finfo(np.float32).eps)) for each in ['cpu','mem','disk','p','m','pm']}
        self.mn_identity['cm']=np.identity(MA_NUM)

        self.regex_gen()

    def m(self, p_value):
        if self._not_quick_roll!=p_value:
            print(p_value)

    @property
    def not_quick_roll(self):
        return self._not_quick_roll

    @not_quick_roll.setter
    def not_quick_roll(self, value):
        self.m(value)
        self._not_quick_roll = value

    def init_deploy_state(self):
        def init_str():
            # x=np.empty_like(self.app_inter['ab_encode'])
            # x.fill([])
            # x=x[:MA_NUM]
            x=np.empty_like(np.zeros(shape=(MA_NUM,)),dtype='O')
            x.fill('')

            return x
        return {'c':np.zeros(shape=(MA_NUM,))
                ,'m':np.zeros(shape=(MA_NUM,))
                ,'a':init_str()
                ,'a_encode':np.zeros(shape=(MA_NUM,))
                ,'d':np.zeros(shape=(MA_NUM,))
                ,'cm':np.zeros(shape=(MA_NUM,))
                ,'p_pm':np.zeros(shape=(MA_NUM,))
                ,'m_pm':np.zeros(shape=(MA_NUM,))
                ,'pm':np.zeros(shape=(MA_NUM,))
                }

    def li_init(self):

        self.li_cpu=[]
        self.li_mem=[]
        self.li_disk=[]
        self.li_p=[]
        self.li_m=[]
        self.li_pm=[]

    def ret_init(self):

        self.ret_v_cpu=[]
        self.ret_v_mem=[]
        self.ret_v_disk=[]
        self.ret_v_p=[]
        self.ret_v_m=[]
        self.ret_v_pm=[]
        self.ret_v_cpu_mean=[]
        self.ret_app_infer=[]

    def save_checkpoints(self):
        # fn='policy{}.pth.tar'.format(e)
        # dic['saved_log_probs'].append(log_probs)
        # dic['rewards'].append(rewards)
        self.dic['counter']=self.counter
        self.dic['matrix']=self.matrix
        self.dic['deploy_state']=self.deploy_state
        # dic['policy_rewards']=[]
        # dic['policy_rewards'].append(policy_rewards)
        # dic['len'].append(len(self.rewards))
        # torch.save(dic,fn)

    def clean4room(self):
        del self.app
        gc.collect()

    def load_checkpoints(self,dic):
        # self.li_cpu=dic['li_cpu']
        # self.li_mem=dic['li_mem']
        # self.ret_v_cpu=dic['ret_v_cpu']
        # self.ret_v_mem=dic['ret_v_mem']
        # self.ret_app_infer=dic['ret_app_infer']
        self.matrix=dic['matrix']
        self.deploy_state=dic['deploy_state']

    def pack_plot(self,li,axis=0,verbose=0):
        v_=np.sum(li,axis=axis)
        p=pd.Series(v_)
        if verbose:
            p.plot()
        return v_

    def pack_plot_max(self,li,axis=0,verbose=0):
        v_=np.max(li,axis=axis)
        p=pd.Series(v_)
        if verbose:
            p.plot()
        return v_

    def reset(self):
        self.ret_init()

        self.n=0
        self.i=0
        self.li_cpu=[]
        self.li_mem=[]
        self.li_disk=[]
        self.li_p=[]
        self.li_m=[]
        self.li_pm=[]
        self.counter=[0,0]

        # self.dic={}
        self.init_matrix()
        self.deploy_state=self.init_deploy_state()

    def dump(self):

        self.dic['ret_v_cpu']=self.ret_v_cpu
        self.dic['ret_v_mem']=self.ret_v_mem
        self.dic['ret_app_infer']=self.ret_app_infer

        self.dic['li_cpu']=self.li_cpu
        self.dic['li_mem']=self.li_mem
        self.dic['li_disk']=self.li_disk

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

    def regex_gen(self):
        import re
        g_li=[]
        v_li=[]
        for g,v in self.app_inter.groupby('aid') :
            g_li.append(g)
            for each in v.ab:
                v_li.append(each)
        pat_v=')|('.join(v_li)
        # pat_g=')|('.join(g_li)
        pat=pat_v
        pat=r'('+pat+')'
        self.p=re.compile(pat)

    def evaluate(self,cur,choice,proc,f,mid_real):

        self.ret_init()
        self.li_init()

        # a=np.ones(250)
        # b=np.zeros(250)
        # b[-1]=1

        # self.i=self.i+1
        # return a[self.i-1],b[self.i-1]

        if choice>self.mn.shape[0]-1:
            return 1,1
        # print(choice,mid_real)
        int_mid_real=0
        if mid_real=='ff':
            int_mid_real=-1
        if choice==-1|int_mid_real==-1:
            if int_mid_real==-1:
                self.n=(self.n+choice)%MA_NUM
                choice=self.mn.mid[self.n]
                self.deploy_state=self.update(cur,choice,self.n)
            if choice==-1:
                self.n=self.mn.mid.tolist().index(mid_real)
                self.deploy_state=self.update(cur,mid_real,self.n)
        else:
            self.n=(self.n+choice)%MA_NUM
            choice=self.mn.mid[self.n]
            n_minus=self.mn.mid.tolist().index(mid_real)
            # choice_minus=self.mn.mid[n_minus]
            self.deploy_state=self.update(cur,choice,self.n,mid_real,n_minus)

        # print(choice)


        self.env_matrix(cur)
        # if not self.not_quick_roll:
            # return 1,0

        for k in  ['c','m','d','p_pm','m_pm','pm']:
            # if any(self.deploy_state[k]>1):
            if self.deploy_state[k][self.n]>1:
                # print('\n')
                # print(k ,'end')
                self.counter[0]=self.counter[0]+1
                return 0,1

        # text=(self.deploy_state['a']+' ').sum()

        text=self.deploy_state['a'][self.n]

        li_=self.deploy_state['a'][self.n].split('app_')[1:]
        m=pd.Series(li_,dtype=int).tolist()

        # if self.not_quick_roll==1:
            # if use_ma:
            #     v_li=[]
            #     for g,v in self.app_inter.groupby('aid') :
            #         if re.findall(g,text):
            #             for each in v.ab:
            #                 v_li.append(each)

            # startf=time.time()
            # end=self.re_find(text,0)
        ignore=0

        if not ignore:
            qsplit=self.app_inter
            unit=(len(qsplit)-len(qsplit)%NUM_PROC)/NUM_PROC
            splits=proc_split(qsplit,unit,NUM_PROC)

            result=[]

            for i in range(NUM_PROC):
                # result.append(proc.apply_async(f,(text,splits[i])))
                result.append(proc.apply_async(f,(text,m,self.n,self.deploy_state,splits[i])))

            end_flag=0

            for i in range(NUM_PROC):
                end,v=result[i].get()
                # if result[i].get()==1:
                if end:
                    end_flag=1
                    # print('\ninfer end')
                    # print('\ninfer end')

                    # print(v)
                    # self.counter[1]=self.counter[1]+1
                if v>self.deploy_state['a_encode'][self.n]:
                    self.deploy_state['a_encode'][self.n]=v

            if end_flag:
                self.counter[1]=self.counter[1]+1
                return 0,1
                # if result[i].get()==1:

            # result=p.apply_async(self.re_find,(text,app_inter_a))
            # result.get()
            # assert re_find(text,0)==re_find(text,1)
            # endf=time.time()
            # print('\ninfer end {}'.format(endf-startf))

        # a=np.ones(250)
        # b=np.zeros(250)
        # b[-1]=1

        # self.i=self.i+1
        # return a[self.i-1],b[self.i-1]

        # if any(r)==True:
            # print('end')

        # return self.env_cpu.sum(0).sum(0),end

        return 1,0

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



    def update(self,cur,choice,choice_idx,choice_minus=None,choice_idx_minus=None):
        def get_usage(cur,key,res_total):
            return self.unit[key][cur].astype(float)/res_total

        s=self.mn.iloc[:][['mid','cpu','mem','disk','p','m','pm']]
        add=[]
        for each in ['cpu','cm','mem','disk','p','m','pm']:
            # self.ret_v_cpu.append(s.apply(lambda x:get_usage(cur,'c',x['cpu']) if choice==x['mid'] else 0,axis=1).values)
            name_map={'cpu':'c','cm':'cm','mem':'m','disk':'d','p':'p_pm','m':'m_pm','pm':'pm'}
            if choice_idx_minus==None:
                add.append([
                        get_usage(cur,name_map[each],1)*self.mn_identity[each][choice_idx,:],
                        self.deploy_state[name_map[each]]])
            else:
                add.append([(-1)*get_usage(cur,name_map[each],1)*self.mn_identity[each][choice_idx_minus,:],
                        get_usage(cur,name_map[each],1)*self.mn_identity[each][choice_idx,:],
                        self.deploy_state[name_map[each]]])

        ref=[]
        # ref.append(s.apply(lambda x:get_usage(cur,'c',x['cpu']) if choice==x['mid'] else 0,axis=1).values)
        # ref.append(self.deploy_state['c'])

        [self.ret_v_cpu.append(each) for each in add[0]]
        # self.ret_v_cpu.append(add[0][1])
        # self.ret_v_cpu.append(add[0][2])
        # assert np.any(self.ret_v_cpu[0]==ref[0])

        ref=[]
        # ref.append(s.apply(lambda x:get_usage(cur,'cm',1) if choice==x['mid'] else 0,axis=1).values)
        # ref.append(self.deploy_state['cm'])

        [self.ret_v_cpu_mean.append(each) for each in add[1]]
        # self.ret_v_cpu_mean.append(add[1][1])
        # self.ret_v_cpu_mean.append(add[1][2])
        # assert np.any(self.ret_v_cpu_mean[0]==ref[0])

        ref=[]
        # ref.append(s.apply(lambda x:get_usage(cur,'m',x['mem']) if choice==x['mid'] else 0,axis=1).values)
        # ref.append(self.deploy_state['m'])

        [self.ret_v_mem.append(each) for each in add[2]]

        # self.ret_v_mem.append(add[2][1])
        # self.ret_v_mem.append(add[2][2])
        # assert np.any(self.ret_v_mem[0]==ref[0])

        ref=[]
        # ref.append(s.apply(lambda x:get_usage(cur,'d',x['disk']) if choice==x['mid'] else 0,axis=1).values)
        # ref.append(self.deploy_state['d'])
        [self.ret_v_disk.append(each) for each in add[3]]
        # self.ret_v_disk.append(add[3][1])
        # self.ret_v_disk.append(add[3][2])
        # assert np.any(self.ret_v_disk[0]==ref[0])

        ref=[]
        # ref.append(s.apply(lambda x:get_usage(cur,'p_pm',x['p']) if choice==x['mid'] else 0,axis=1).values)
        # ref.append(self.deploy_state['p_pm'])
        [self.ret_v_p.append(each) for each in add[4]]

        # self.ret_v_p.append(add[4][1])
        # self.ret_v_p.append(add[4][2])
        # assert np.any(self.ret_v_p[0]==ref[0])

        ref=[]
        # ref.append(s.apply(lambda x:get_usage(cur,'m_pm',x['m']) if choice==x['mid'] else 0,axis=1).values)
        # ref.append(self.deploy_state['m_pm'])
        [self.ret_v_m.append(each) for each in add[5]]

        # self.ret_v_m.append(add[5][1])
        # self.ret_v_m.append(add[5][1])
        # assert np.any(self.ret_v_m[0]==ref[0])

        ref=[]
        # ref.append(s.apply(lambda x:get_usage(cur,'pm',x['pm']) if choice==x['mid'] else 0,axis=1).values)
        # ref.append(self.deploy_state['pm'])
        [self.ret_v_pm.append(each) for each in add[6]]

        # self.ret_v_pm.append(add[6][1])
        # assert np.any(self.ret_v_pm[0]==ref[0])

        if choice_idx_minus!=None:
            self.deploy_state['a'][choice_idx_minus]=''.join(self.deploy_state['a'][choice_idx_minus].split(self.df_a_i.aid[cur],1))
        self.ret_app_infer.append(s['mid'].apply(lambda x:self.df_a_i.aid[cur] if choice==x else '').values)
        self.ret_app_infer.append(self.deploy_state['a'])

        # if choice_idx_minus!=None:
        #     print('pass')
        #     pass
        #     # self.deploy_state['a'][choice_idx_minus]=''.join(self.deploy_state['a'][choice_idx_minus].split(self.df_a_i.aid[cur],1))
        # self.ret_app_infer_encode.append(s['mid'].apply(lambda x:[cur] if choice==x else []))
        # self.ret_app_infer_encode.append(self.deploy_state['a_encode'])

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
                ,'a_encode':self.deploy_state['a_encode']
                ,'cm':np.sum(self.ret_v_cpu_mean,0)
                ,'d':np.sum(self.ret_v_disk,0)
                ,'p_pm':np.sum(self.ret_v_p,0)
                ,'m_pm':np.sum(self.ret_v_m,0)
                ,'pm':np.sum(self.ret_v_pm,0)
                }

    def step_op(self,step):
        self.li_cpu.append(self.unit['c']*np.identity(APP_NUM)[step,:]*-1)
        self.li_cpu.append(self.matrix[0][:APP_NUM])

        self.li_mem.append(self.unit['m']*np.identity(APP_NUM)[step,:]*-1)
        self.li_mem.append(self.matrix[1][:APP_NUM])

        self.li_disk.append(self.unit['d']*np.identity(APP_NUM)[step,:]*-1)
        self.li_disk.append(self.matrix[2][:APP_NUM])

        self.li_p.append(self.unit['p_pm']*np.identity(APP_NUM)[step,:]*-1)
        self.li_p.append(self.matrix[3][:APP_NUM])

        self.li_m.append(self.unit['m_pm']*np.identity(APP_NUM)[step,:]*-1)
        self.li_m.append(self.matrix[4][:APP_NUM])

        self.li_pm.append(self.unit['pm']*np.identity(APP_NUM)[step,:]*-1)
        self.li_pm.append(self.matrix[5][:APP_NUM])

    def li_stack(self,verbose):
        def env_sum():
            return {
                   # 'cpu':self.pack_plot(self.env_cpu,axis=1,verbose=0)*self.mn.cpu,
                   # 'c':self.pack_plot_max(self.deploy_state['c'],axis=1,verbose=0)*self.mn.cpu,
                   'c':self.deploy_state['c']*self.mn.cpu,
                   # 'm':self.pack_plot_max(self.deploy_state['m'],axis=1,verbose=0)*self.mn.mem,
                   'm':self.deploy_state['m']*self.mn.mem,
                   'd':self.deploy_state['d']*self.mn.disk,
                   'p_pm':self.deploy_state['p_pm']*self.mn.p,
                   'm_pm':self.deploy_state['m_pm']*self.mn.m,
                   'pm':self.deploy_state['pm']*self.mn.pm,
                   'a_encode':self.deploy_state['a_encode']
                  }
        li_mem_sum=[]
        li_cpu_sum=[]
        li_disk_sum=[]
        li_p_sum=[]
        li_m_sum=[]
        li_pm_sum=[]
        li_a_encode=[]

        # if verbose:
            # cpu_ret=self.pack_plot(self.li_cpu,verbose)
            # mem_ret=self.pack_plot(self.li_mem,verbose)
            # disk_ret=self.pack_plot(self.li_disk,verbose)
            # p_ret=self.pack_plot(self.li_p,verbose)
            # m_ret=self.pack_plot(self.li_m,verbose)
            # pm_ret=self.pack_plot(self.li_pm,verbose)
            # plt.figure()
            # pd.Series(cpu_ret[:]).plot()
            # pd.Series(mem_ret[:]).plot()
            # pd.Series(disk_ret[:]).plot()

        li_cpu_sum.append(self.pack_plot(self.li_cpu))
        li_mem_sum.append(self.pack_plot(self.li_mem))
        li_disk_sum.append(self.pack_plot(self.li_disk))
        li_p_sum.append(self.pack_plot(self.li_p))
        li_m_sum.append(self.pack_plot(self.li_m))
        li_pm_sum.append(self.pack_plot(self.li_pm))
        li_a_encode.append(np.zeros(APP_NUM))

        li_cpu_sum.append(env_sum()['c'])
        li_mem_sum.append(env_sum()['m'])
        li_disk_sum.append(env_sum()['d'])
        li_p_sum.append(env_sum()['p_pm'])
        li_m_sum.append(env_sum()['m_pm'])
        li_pm_sum.append(env_sum()['pm'])
        li_a_encode.append(env_sum()['a_encode'])


        li_cpu_sum.append(np.zeros(14*5))
        li_mem_sum.append(np.zeros(14*5))
        li_disk_sum.append(np.zeros(14*5))
        li_p_sum.append(np.zeros(14*5))
        li_m_sum.append(np.zeros(14*5))
        li_pm_sum.append(np.zeros(14*5))
        li_a_encode.append(np.zeros(14*5))
        self.matrix=np.vstack([np.hstack(li_cpu_sum)
                               ,np.hstack(li_mem_sum)
                               ,np.hstack(li_disk_sum)
                               ,np.hstack(li_p_sum)
                               ,np.hstack(li_m_sum)
                               ,np.hstack(li_pm_sum)
                               ,np.hstack(li_a_encode)])
        # plt.figure()
    def init_matrix(self,verbose=0):


        from gen_expand import min_float
        from gen_expand import normalize


        # li_cpu.append(unit_c*np.random.randint(3,5,size=(unit_v.shape[0],)))
        self.li_cpu.append(self.unit['c']*self.df_a_i.num)
        # li_mem.append(unit_m*np.random.randint(3,5,size=(unit_v.shape[0],)))
        self.li_mem.append(self.unit['m']*self.df_a_i.num)
        self.li_disk.append(self.unit['d']*self.df_a_i.num)
        self.li_p.append(self.unit['p_pm']*self.df_a_i.num)
        self.li_m.append(self.unit['m_pm']*self.df_a_i.num)
        self.li_pm.append(self.unit['pm']*self.df_a_i.num)

        # cpu_ret=normalize(cpu_ret)
        # mem_ret=normalize(mem_ret)
        # disk_ret=normalize(disk_ret)
        # cpu_init=cpu_ret
        # mem_init=mem_ret
        # cpu_init=cpu_ret
        # cpu_ret=cpu_ret/(cpu_init+min_float(np.float16))
        # plt.figure()
        self.li_stack(verbose)

    def env_matrix(self,step,verbose=0):
        from gen_expand import normalize

        self.step_op(step)

        # self.dump()

        # cpu_ret=normalize(cpu_ret)
        # mem_ret=normalize(mem_ret)

        # plt.figure()
        # pd.Series(cpu_ret[:]).plot()
        # pd.Series(mem_ret[:]).plot()

        self.li_stack(verbose)

        # pd.Series(self.matrix[1,:]).plot()

        # cur=cur+1
        # s.apply(lambda x:choice==x['index'],axis=1)
