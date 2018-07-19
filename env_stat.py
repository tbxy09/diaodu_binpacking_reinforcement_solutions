import numpy as np
import pandas as pd
from util.threed_view import *

MA_NUM=6000
APP_NUM=9338

class Env_stat():
    def __init__(self,df_machine,df_app_res,df_app_inter,df_ins_sum,verbose):

        def gen_ai_map(df):
            grouped=df[['iid','aid']].groupby('aid')
            a_i={}
            i_a={}
            def sum_():
                return {
                    'iid':np.sum,
                    'num':lambda x: len(x),
                }
            self.df_a_i=grouped.iid.agg(sum_()).reset_index()
            self.df_a_i=pd.merge(self.app,self.df_a_i,on='aid',how='inner')
            for a,i in zip(self.df_a_i.aid,self.df_a_i.num):
                self.a_i[a]=i

            # a_i['app_7189']
            for i,a in zip(df.iid,df.aid):
            #     print(i,a)
                self.i_a[i]=a

        def get_unit_app_resource(app):

            app_cpu=app['cpu'].str.split('|',expand=True)
            app_mem=app['mem'].str.split('|',expand=True)
            app_disk=app['disk']
            return {
                    'c':app_cpu.astype(float).max(axis=1).values
                    ,'m':app_mem.astype(float).max(axis=1).values
                    ,'d':app['disk'].astype(float).values
                    ,'p_pm':app['p'].astype(float).values
                    ,'m_pm':app['m'].astype(float).values
                    ,'pm':app['pm'].astype(float).values
                   }

        self.ret_v_cpu=[]
        self.ret_v_mem=[]
        self.ret_v_disk=[]
        self.ret_v_p=[]
        self.ret_v_m=[]
        self.ret_v_pm=[]
        self.ret_v_cpu_abs=[]
        self.ret_v_mem_abs=[]
        self.ret_app_infer=[]
        def init_deploy_state():
            return {'c':np.zeros(shape=(MA_NUM,98))
                    ,'m':np.zeros(shape=(MA_NUM,98))
                    ,'a':np.zeros(shape=(MA_NUM,))
                    ,'d':np.zeros(shape=(MA_NUM,))
                    ,'ca':np.zeros(shape=(MA_NUM,98))
                    ,'ma':np.zeros(shape=(MA_NUM,98))
                    ,'p_pm':np.zeros(shape=(MA_NUM,))
                    ,'m_pm':np.zeros(shape=(MA_NUM,))
                    ,'pm':np.zeros(shape=(MA_NUM,))
                    }

        self.deploy_state=init_deploy_state()

        self.mn=df_machine.copy().reset_index()
        self.app=df_app_res.copy()
        self.app_state=df_app_res.copy()
        self.app_inter=df_app_inter.copy()
        self.verbose=verbose
        self.cur=0
#         self.cpu_limit=self.mn.cpu.values
#         self.mem_limit=self.mn.mem.values
        self.res_cpu=self.mn.cpu.values
        self.res_mem=self.mn.mem.values

        self.col_req=np.arange(98)
        self.col_cpu_req=np.arange(98)*2
        self.col_mem_req=np.arange(98)*2+1
        # self.env_mem=np.zeros(shape=(self.mn.shape[0],98))
        # self.env_cpu=np.zeros(shape=(self.mn.shape[0],98))
        # self.env_disk=np.zeros(shape=(self.mn.shape[0]))

        self.a_i={}
        self.i_a={}
        gen_ai_map(df_ins_sum)
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

        # used for log
        self.dic={}
        # self.matrix=np.zeros([])
        # self.ab=self.app_inter.groupby('aid').apply(lambda x:x.sort_values(by='v'))
#         pass

    # def save_checkpoints(self):
    #     fn='policy{}.pth.tar'.format(e)
    #     dic['saved_log_probs'].append(log_probs)
    #     dic['rewards'].append(rewards)
    #     # dic['policy_rewards']=[]
    #     # dic['policy_rewards'].append(policy_rewards)
    #     # dic['len'].append(len(self.rewards))
    #     torch.save(dic,fn)
    def load_checkpoints(self,dic):
        self.li_cpu=dic['li_cpu']
        self.li_mem=dic['li_mem']
        self.ret_v_cpu=dic['ret_v_cpu']
        self.ret_v_mem=dic['ret_v_mem']
        self.ret_app_infer=dic['ret_app_infer']

    def pack_plot(self,li,axis=0,verbose=0):
        v_=np.sum(li,axis=axis)
        p=pd.Series(v_)
        if verbose:
            p.plot()
        return v_
    def reset(self):
        self.li_cpu=[]
        self.li_mem=[]
        self.ret_app_infer=[]
        self.ret_v_cpu=[]
        self.ret_v_mem=[]
        self.dic={}
        self.init_matrix()

    def dump(self):

        self.dic['ret_v_cpu']=self.ret_v_cpu
        self.dic['ret_v_mem']=self.ret_v_mem
        self.dic['ret_app_infer']=self.ret_app_infer

        self.dic['li_cpu']=self.li_cpu
        self.dic['li_mem']=self.li_mem
        self.dic['li_disk']=self.li_disk

    def evaluate(self,cur,choice):
#         pass
        import re
        def re_find(text):
            for g,v in self.app_inter.groupby('aid') :
                if re.findall(g,text):
                    for each in v.ab:
                        if re.findall(each,text):
                            print(each)
                            return 1
            return 0

        if choice>self.mn.shape[0]-1:
            # return self.env_cpu.sum(0).sum(0),1
            return 1,1

        choice=self.mn.mid[choice]

        # self.env_cpu,self.env_mem,self.env_app,env_cpu_abs,env_mem_abs,self.env_disk=self.update(cur,choice)

        self.deploy_state=self.update(cur,choice)

#         if any(self.update(cur,choice).max(0)>threshold):

        # print('{0},{1}'.format(self.env_cpu.max(1).argmax(),self.env_cpu.max(1).max()))
        # print('{0}'.format(env_cpu_abs.sum(1).max()))
        # print('{0},{1}'.format(self.env_mem.max(1).argmax(),self.env_mem.max(1).max()))
        # print('{0}'.format(self.env_mem.sum(1).max()))

        self.env_matrix(cur)

        self.dic['matrix']=self.matrix

        # if any(self.env_mem.max(1)>1):
        for k in  ['p_pm','m_pm','pm']:
            if any(self.deploy_state[k])>1:
                print(k ,'end')
                # return self.env_cpu.sum(0).sum(0),1
                return 1,1

        for k in  ['c','m']:
            if any(self.deploy_state[k].max(1))>1:
                print(k ,'end')
                # return self.env_cpu.sum(0).sum(0),1
                return 1,1

        # text=(self.env_app+' ').sum()
        text=(self.deploy_state['a']+' ').sum()

        # ab=self.app_inter.ab.sort_values()
        # r=self.app_inter[['ab']].apply(lambda x: re.findall(x.ab,text),axis=1)
        # r=[re.findall(each,text) for each in self.ab.ab]
                                       # ,axis=1)
        # [re.findall('(app_3432).*?(app_7652).*?(app_8618).*?(app_1300).*?(app_4663).*?(app_8324)',text) for each in ab]
        # r=pd.Series(r).apply(lambda x: len(x)!=0)
        # r=[[ for each in v.ab if re.findall(each,text) ] for g,v in self.app_inter.groupby('aid') if re.findall(g,text)]
        end=re_find(text)
        if end==1:
            print('infer end')
        # if any(r)==True:
            # print('end')

        # return self.env_cpu.sum(0).sum(0),end
        return 1,end

    def update(self,cur,choice):
#         cur is kind of timmer
        def get_cpu_usage(cur,res_cpu):
        #     res_v=res[index]
            ps=self.app[self.col_req].iloc[cur,self.col_cpu_req].astype(float)
            def f(x):
                return pd.Series([x/res_cpu,x],index=['per','v'])
            if self.verbose==1:
                ps.apply(lambda x: x/res_cpu).plot()
            return ps.apply(lambda x: x/res_cpu)
            # return ps.apply(f)

        def get_mem_usage(cur,res_mem):
        #     res_v=res[index]
            ps=self.app[self.col_req].iloc[cur,self.col_mem_req].astype(float)
            if self.verbose==1:
                ps.apply(lambda x: x/res_mem).plot()
            return ps.apply(lambda x: x/res_mem)
#         op_policy=lambda cur: cur*10+1

        def get_fe_nt_usage(cur,key,res_total):
        #     res_v=res[index]
            return self.app[key].iloc[cur].astype(float)/res_total

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

        self.ret_v_p.append(s.apply(lambda x:get_fe_nt_usage(cur,'p',x['p']) if choice==x['mid'] else 0,axis=1).values)
        self.ret_v_m.append(s.apply(lambda x:get_fe_nt_usage(cur,'m',x['m']) if choice==x['mid'] else 0,axis=1).values)
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


    def step_op(self,step):
        self.li_cpu.append(self.unit['c']*np.identity(APP_NUM)[step,:]*-1)
        self.li_mem.append(self.unit['m']*np.identity(APP_NUM)[step,:]*-1)
        self.li_disk.append(self.unit['d']*np.identity(APP_NUM)[step,:]*-1)
        self.li_p.append(self.unit['p_pm']*np.identity(APP_NUM)[step,:]*-1)
        self.li_m.append(self.unit['m_pm']*np.identity(APP_NUM)[step,:]*-1)
        self.li_pm.append(self.unit['pm']*np.identity(APP_NUM)[step,:]*-1)

    def li_stack(self,verbose):
        def env_sum():
            return {
                   # 'cpu':self.pack_plot(self.env_cpu,axis=1,verbose=0)*self.mn.cpu,
                   'c':self.pack_plot(self.deploy_state['c'],axis=1,verbose=0)*self.mn.cpu,
                   'm':self.pack_plot(self.deploy_state['m'],axis=1,verbose=0)*self.mn.mem,
                   'd':self.deploy_state['d']*self.mn.disk,
                   'p_pm':self.deploy_state['p_pm']*self.mn.p,
                   'm_pm':self.deploy_state['m_pm']*self.mn.m,
                   'pm':self.deploy_state['pm']*self.mn.pm
                  }
        li_mem_sum=[]
        li_cpu_sum=[]
        li_disk_sum=[]
        li_p_sum=[]
        li_m_sum=[]
        li_pm_sum=[]

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

        li_cpu_sum.append(env_sum()['c'])
        li_mem_sum.append(env_sum()['m'])
        li_disk_sum.append(env_sum()['d'])
        li_p_sum.append(env_sum()['p_pm'])
        li_m_sum.append(env_sum()['m_pm'])
        li_pm_sum.append(env_sum()['pm'])


        li_cpu_sum.append(np.zeros(14*5))
        li_mem_sum.append(np.zeros(14*5))
        li_disk_sum.append(np.zeros(14*5))
        li_p_sum.append(np.zeros(14*5))
        li_m_sum.append(np.zeros(14*5))
        li_pm_sum.append(np.zeros(14*5))
        self.matrix=np.vstack([np.hstack(li_cpu_sum)
                               ,np.hstack(li_mem_sum)
                               ,np.hstack(li_disk_sum)
                               ,np.hstack(li_p_sum)
                               ,np.hstack(li_m_sum)
                               ,np.hstack(li_pm_sum)])
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

        self.dump()

        # cpu_ret=normalize(cpu_ret)
        # mem_ret=normalize(mem_ret)

        # plt.figure()
        # pd.Series(cpu_ret[:]).plot()
        # pd.Series(mem_ret[:]).plot()

        self.li_stack(verbose)

        # pd.Series(self.matrix[1,:]).plot()

        # cur=cur+1
        # s.apply(lambda x:choice==x['index'],axis=1)
