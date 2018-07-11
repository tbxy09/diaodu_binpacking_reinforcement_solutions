class Env_stat():
    def __init__(self,df_machine,df_app_res,verbose):
        self.ret_v=[]
        self.mn=df_machine.copy().reset_index()
        self.app=df_app_res.copy()
        self.verbose=verbose
        self.cur=0
#         self.cpu_limit=self.mn.cpu.values
#         self.mem_limit=self.mn.mem.values
        self.res_cpu=self.mn.cpu.values
        self.res_mem=self.mn.mem.values

        self.col_req=np.arange(98)
        self.col_cpu_req=np.arange(98)*2
        self.col_mem_req=np.arange(98)*2+1
#         pass
    def evaluate(self,cur,choice):
#         pass
        env=self.update(cur,choice)
#         if any(self.update(cur,choice).max(0)>threshold):
        print(''.format(self.env.max(0)))
        if any(env.max(0)>1):
            print('end')

        return self.env.sum(0).sum(0)
    def update(self,cur,choice):
#         cur is kind of timmer
        def get_cpu_usage(cur,index):
        #     res_v=res[index]
            ps=self.app[self.col_req].iloc[cur,self.col_cpu_req].astype(float)
            if self.verbose==1:
                ps.apply(lambda x: x/self.res_cpu[index]).plot()
            return ps.apply(lambda x: x/self.res_cpu[index])

        def get_mem_usage(cur,index):
        #     res_v=res[index]
            ps=self.app[self.col_req].iloc[cur,self.col_mem_req].astype(float)
            if self.verbose==1:
                ps.apply(lambda x: x/self.res_mem[index]).plot()
            return ps.apply(lambda x: x/self.res_mem[index])
#         op_policy=lambda cur: cur*10+1

        # s=df_machine.iloc[cur-3:cur+3][['index','cpu']]
#         choice=op_policy(cur)
        # s=df_machine.iloc[choice-3:choice+3][['index','cpu']]

        z=pd.Series(np.zeros(98))
        v=get_cpu_usage(cur,choice)
        map_={0:z,1:v}

        s=self.mn.iloc[:][['index','cpu']]

        self.ret_v.append(s.apply(lambda x:map_[choice==x['index']],axis=1).values)

        if self.verbose==1:
            threed_view(np.sum(ret_v,0)[choice-10:choice+10,:].T,end=100)
        # df_machine['cpu_deploy'].sum(axis=1).plot()

        return np.sum(self.ret_v,0)

# cur=cur+1
# s.apply(lambda x:choice==x['index'],axis=1)
