import pandas as pd
from funtest.test_pathlib import first_try
from util.gen_expand import encode_page_features
def en_vec(df):
    vec7_id={vec:id for id,vec in enumerate(df.vec7.unique())}
    vec5_id={vec:id for id,vec in enumerate(df.vec5.unique())}
    vec4_id={vec:id for id,vec in enumerate(df.vec4.unique())}
    vec3_id={vec:id for id,vec in enumerate(df.vec3.unique())}
    vec1_id={vec:id for id,vec in enumerate(df.vec1.unique())}

    df['vec7']=df.vec7.apply(lambda x:vec7_id[x])
    df['vec5']=df.vec5.apply(lambda x:vec5_id[x])
    df['vec4']=df.vec4.apply(lambda x:vec4_id[x])
    df['vec3']=df.vec3.apply(lambda x:vec3_id[x])
    df['vec1']=df.vec1.apply(lambda x:vec1_id[x])
    col_in_en=[]
    col_in_en+=['vec7']
    col_in_en+=['vec5']
    col_in_en+=['vec4']
    col_in_en+=['vec3']
    col_in_en+=['vec1']
# df_sum_dp.isnull().sum()

# col_in_en=col_in[1:]

    fe=encode_page_features(df[col_in_en])
    return fe
class Preprocess():
    def __init__(self):
        self.data_read()
    def data_read(self):
        base='/mnt/osstb/tianchi/diaodu/'

        fn=first_try(base,'*.csv')

        col_li_aid='aid_cpu_mem_disk_p_m_pm'
        col_li_mid='mid_cpu_mem_disk_p_m_pm'
        col_li_aid=col_li_aid.split('_')
        col_li_mid=col_li_mid.split('_')

        self.df_app_inter=pd.read_csv(fn[0],names=['aid','bid','v'],usecols=None,index_col=None)

        self.df_app_res=pd.read_csv(fn[1],names=list('abcdefg'))

        self.df_app_res=pd.read_csv(fn[1],names=col_li_aid)

        self.df_ins=pd.read_csv(fn[2],names=['iid','aid','mid'])

        self.df_machine=pd.read_csv(fn[3],names=col_li_mid)

        self.df_su=pd.read_csv(fn[4])

# df_app_res['aid']=df_app_res.aid.str.split('_',expand=True)[1]
    def df_app_res_handling(self):
        for c in ['cpu','mem']:
            self.df_app_res=pd.concat( [self.df_app_res,self.df_app_res[c].str.split('|',expand=True)],axis=1)
    def df_machine_handling(self):
        cols=['cpu','mem','disk','p','m','pm']
        cols_rn=[]
        for c in cols:
            val_map={each:str(id) for id,each in enumerate(self.df_machine[c].unique())}
            c_rn='_'.join([c,'en'])
            cols_rn.append(c_rn)
            self.df_machine[c_rn]=self.df_machine[c].map(val_map)

        self.df_machine[cols_rn].sum(axis=1).value_counts()

        self.df_machine['target']=self.df_machine[cols_rn].sum(axis=1)

        self.df_machine['target']
        val_map={each:id for id,each in enumerate(self.df_machine.target.unique())}
        self.df_machine['target']=self.df_machine['target'].map(val_map)

    def make_vector(self,aids):
        vec=np.zeros(df_ins.aid.nunique())



        for id,values in zip(aids.bid,aids.v):
            index=int(id.split('_')[-1])-1
    #         vec[int(id.split('_')[-1])-1]+=values
    #         if id=='app_8129':
    #             print(id,values)
    #             print(vec.max())
            vec[index]+=values
        if len(np.argwhere(vec==7))>0:
            print('----------')
            print(np.argwhere(vec==7))
        vec=vec.astype(int)
    #     return pd.Series([''.join(vec.astype(str)),vec.max()],index=['vec','vec_max'])
        return pd.Series([(vec==7)*vec.T,(vec==5)*vec.T,(vec==4)*vec.T,(vec==3)*vec.T,(vec==1)*vec.T],
                         index['vec7','vec5','vec4','vec3','vec1'])
    def df_app_inter_handling(self):
        nonly=self.df_ins.aid.nunique()
        def make_vector(nonly,aids):
            vec=np.zeros(nonly)

            for id,values in zip(aids.bid,aids.v):
                index=int(id.split('_')[-1])-1
        #         vec[int(id.split('_')[-1])-1]+=values
        #         if id=='app_8129':
        #             print(id,values)
        #             print(vec.max())
                vec[index]+=values
            if len(np.argwhere(vec==7))>0:
                print('----------')
                print(np.argwhere(vec==7))
            vec=vec.astype(int)
        #     return pd.Series([''.join(vec.astype(str)),vec.max()],index=['vec','vec_max'])
            return pd.Series([(vec==7)*vec.T,(vec==5)*vec.T,(vec==4)*vec.T,(vec==3)*vec.T,(vec==1)*vec.T],
                             index['vec7','vec5','vec4','vec3','vec1'])
        cols=['bid','v']
        values=np.vstack([df_app_inter.values,df_app_inter[['bid','aid','v']].values])
        df_app_inter=pd.DataFrame(values,columns=['aid','bid','v'])
        # df_app_inter=df_app_inter.groupby('aid')[cols].apply(lambda x: ':'.join(x.v),axis=1).reset_index()
        self.df_app_inter=self.df_app_inter.groupby('aid')[cols].apply(lambda x: make_vector(nonly,x)).reset_index()

        self.df_app_inter['vec7']=self.df_app_inter.vec7.apply(lambda x:''.join(x.astype(str)))
        self.df_app_inter['vec4']=self.df_app_inter.vec4.apply(lambda x:''.join(x.astype(str)))
        self.df_app_inter['vec3']=self.df_app_inter.vec3.apply(lambda x:''.join(x.astype(str)))
        self.df_app_inter['vec5']=self.df_app_inter.vec5.apply(lambda x:''.join(x.astype(str)))
        self.df_app_inter['vec1']=self.df_app_inter.vec1.apply(lambda x:''.join(x.astype(str)))

        #### here, i freeze the handler into a csv file

        self.df_app_inter.to_csv('./data/diaodu/df_app_inter.csv')

        #### reading from the csv
    def df_app_inter_fromcsv(self):
        self.df_app_inter=pd.read_csv('/opt/playground/data/diaodu/df_app_inter.csv')

    def df_app_inter_pat(self):
        self.df_app_inter[['aid','bid']].sum(axis=1)
        # self.df_app_inter['ab']=self.df_app_inter[['aid','bid','v']].apply(lambda x:'({})'.format(x.aid)
        #                                                          +'.*?({})'.format(x.bid)*(int(x.v)+1),axis=1)

        self.df_app_inter['ab']=self.df_app_inter[['aid','bid','v']].apply(lambda x:'{}'.format(x.aid)
                                                                 +'[^( )^({})]+{}'.format(x.aid,x.bid)*(int(x.v)+1),axis=1)

        self.df_app_inter['ab']=self.df_app_inter[['aid','bid','v']].apply(lambda x:'(^.*?){}'.format(x.aid)
                                                                 +'[^( )]*?{}'.format(x.bid)*(int(x.v)+1),axis=1)
        # df_app_inter['ab']=df_app_inter[['aid','bid']].sum(axis=1)
        # df_app_inter['reg']
        return self.df_app_inter
    def df_sum_ins(self):
        df_sum=pd.merge(self.df_machine,self.df_ins,on='mid',how='outer')
        df_sum=pd.merge(df_sum,self.df_app_res,on='aid',how='outer')
        df_sum['disk']=df_sum['disk'+'_y']/df_sum['disk'+'_x']

        # df_sum_dp['disk'].value_counts()

        df_sum['deploy']=df_sum['mid'].notnull()&df_sum['iid'].notnull()
        # df_sum['deploy']=df_sum['mid'].notnull()

        df_sum=pd.merge(df_sum,df_app_inter_grouped,on='aid',how='outer')

        # del df_app_inter
        # gc.collect()
        # fe=en_vec(df_sum[col_in_en])
        df_sum_ins=df_sum[df_sum['iid'].notnull()]

    def df_app_inter_grouped(self):
        grouped=self.df_app_inter.iloc[:].groupby('aid')

        grouped.first()
        # grouped.get_group('app_100')
        # grouped.get_group('app_100')
        # grouped.boxplot()
        def f(group):
            return pd.DataFrame({
        #         'aid':group.aid,
                'bid':'|'.join(group.bid.tolist()),
                'len': len(group.bid),})
        #         'v': group.v})
        def f(group):
            return{
        #         'aid':group.aid,
                'bid':'|'.join(group.bid.tolist()),
                'len': len(group.bid),
                'v': '|'.join(group.v.values.astype(str))}
        def fa(group):
            return  len(group.bid),
        def fb(group):
            return '|'.join(group.bid.tolist())
        # df_app_inter_grouped=grouped['v','bid'].agg(lambda x:len(x.bid))
        return grouped['v','bid'].agg(f)


    def run_pre(self):
        # self.handlings=[self.df_app_res_handling,self.df_machine_handling,self.df_app_inter_fromcsv]
        self.handlings=[self.df_app_res_handling,self.df_machine_handling,self.df_app_inter_pat]
        for h in self.handlings:
            h()
            # pass

        df_sum=pd.merge(self.df_machine,self.df_ins,on='mid',how='outer')
        df_sum=pd.merge(df_sum,self.df_app_res,on='aid',how='outer')
        df_sum['disk']=df_sum['disk'+'_y']/df_sum['disk'+'_x']

        # df_sum_dp['disk'].value_counts()

        df_sum['deploy']=df_sum['mid'].notnull()&df_sum['iid'].notnull()
        # df_sum['deploy']=df_sum['mid'].notnull()

        df_sum=pd.merge(df_sum,self.df_app_inter_grouped(),on='aid',how='outer')

        # del df_app_inter
        # gc.collect()
        # fe=en_vec(df_sum[col_in_en])
        df_sum_ins=df_sum[df_sum['iid'].notnull()]
        # self.df_app_inter_fromcsv()
        # fe=en_vec(self.df_app_inter)
        return (self.df_app_res.copy(),
                self.df_machine.copy(),
                self.df_ins.copy(),
                self.df_app_inter.copy(),
                df_sum_ins)
