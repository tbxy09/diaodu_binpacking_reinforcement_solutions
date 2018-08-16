b=df_ins_sum_a['cpu_39'].values
# for id_,v in enumerate(b):
b_dic={id_:float(v) for id_,v in enumerate(b)}

# b
bins=binpacking.to_constant_bin_number(b_dic,6000)

# some=random.randint(1,600)
def get_max(bins):
    def fun(id_,each):
    #     pd.DataFrame.from_dict(each,orient='index')
        li=[]
        li.append(list(each.keys()))
        li.append(list(each.values()))
        li.append([id_]*len(each))
        li.append([sum(each.values())]*len(each))

        return np.vstack(li).T
    #     return np.stack(each.values(),each.keys,[1])
    #     df['a']=id_
    #     return df.values
    #     df['a']=each

    ret=[fun(id_,each) for id_,each in enumerate(bins[:])]

    ret=np.vstack(ret)

    # ar=np.vstack(ret)
    # ret[72].shape

    df=pd.DataFrame(ret).rename(columns={0:'iid',1:'v',2:'gid',3:'vs'})
    # df['iid']=df[0].astype(int)
    # df.rename(columns={1:})

    df['vs'].max()
