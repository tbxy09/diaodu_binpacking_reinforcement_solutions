def f(x):
    return {
        'cpu_v':lambda x:x.aid,
        'per':lambda x:x.aid
    }
def f(x):
    return pd.Series([x.aid,x.cpu] ,index=['aid_','cpu_'])

# df_app_res[0].iloc[:,0].astype(float).apply(lambda x: x/10)
# df_app_res[0].iloc[:,0].astype(float).apply(f)
# df_app_res.groupby('cpu').apply(f)
df_app_res[['aid','cpu']].apply(f,axis=1)
