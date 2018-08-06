import numpy as np
import time
from multiprocessing import Pool
a=np.ones(100000)
a[50000]=0
def f(x):
    # return x*x
    if a[x-1]==0:
        print(1)
        return 1
def fa(x):
    for each in x:
        if a[each]==0:
            print(1)
            return 1
    return 0
def fb(x):
    for each in x:
        if a[x]==0:
            print(1)
            return 1
    return 0
def wrap(p,fa):
    result=p.apply_async(fa,(np.arange(n),))
    # result.get(timeout=2)
    result.get()
    result=p.apply_async(fa,(np.arange(n,2*n),))
    # result.get(timeout=2)
    result.get()
    p.close()
    p.join()
    # p.map(f,np.arange(1000000))


if __name__=='__main__':
    n=300000
    a=np.ones(2*n)
    a[50]=0
    start=time.time()
    # map(f,np.arange(n))
    for each in np.arange(2*n):
        f(each)
    end=time.time()

    print(end-start)
    p=Pool(2)

    start=time.time()
    # p.map(f,np.arange(n))
    result=p.apply_async(fa,(np.arange(n),))
    # result.get(timeout=2)
    ret=result.get()
    print(ret)
    result=p.apply_async(fa,(np.arange(n,2*n),))
    # result.get(timeout=2)
    ret=result.get()
    p.close()
    p.join()
    end=time.time()
    print(end-start)
    print(ret)

    p=Pool(2)
    start=time.time()
    wrap(p,fa)
    end=time.time()
    print(end-start)

    # p.map(f,np.arange(10000000))
