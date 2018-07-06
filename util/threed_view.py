# %load web_traffic/threed_view.py
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
def get_index(inp):
    a=np.zeros_like(inp)
    d1=np.where(a>-1)[0]
    d2=np.where(a>-1)[1]
    return d1,d2


def threed_view(pf_agent,end,start=0):
    d1,d2=get_index(pf_agent[start:end,:])

    fig=plt.figure()
    ax = Axes3D(fig,
#         rect=[0, 0, .95, 1], elev=0, azim=0)
        rect=[0, 0, 1, 1], elev=48, azim=134)

    # ax.scatter(time_x[:,:,0] ,time_x[:,:,1] )
    ax.scatter(d1,d2,pf_agent[start:end,:].reshape(1,-1))
#     d1,d2=get_index(pf_agent[start:end,:])
#     ax.scatter(d1,d2,pf_agent[start:end,:].reshape(1,-1))
    # ax.plot(d1,d2,'-y')


def threed_view(pf_agent,end,start=0,ax=None):
    pf_agent=pf_agent[start:end,:]
    d1,d2=get_index(pf_agent)

    if ax==None:
        fig=plt.figure()
        ax = Axes3D(fig,
    #         rect=[0, 0, .95, 1], elev=0, azim=0)
    #         rect=[0, 0, 1, 1], elev=48, azim=134)
            rect=[0, 0, 1, 1], elev=48, azim=134)

    # ax.scatter(time_x[:,:,0] ,time_x[:,:,1] )
#     print(d1,d2)
    ax.scatter(d1,d2,pf_agent.reshape(1,-1))
    fig=plt.figure()
    plt.plot(pf_agent)
    return ax
#     ax.plot(d1,d2,pf_agent[:,start:end].reshape(1,-1))
#     d1,d2=get_index(pf_agent[start:end,:])
#     ax.scatter(d1,d2,pf_agent[start:end,:].reshape(1,-1))
#     ax.plot(d1,d2,'xy')


