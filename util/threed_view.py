# %load web_traffic/threed_view.py
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
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
        # ax = Axes3D(fig,
    #         rect=[0, 0, .95, 1], elev=0, azim=0)
    #         rect=[0, 0, 1, 1], elev=48, azim=134)
            # rect=[0, 0, pf_agent.shape[0],pf_agent.shape[1]], elev=48, azim=134)
        # ax=fig.gca(projection='3d')

    # ax.scatter(time_x[:,:,0] ,time_x[:,:,1] )
#     print(d1,d2)
    ax.scatter(d1,d2,pf_agent.reshape(1,-1))
    # for i in range(pf_agent.shape[1]):
        # ax.plot(np.arange(pf_agent.shape[0]),pf_agent[:,i],zs=i,zdir='y')
    # ax.bar(d1,np.ones(d1.shape[0]),d2)
    # ax.plot_wireframe(d1,d2,pf_agent.reshape(1,-1),rstride=2,cstride=2)

    # xs=np.arange(pf_agent.shape[1])
    # verts=[]
    # zs=np.arange(pf_agent.shape[0])
    # for z in zs:
    #     ys = pf_agent[z,:]
    #     # ys[0], ys[-1] = 0, 0
    #     verts.append(list(zip(xs, ys)))

    # poly = PolyCollection(verts)
    # poly.set_alpha(0.7)
    # ax.add_collection3d(poly, zs=zs,zdir='y')

    fig=plt.figure()
    plt.plot(pf_agent)
    return ax
#     ax.plot(d1,d2,pf_agent[:,start:end].reshape(1,-1))
#     d1,d2=get_index(pf_agent[start:end,:])
#     ax.scatter(d1,d2,pf_agent[start:end,:].reshape(1,-1))
#     ax.plot(d1,d2,'xy')


