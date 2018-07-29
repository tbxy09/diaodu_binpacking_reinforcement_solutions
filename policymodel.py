import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F
# inp=torch.empty(size=(3,1),dtype=torch.float)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Policy(nn.Module):

    def __init__(self):
        super(Policy,self).__init__()
        # self.conv1=nn.Conv2d(24,16,3)
        # self.bn1=nn.BatchNorm2d(16)
        # self.conv2=nn.Conv2d(16,32,3)
        # self.bn2=nn.BatchNorm2d(32)
        # mm1=nn.Conv2d(24,16,3,2)
        # mm1b=nn.Conv2d(4,16,3,2)
        # bn1=nn.BatchNorm2d(16)
        # mm2=nn.Conv2d(16,32,3,2)
        # bn2=nn.BatchNorm2d(32)
        # mm3=nn.Conv2d(32,64,3,2)
        # bn3=nn.BatchNorm2d(64)
        # self.mm1c=nn.Conv2d(4,128,3,3)

        self.mm1c = nn.Conv2d(4,128,kernel_size=7, stride=2, padding=3,
                               bias=False)
        # self.mm1c=conv3x3(4,128,3)

        # mm4=nn.Conv2d(64,128,3,2)
        self.bn4=nn.BatchNorm2d(128)
        self.mm5=conv3x3(128,4,2)
        self.bn5=nn.BatchNorm2d(4)
        # mm6=nn.Conv2d(64,16,3,2)
        # bn6=nn.BatchNorm2d(16)
        # self.fc1=nn.Linear(128,100)
        # self.fc2=nn.Linear(28,6)
        # self.fc3=nn.Linear(99,10)
        # self.fc4=nn.Linear(1248,1000)
        self.fc4=nn.Linear(5832,6000)
        self.relu = nn.ReLU(inplace=True)
        # self.net=nn.Sequential(mm1,bn1,mm2,bn2,mm3,bn3,mm4,bn4)
        # self.netb=nn.Sequential(mm1b,bn1,mm2,bn2,mm3,bn3,mm4,bn4)
        # self.netc=nn.Sequential(mm1c,bn4,mm5)
                                # ,mm5,bn5)
        # self.fc1 = nn.Linear(36, 125)
        # self.fc2 = nn.Linear(125,10)
        self.it = iter(self.parameters())
        self.dic= {}
        self.logprob_history= []
        self.rewards=[]
        # self.h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size)
        # self.c0 = torch.randn(self.num_layers, x.size(0), self.hidden_size)

    def reset(self):
        self.dic={}
        self.logprob_history= []
        self.rewards=[]

    def forward_history(self, x):
        # Set initial states

        # inp=torch.tensor(env.matrix,dtype=torch.float).view(8,107,-1)
        # inp=torch.randn(20,8,50,100)
        x=x.unsqueeze(0)
        o=self.net(x)
        o=o.squeeze(0)

        # fc(o).shape
        # x=torch.randn(2,128)
        # fc(x).shape
        o=self.fc1(o.transpose(0,2))
        o=self.fc2(o.transpose(0,2))
        o=self.fc3(o.transpose(2,1))
        # o.view(-1,4).shape
        # nn.ConstantPad2d?
        # m = nn.ConstantPad2d(2,1)
        # x=torch.randn(1,2,2)
        o=o.squeeze(1)
        o=o.view(1,-1)

        # out= self.fc1(x)
        # out=F.relu(out)
        # Decode hidden state of last time step
        # out = self.fc2(out)
        o=F.softmax(o,dim=1)
        self.o_=Categorical(o)
        # out = F.log_softmax(out,dim=1)
        return self.o_.sample()

    def block1(self,x):
        o = self.mm1c(x)
        o = self.bn4(o)
        o = self.relu(o)
        return o

    def block2(self,x):
        o = self.mm5(x)
        o = self.bn5(o)
        o = self.relu(o)
        return o

    def forward(self, x):
        residual=x
        # Set initial states

        # inp=torch.tensor(env.matrix,dtype=torch.float).view(8,107,-1)
        # inp=torch.randn(20,8,50,100)
        # o=self.net(x)
        o = self.mm1c(x)
        o = self.bn4(o)
        o = self.relu(o)

        o = self.mm5(o)
        o = self.bn5(o)

        # o += residual
        o = self.relu(o)

        o=o.view(1,-1)
        o=o.squeeze(1)
        o=self.fc4(o)
        o = self.relu(o)
        # out = self.fc2(out)
        o=F.softmax(o,dim=1)
        self.o_=Categorical(o)
        # out = F.log_softmax(out,dim=1)
        return self.o_.sample()

    def get_logprob(self,choice):
        # self.logprob_history.append(
        return self.o_.log_prob(choice)

    def save_logprob(self,loss,reward):
        self.logprob_history.append(loss)
        self.rewards.append(reward)

class PolicyX(Policy):

    def __init__(self,p_dic):
        super(PolicyX,self).__init__()
        self.weight_action(init='dict',p_dic=p_dic)

    def expand(self):
        [print("id:{},{}".format(id_,each.shape)) for id_,each in self.state_dict().items()]
        # pd.DataFrame.from_records(self.state_dict())

    def next(self):
        print(next(self.it))

    def init_it(self):
        self.it=iter(self.state_dict().items())

    def weight_init():
        import torch.nn.init as weight_init
        for name, param in self.named_parameters():
            weight_init.normal_(param)
        #     weight_init.uniform_(param)
        #     weight_init.constant_(param,1)
            self.dic[name] = param
        self.load_state_dict(self.dic)

    def weight_action(self,init=None,p_dic=None):
        dic = {}       #we can store the weights in this dict for convenience
        import torch.nn.init as weight_init
        if init=='dict':
            self.load_state_dict(p_dic)
        for name, param in self.state_dict().items():
            if init=='normal':
                weight_init.normal_(param)
            if init=='zeros':
                weight_init.constant_(param,0)
            if init=='uniform':
                weight_init.uniform_(param)
            # dic[name] = torch.tensor(param.data)
        # self.load_state_dict(dic)
        #     weight_init.constant_(param,1)
        return dic

    def save_checkpoint(self,state_dict,is_best,fn):
        torch.save(state_dict,fn)
        if is_best:
            shutil.copyfile(fn,'./model/diao_du/model_best.pth.tar')

    def show(self):
        fig=plt.figure()
        ax=plt.gca(projection='3d')
        for id_,(key,value) in enumerate(self.dic.items()):
            xs=np.arange(dict[key].shape[0])
            ys=value
        #     xs=np.arange(ys.shape[0])
            print(xs.shape,ys.shape)
            ax.bar(xs,ys.data.numpy().T[1],zs=id_,zdir='y')

    def write_file():
        self.save_checkpoint({'state_dict':self.state_dict()},is_best=False,fn='./models/diaodu/policy_{}.pth.tar'.format(self.rewards[-1]))

# inp=torch.randn(4,36)
# m1=nn.Linear(36,125)
# m2=nn.Linear(125,10)
# m=nn.Sequential(m1,m2)
# m=Categorical(inp)
# inp=F.softmax(inp,dim=1)

# m(inp)
# o=F.softmax(m(inp),dim=1)

