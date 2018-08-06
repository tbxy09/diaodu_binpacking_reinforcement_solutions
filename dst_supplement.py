it=iter(dataloader)
ss,label=next(it)
ss=torch.tensor(ss,dtype=torch.float)
label=torch.tensor(label,dtype=torch.long)

# plt.plot(ss.data.numpy()[1][:,-1])
plt.plot(ss.data.numpy()[0])