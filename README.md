# diaodu_binpacking_reinforcement_solutions
this not wining solution, and actually ,still in develop, but upload any way, check my diary wiki for more info


Here is diary after the diaodu

### np.identity

I use a np.identity as filter to put the app unit info to the place where it should be

```

li.append(each_unit_update_t*np.identity[:,step])
np.stack(li)

```

I thouhgt it is a great idea,however, here is the problem, np.idenity as dynamic created inside running it has the app exactly size APP_NUM=more than 9000 what I am thinking, why keep creating np class(this cause my program extremely slow), I thought it like a np constant value, but it is not

here is two examples,you can try

### another topic I want to share,is the gradient expoding and gradient vanish issue

rewards = 0.1*1 the most effective one to solve the gradient explode the gradient explode will finally happen in my case, it almost same inp input,

change the mid def -> as a moving step

if the sample action will sample out 0,if sample out 0 means stay at the begining, if other value, means move_step%MACHINE_NUM , a loop series if mid is just the mid, the app will keep backward the positive reaction, to make the weight too hight to saturate the softmax

the aj blog, is very helpful

### The Classic Solutions

https://github.com/skjolber/3d-bin-container-packing

https://github.com/benmaier/binpacking
### What I thought today(08-15) see what happend

I need to work first on some very simple bin-packing example with my policy gradient solution

### the positvie rewards vs the negative rewards

I finally choose a negative rewards loss backward solution, why? and I am still confused about the finall loss calculation when it is a negative rewards, do I need to flip the sign of loss? I choose not, because it is like I just set a different zero axis reference for numeric the rewards, like (1,0) when 0 as ref, but it because (0,-1) when 1 as ref
