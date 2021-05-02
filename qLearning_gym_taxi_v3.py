import gym
import numpy as np
import random
import matplotlib.pyplot as plt


env=gym.make("Taxi-v3").env

# Q TABLE
q_table=np.zeros([env.observation_space.n,env.action_space.n])
#HYPER PARAMETER (bizim tarafımızdan ayarlananabilen parametreler)
alpha=0.1
gamma=0.9
epsilon=0.1
#PLOTTING METRIX (Görselleştirmek için değişkenler)
reward_list=[]#ne kadar ödül kazandı
dropouts_list=[]
episode_number=10000
for i in range(1,episode_number):
    #initialize environment
    state=env.reset()
    reward_count=0
    dropouts=0
    while True:
        #exploit vs explore to find action
        #%10=explore, %90 exploit
        if random.uniform(0,1)<epsilon:
            action=env.action_space.sample()#rastgele action veriyoruz keşfetmek için(yolcu indir, sağa git gibi)
        else:
            action=np.argmax(q_table[state])#en yüksek değeri olan yerin sutun değerini döndürür yani buda action verir 4 numaralı action gibi
        #action process and take reward/ take observation(gözlem)
        next_state,reward,done,_=env.step(action)#actionu gerçekler ve parametreler döndürür
        #Q learning function
        old_value=q_table[state,action]
        next_max= np.max( q_table[next_state])
        next_value=(1-alpha)*old_value+alpha*(reward+gamma*next_max)
        #Q Table update 
        q_table[state,action]=next_value
        #update state
        state=next_state
        #find wrong dropouts
        if reward==-10:
            dropouts+=1
        if done:
            break
        
        reward_count+=reward
    
    if i%10==0:
        dropouts_list.append(dropouts)
        reward_list.append(reward_count)
        print("Episode: {}, reward{}, wrong dropout {}".format(i,reward_count,dropouts))
        
#%% visualize
fig,axs=plt.subplots(1,2) #1 satırda iki plot yarat. 
axs[0].plot(reward_list) #1. satır 1. sutun plot
axs[0].set_xlabel("episode") #x ekseninin ismi
axs[0].set_ylabel("reward") #y ekseninin ismi

axs[1].plot(dropouts_list)#2. satır 1.sutun plot
axs[1].set_xlabel("episode")#x ekseninin ismi
axs[1].set_ylabel("dropouts")#y ekseninin ismi

axs[0].grid(True)
axs[1].grid(True)
plt.show()













