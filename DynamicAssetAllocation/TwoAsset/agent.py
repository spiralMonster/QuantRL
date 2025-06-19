import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from pylab import plt,mpl
from collections import deque
from scipy.optimize import minimize
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,LSTM,Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

plt.style.use("seaborn-v0_8")
mpl.rcParams["figure.dpi"]=300
mpl.rcParams["savefig.dpi"]=300
mpl.rcParams["font.family"]="serif"

model_dir_path=r"/home/spiralmonster/Projects/ReinforcementLearningForFinance/DynamicAssetAllocation/TwoAsset/Models" ## Adjust Path

class Agent:
    def __init__(
        self,
        env,
        model_config,
        optimizer_config,
        batch_size,
        buffer_size,
        gamma,
        epsilon,
        epsilon_min,
        epsilon_decay

    ):
        self.env=env
        self.model_config=model_config
        self.optimizer_config=optimizer_config
        self.batch_size=batch_size
        self.buffer_size=buffer_size
        self.gamma=gamma
        self.epsilon=epsilon
        self.epsilon_min=epsilon_min
        self.epsilon_decay=epsilon_decay
        self.memory=deque(maxlen=self.buffer_size)
        
        self.create_model()
        self.model_dir_path=model_dir_path
        
    def create_model(self):
        inp1=Input(shape=(self.env.number_lags+1,2),dtype=tf.float64)
        inp2=Input(shape=(self.env.number_lags+1,1),dtype=tf.float64)
        inp3=Input(shape=(4,),dtype=tf.float64)
        inp4=Input(shape=(5,),dtype=tf.float64)

        model1_config=self.model_config["Model_1"]
        model2_config=self.model_config["Model_2"]
        model3_config=self.model_config["Model_3"]
        model4_config=self.model_config["Model_4"]

        x1=inp1
        x2=inp2
        x3=inp3
        x4=inp4

        for config in model1_config:
            x1=LSTM(
                units=config["units"],
                activation=config["activation"],
                kernel_initializer=config["kernel_initializer"],
                kernel_regularizer=config["kernel_regularizer"],
                return_sequences=config["return_sequences"]
            )(x1)

        for config in model2_config:
            x2=LSTM(
                units=config["units"],
                activation=config["activation"],
                kernel_initializer=config["kernel_initializer"],
                kernel_regularizer=config["kernel_regularizer"],
                return_sequences=config["return_sequences"]
            )(x2)

        for config in model3_config:
            x3=Dense(
                units=config["units"],
                activation=config["activation"],
                kernel_initializer=config["kernel_initializer"],
                kernel_regularizer=config["kernel_regularizer"]
            )(x3)

        for config in model4_config:
             x4=Dense(
                units=config["units"],
                activation=config["activation"],
                kernel_initializer=config["kernel_initializer"],
                kernel_regularizer=config["kernel_regularizer"]
            )(x4)

        x=Concatenate(axis=-1)([x1,x2,x3,x4])

        final_model_config=self.model_config["Final_Model"]
        for config in final_model_config:
            x=Dense(
                units=config["units"],
                activation=config["activation"],
                kernel_initializer=config["kernel_initializer"],
                kernel_regularizer=config["kernel_regularizer"]
            )(x)

        model=Model(inputs=[inp1,inp2,inp3,inp4],outputs=x)

        optimizer=Adam(
            learning_rate=self.optimizer_config["learning_rate"],
            beta_1=self.optimizer_config["beta_1"],
            beta_2=self.optimizer_config["beta_2"],
        )

        model.compile(
            optimizer=optimizer,
            loss="mse"
        )

        self.model=model

    def get_model_architecture(self):
        self.model.summary()

    
    def optimal_action(self,state):
        bnds=[(0,1)]
        def f(state,x):
            state[2][2]=x[0]
            state[2][3]=1-x[0]
            return self.model.predict([np.expand_dims(np.array(state[0]),axis=0),
                                       np.expand_dims(np.array(state[1]),axis=0),
                                       np.expand_dims(np.array(state[2]),axis=0),
                                       np.expand_dims(np.array(state[3]),axis=0)],
                                       verbose=False)[0][0]

        action=minimize(lambda x: -f(state,x),0.5,bounds=bnds,method="Nelder-Mead")['x'][0]
        return action
        
    
    def act(self,state):
        if random.random()<self.epsilon:
            action=self.env.action_space.sample()

        else:
            action=self.optimal_action(state)
            
        return action


    def step(self,action):
        curr_xt=self.env.final_data["Xt"].iloc[self.env.index]
        curr_yt=self.env.final_data["Yt"].iloc[self.env.index]
        
        self.env.index+=1
        send_report=False

        if self.env.index==1:
            self.xt=action
            self.yt=1-action
            pl=0.0
            pl_percent=0.0
            reward=0.0

        else:
            new_xt=self.env.final_data["Xt"].iloc[self.env.index]
            new_yt=self.env.final_data["Yt"].iloc[self.env.index]

            new_portfolio_value=(self.xt*self.env.portfolio_value*(new_xt/curr_xt)+
                                 self.yt*self.env.portfolio_value*(new_yt/curr_yt)
                                )

            pl=new_portfolio_value-self.env.portfolio_value
            pl_percent=pl/new_portfolio_value
            reward=pl
            

            self.env.portfolio_value=new_portfolio_value
            self.xt=action
            self.yt=1-action

        if self.env.index==self.env.steps-1:
            done=True
            send_report=True

        else:
            done=False

        
        next_state=self.env.get_state()
        
        self.env.pl.append(pl)
        self.env.pl_percent.append(pl_percent)
        self.env.pvalue.append(self.env.portfolio_value)
        self.env.xt_values.append(self.xt)
        

        if send_report:
            mse=MeanSquaredError()
            report={
                "MSE":mse(self.env.pl,self.env.predicted_pl).numpy(),
                "Pvalue":self.env.portfolio_value,
                "xt":self.xt
            }
            
        else:
            report={}
            

        return next_state,reward,done,report
        

    def qvalue(self,next_state):
        action=self.optimal_action(next_state)

        next_state[2][2]=action
        next_state[2][3]=1-action

        value=self.model.predict([np.expand_dims(np.array(next_state[0]),axis=0),
                                       np.expand_dims(np.array(next_state[1]),axis=0),
                                       np.expand_dims(np.array(next_state[2]),axis=0),
                                       np.expand_dims(np.array(next_state[3]),axis=0)],
                                       verbose=False)[0][0]
        return value


    def replay(self,steps_per_episode):
        data_X1=[]
        data_X2=[]
        data_X3=[]
        data_X4=[]
        data_Y=[]

        data_batch=random.sample(self.memory,self.batch_size)
        for (state,action,next_state,reward,done) in data_batch:
            if not done:
                target=reward
                reward_next=self.qvalue(next_state)
                target+=self.gamma*reward_next
                
                data_X1.append(state[0])
                data_X2.append(state[1])
                data_X3.append(state[2])
                data_X4.append(state[3])
                data_Y.append(target)

        batch_size=len(data_Y)
        
        data_X1=np.array(data_X1)
        data_X2=np.array(data_X2)
        data_X3=np.array(data_X3)
        data_X4=np.array(data_X4)
        data_Y=np.array(data_Y)
        
        self.model.fit([data_X1,data_X2,data_X3,data_X4,],data_Y,batch_size=batch_size,epochs=steps_per_episode,verbose=False)

        if self.epsilon>self.epsilon_min:
            self.epsilon*=self.epsilon_decay

    
    def train_agent(self,episodes,steps_per_episode,training_version,verbose=True):
        self.training_episodes=episodes
        
        self.trewards=list()
        self.end_pvalue=list()
        self.pl_per_episode=list()
        self.pvalue_per_episode=list()
        self.pl_percent_per_episode=list()
        self.xt_values_per_episode=list()
        self.end_xt=list()
        self.mse_error=list()
        
        for ep in range(1,episodes+1):
            state,done=self.env.reset()
            total_reward=0

            while not done:
                pred_pl=self.model.predict([np.expand_dims(np.array(state[0]),axis=0),
                                       np.expand_dims(np.array(state[1]),axis=0),
                                       np.expand_dims(np.array(state[2]),axis=0),
                                       np.expand_dims(np.array(state[3]),axis=0)],
                                       verbose=False)[0][0]
                
                self.env.predicted_pl.append(pred_pl)
                
                action=self.act(state)
                next_state,reward,done,report=self.step(action)
                total_reward+=reward
                self.memory.append(
                    [state,action,next_state,reward,done]
                )

                state=next_state

            mse_error=report["MSE"]
            pvalue_last=report["Pvalue"]
            xt=report["xt"]
            
            self.pl_per_episode.append(self.env.pl)
            self.pl_percent_per_episode.append(self.env.pl_percent)
            self.pvalue_per_episode.append(self.env.pvalue)
            self.xt_values_per_episode.append(self.env.xt_values)
            
            self.trewards.append(total_reward)
            self.mse_error.append(mse_error)
            self.end_pvalue.append(pvalue_last)
            self.end_xt.append(xt)
            
            

            if len(self.memory)>self.batch_size:
                self.replay(steps_per_episode)

            if verbose:
                if ep%10==0:
                    info=f"Episode : {ep}/{episodes}| Reward: {total_reward}| Risky Asset weight: {xt}|"
                    info+=f" MSE:{mse_error}| Epsilon: {self.epsilon}"
                    print(info)

            if ep==episodes:
                print("Training Agent completed...")
                if self.env.env_type=="simulated":
                    model_name=f"simulated_model_version_{training_version}.keras"
                    model_path=os.path.join(self.model_dir_path,model_name)
                    self.model.save(model_path)

                else:
                    model_name=f"model_version_{training_version}.keras"
                    model_path=os.path.join(self.model_dir_path,model_name)
                    
                    self.model.save(model_path)
                    print(f"Model saved at : {model_path}")

                    
    def sample_indices(self,num_to_sample):
        return random.sample(range(self.training_episodes),num_to_sample)
        
    
    def training_plots(self,num_plots=5):
        time_steps=list(range(1,self.env.steps))

        sampled_indices=self.sample_indices(num_plots)
        for ind in sampled_indices:
            pvalue_data=self.pvalue_per_episode[ind]
        
            data=self.env.final_data[["Xt","Yt"]].iloc[:-1]
            data["Pvalue"]=pvalue_data
        
            data.plot(figsize=(10,6),style=["b","g","r"])
            plt.title(f"Training Episode: {ind+1}| Risky Asset(Xt)| Risk Free Asset(Yt)| Portfolio(Pvalue)")
            plt.xlabel("Time steps")
            plt.ylabel("Price")
            plt.legend()
            plt.show()
        
        sampled_indices=self.sample_indices(num_plots)
        for ind in sampled_indices:
            pl_data=self.pl_per_episode[ind]
            data=pd.DataFrame(pl_data,columns=["Profit-Loss"],index=time_steps)
            
            data.plot(figsize=(10,6),style=["b"])
            plt.xlabel("Time steps")
            plt.ylabel("Profit-Loss")
            plt.title(f"Training Episode: {ind}| Profit-Loss")
            plt.show()
        
        
        sampled_indices=self.sample_indices(num_plots)
        for ind in sampled_indices:
            pl_percent_data=self.pl_percent_per_episode[ind]
            data=pd.DataFrame(pl_percent_data,columns=["Profit-Loss%"],index=time_steps)
            
            data.plot(figsize=(10,6),style=["g"])
            plt.xlabel("Time steps")
            plt.ylabel("Profit-Loss%")
            plt.title(f"Training Episode: {ind}| Profit-Loss%")
            plt.show()
            
        
        sampled_indices=self.sample_indices(num_plots)
        for ind in sampled_indices:
            xt_values_data=self.xt_values_per_episode[ind]
            data=pd.DataFrame(xt_values_data,columns=["xt"],index=time_steps)
        
            data.plot(figsize=(10,6),style=["r"])
            plt.xlabel("Time Steps")
            plt.ylabel("Weight")
            plt.title(f"Training Episode: {ind}| Risky Asset Weight in Portfolio")
            plt.show()

    def episode_plots(self):
        episodes=list(range(1,self.training_episodes+1))
        
        plt.plot(episodes,self.trewards,lw=1.0,c="b")
        plt.xlabel("Training Episodes")
        plt.ylabel("Reward")
        plt.xlim(0,self.training_episodes+2)
        plt.ylim(min(self.trewards)-1,max(self.trewards)+1)
        plt.title("Episodes VS Rewards")
        plt.show()
        
        plt.plot(episodes,self.mse_error,lw=1.0,c="r")
        plt.xlabel("Training Episodes")
        plt.ylabel("Mean Squared Error")
        plt.xlim(0,self.training_episodes+2)
        plt.ylim(0,max(self.mse_error))
        plt.title("Episodes VS MSE")
        plt.show()
        
        plt.plot(episodes,self.end_pvalue,lw=1.0,c="g")
        plt.xlabel("Training Episodes")
        plt.ylabel("Portfolio Value")
        plt.title("Episodes VS Portofolio Value At Episode End")
        plt.show()
        
        plt.plot(episodes,self.end_xt,lw=1.0,c="c")
        plt.xlabel("Training Episodes")
        plt.ylabel("Weight")
        plt.xlim(0,self.training_episodes+2)
        plt.ylim(0,1)
        plt.title("Episodes VS Risky Asset Weight in Portfolio at Episode End")
        plt.show()

    def test_agent(self,verbose=True,plots=True):
        state,done=self.env.reset()
        total_reward=0
        while not done:
            predicted_pl=self.model.predict([np.expand_dims(np.array(state[0]),axis=0),
                                       np.expand_dims(np.array(state[1]),axis=0),
                                       np.expand_dims(np.array(state[2]),axis=0),
                                       np.expand_dims(np.array(state[3]),axis=0)],
                                       verbose=False)[0][0]
                
            self.env.predicted_pl.append(predicted_pl)
            
            action=self.optimal_action(state)
            next_state,reward,done,report=self.step(action)
            total_reward+=reward
            state=next_state

        if verbose:
            mse=report["MSE"]
            info=f"Testing| Reward: {total_reward}| MSE: {mse}"
            print(info)

        if plots:
            self.test_plots()
            

    def test_plots(self):
        time_steps=list(range(1,self.env.steps))

        data=self.env.final_data[["Xt","Yt"]].iloc[:-1]
        data["Portfolio_Value"]=self.env.pvalue
        data.index=time_steps
        
        data.plot(figsize=(10,6),style=["b","g","r"])
        plt.xlabel("Time steps")
        plt.ylabel("Price")
        plt.legend()
        plt.title("Testing| Time VS (Risk Asset VS Risk Free Asset VS Portfolio) Price")
        plt.show()
        
        plt.plot(time_steps,self.env.pl,lw=1.0,c="b")
        plt.xlabel("Time Steps")
        plt.ylabel("Profit-Loss")
        plt.xlim(0,self.env.steps+2)
        plt.ylim(min(self.env.pl),max(self.env.pl))
        plt.title("Testing| Time Steps VS Profit-Loss")
        plt.show()
        
        plt.plot(time_steps,self.env.pl_percent,lw=1.0,c="g")
        plt.xlabel("Time Steps")
        plt.ylabel("Profit-Loss%")
        plt.xlim(0,self.env.steps+2)
        plt.ylim(min(self.env.pl_percent),max(self.env.pl_percent))
        plt.title("Testing| Time Steps VS Profit-Loss%")
        plt.show
        
        plt.plot(time_steps,self.env.xt_values,lw=1.0,c='c')
        plt.xlabel("Time Steps")
        plt.ylabel("Weight")
        plt.xlim(0,self.env.steps+2)
        plt.ylim(0,1)
        plt.title("Testing| Time Steps VS Risk Asset Weight in Portfolio")
        plt.show()

        