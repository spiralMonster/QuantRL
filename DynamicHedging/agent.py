import os
import random
import numpy as np
import pandas as pd
from collections import deque
from scipy.optimize import minimize
from pylab import plt,mpl
from sklearn.metrics import mean_squared_error,mean_absolute_error
import tensorflow as tf
from tensorflow.keras.layers import LSTM,Dense,Input,Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from environment import Environment

plt.style.use("seaborn-v0_8")
mpl.rcParams["figure.dpi"]=300
mpl.rcParams["savefig.dpi"]=300
mpl.rcParams["font.family"]="serif"



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

        self.model_dir=r"/home/spiralmonster/Projects/ReinforcementLearningForFinance/DynamicHedging/Models" 

        self.create_model()

    def create_model(self):
        inp1=Input(shape=(self.env.number_lags+1,3),dtype=tf.float64)
        inp2=Input(shape=(self.env.number_lags+1,2),dtype=tf.float64)
        inp3=Input(shape=(self.env.number_lags+1,2),dtype=tf.float64)
        inp4=Input(shape=(self.env.number_lags+1,2),dtype=tf.float64)
        inp5=Input(shape=(8,),dtype=tf.float64)

        model1_config=self.model_config["Model_1"]
        model2_config=self.model_config["Model_2"]
        model3_config=self.model_config["Model_3"]
        model4_config=self.model_config["Model_4"]
        model5_config=self.model_config["Model_5"]

        final_model_config=self.model_config["Final_Model"]

        x1=inp1
        x2=inp2
        x3=inp3
        x4=inp4
        x5=inp5

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
            x3=LSTM(
                units=config["units"],
                activation=config["activation"],
                kernel_initializer=config["kernel_initializer"],
                kernel_regularizer=config["kernel_regularizer"],
                return_sequences=config["return_sequences"]
            )(x3)

        for config in model4_config:
            x4=LSTM(
                units=config["units"],
                activation=config["activation"],
                kernel_initializer=config["kernel_initializer"],
                kernel_regularizer=config["kernel_regularizer"],
                return_sequences=config["return_sequences"]
            )(x4)

        for config in model5_config:
            x5=Dense(
                units=config["units"],
                activation=config["activation"],
                kernel_initializer=config["kernel_initializer"],
                kernel_regularizer=config["kernel_regularizer"]
            )(x5)

        x=Concatenate(axis=-1)([x1,x2,x3,x4,x5])

        for config in final_model_config:
            x=Dense(
                units=config["units"],
                activation=config["activation"],
                kernel_initializer=config["kernel_initializer"],
                kernel_regularizer=config["kernel_regularizer"]
            )(x)

        model=Model(inputs=[inp1,inp2,inp3,inp4,inp5],outputs=x)

        optimizer=Adam(
            learning_rate=self.optimizer_config["learning_rate"],
            beta_1=self.optimizer_config["beta_1"],
            beta_2=self.optimizer_config["beta_2"]
        )

        model.compile(
            optimizer=optimizer,
            loss="mse"
        )

        self.model=model

    def get_model_architecture(self):
        self.model.summary()


    def prepare_model_input(self,state):
        inp=[]

        def prepare(state):
            return np.expand_dims(np.array(state),axis=0)

        for element in state:
            inp.append(prepare(element))

        return inp

    def compute_qvalue(self,next_state,next_state_optimal_action,reward):
        next_state[4][6]=next_state_optimal_action[0]
        next_state[4][7]=next_state_optimal_action[1]

        model_inp=self.prepare_model_input(next_state)
        value=self.model.predict(model_inp,verbose=False)[0][0]

        qvalue=reward+self.gamma*value
        return qvalue
        
    
    
    def optimal_action(self,state):
        bnds=[(0,1)]
        
        def f(state,x):
            state[5][6]=x[0]
            state[5][7]=(state[5][2]-x[0]*state[5][0])/state[5][1]

            inp=self.prepare_model_input(state)
            return self.model.predict(input,verbose=False)[0][0]

        action=minimize(lambda x: -f(state,x),0.5,bounds=bnds,method="Powell")['x'][0]
        return action
            
    
    def act(self,state):
        if random.random()<self.epsilon:
            action=self.env.action_space.sample()

        else:
            action=self.optimal_action(state)

        return action


    def step(self,action):
        next_Ct=self.env.final_data["Ct"].iloc[self.env.index+1]
        next_Xt=self.env.final_data["Xt"].iloc[self.env.index+1]
        next_Yt=self.env.final_data["Yt"].iloc[self.env.index+1]

        send_report=False

        if self.env.index==0:
            curr_Ct=self.env.final_data["Ct"].iloc[self.env.index]
            curr_Xt=self.env.final_data["Xt"].iloc[self.env.index]
            curr_Yt=self.env.final_data["Yt"].iloc[self.env.index]

            self.env.stock=action
            self.env.bond=(curr_Ct-self.env.stock*curr_Xt)/curr_Yt

            reward=0

        else:
            phi_value=self.env.stock*next_Xt+self.env.bond*next_Yt
            pl=phi_value-next_Ct
            pl_per=pl/next_Ct
            reward=-(pl**2)

            self.env.stock=action
            self.env.bond=(next_Ct-self.env.stock*next_Xt)/next_Yt

            self.env.phi_value_per_step.append(phi_value)
            self.env.reward_per_step.append(reward)
            self.env.pl_per_step.append(pl)
            self.env.pl_percent_per_step.append(pl_per)
            

        state=self.env.get_state()
        self.env.index+=1
        next_state=self.env.get_state()
        
        self.env.model_delta_per_step.append(self.env.stock)
        self.env.bond_weight_per_step.append(self.env.bond)
        
        if self.env.index!=1:
            delta=self.env.final_data["delta"].iloc[self.env.index]
            bond_weight=(next_state[4][2]-delta*next_state[4][0])/next_state[4][1]
            opt_action=(delta,bond_weight)
            
            real_qvalue=self.compute_qvalue(next_state,opt_action,reward)
            self.env.real_qvalue_per_step.append(real_qvalue)

            model_inp=self.prepare_model_input(state)
            pred_qvalue=self.model.predict(model_inp,verbose=False)[0][0]
            self.env.predicted_qvalue_per_step.append(pred_qvalue)
            

       

        if self.env.index==self.env.steps-1:
            done=True
            send_report=True

        else:
            done=False

        if send_report:
            report={
                "Total Reward":sum(self.env_reward_per_step),
                "Average Reward":sum(self.env.reward_per_step)/self.env.steps,
                "Average Profit-Loss":sum(self.env.pl_per_step)/self.env.steps,
                "Average Profit-Loss%":sum(self.env.pl_percent_per_step)/self.env.steps,
                
                "MAE between Option Value and Replication Portfolio":mean_absolute_error(
                    self.env.phi_value_per_step,
                    list(self.env.final_data["Ct"].iloc[1:])
                ),
                
                "MSE between theoretical and predicted delta":mean_squared_error(
                    list(self.env.final_data["delta"].iloc[1:]),
                    self.env.model_predicted_delta
                ),

                "MSE between real and predicted Qvalue":mean_squared_error(
                    self.env.real_qvalue_per_step,
                    self.env.predicted_qvalue_per_step
                ),
                
            }
            
        else:
            report={}
            
        return next_state,reward,done,report
        

    def replay(self,steps_per_episode):
        data_X1=[]
        data_X2=[]
        data_X3=[]
        data_X4=[]
        data_X5=[]
        data_Y=[]

        batch_data=random.sample(self.memory,self.batch_size)
        
        for(state,action,next_state,reward,done) in batch_data:
            
            if done:
                delta=self.optimal_action(next_state)
                bond_weight=(next_state[4][2]-delta*next_state[4][0])/next_state[4][1]
                next_state_opt_action=(delta,bond_weight)
                
                target=self.compute_qvalue(next_state,next_state_opt_action,reward)
    
                data_Y.append(target)
    
                data_X1.append(state[0])
                data_X2.append(state[1])
                data_X3.append(state[2])
                data_X4.appenda(state[3])
                data_X5.append(state[4])
                
        batch_size=len(data_Y)
        
        data_X1=np.array(data_X1)
        data_X2=np.array(data_X2)
        data_X3=np.array(data_X3)
        data_X4=np.array(data_X4)
        data_X5=np.array(data_X5)
        data_Y=np.array(data_Y)

        model.fit([data_X1,data_X2,data_X3,data_X4,data_X5],data_Y,batch_size=batch_size,epochs=steps_per_episode,verbose=False)

        if self.epsilon>self.epsilon_min:
            self.epsilon*=self.epsilon_decay
        
    
    
    def train_agent(self,episodes,steps_per_episode,training_version,verbose=True):
        self.training_episodes=episodes
        
        self.trewards=[]
        self.avg_reward=[]
        self.pl_avg_per_episode=[]
        self.pl_percent_avg_per_episode=[]
        self.mse_qvalues=[]
        self.mse_deltas=[]
        self.mae_C_and_phi=[]
        
        self.reward_per_step_per_episode=[]
        self.phi_value_per_step_per_episode=[]
        self.predicted_delta_per_step_per_episode=[]
        self.bond_weight_per_step_per_episode=[]
        
        
        for ep in range(1,episodes+1):
            state,done=self.env.reset()
            while not done:
                action=self.act(state)
                next_state,reward,done,report=self.step(action)

                self.memory.append([
                    state,action,next_state,reward,done
                ])
                
                state=next_state

            self.trewards.append(report["Total Reward"])
            self.avg_reward.append(report["Average Reward"])
            self.pl_avg_per_episode.append(report["Average Profit-Loss"])
            self.pl_percent_avg_per_episode.append(report["Average Profit-Loss%"])
            self.mse_qvalues.append(report["MSE between real and predicted Qvalue"])
            self.mse_deltas.append(report["MSE between theoretical and predicted delta"])
            self.mae_C_and_phi.append(report["MAE between Option Value and Replication Portfolio"])
            
            self.reward_per_step_per_episode.append(self.env.reward_per_step)
            self.phi_value_per_step_per_episode.append(self.env.phi_value_per_step)
            self.predicted_delta_per_step_per_episode.append(self.env.model_delta_per_step)
            self.bond_weight_per_step_per_episode.append(self.env.bond_weight_per_step)

            if verbose:
                if(env%10)==0:
                    info=f"Episode: {ep}/{self.training_episodes}| Epsilon: {self.epsilon} "
                    for key,value in report.items():
                        info+=f"{key}: {value}| "

                    print(info)

            if len(self.memory)>self.batch_size:
                self.replay(steps_per_episode)

            if ep==self.training_episodes:
                if self.env.env_type=="simulated":
                    model_name=f"simulated_model_version_{training_version}.keras"
                    
                else:
                    model_name=f"model_version_{training_version}.keras"

                model_path=os.path.join(self.model_dir,model_name)
                self.model.save(model_path)
                print(f"Model Saved at: {model_path}")
                
                
    def sample_episodes(self,num_plots):
        sampled_ep=random.sample(range(self.training_episodes),num_plots)
        return sampled_ep
    
    def training_plots(self,num_plots=5):
        time_step=list(range(1,self.env.steps))
        
        sampled_ep=self.sample_episodes(num_plots)
        for ep in sample_ep:
            reward_data=self.reward_per_step_per_episode[ep]

            plt.plot(time_step,reward_data,lw=1.0,c="b")
            plt.xlabel("Time Steps")
            plt.ylabel("Reward")
            plt.title(f"Training Episode: {ep+1}| Time VS Reward")
            plt.show()
            

        sampled_ep=self.sample_episodes(num_plots)
        for ep in sampled_ep:
            phi_data=self.phi_value_per_step_per_episode[ep]
            
            data=self.env.final_data["Ct"].iloc[1:]
            data["Phi"]=phi_data
            data.index=time_step

            data.plot(figsize=(10,6),style=["b","r"])
            plt.xlabel("Time Steps")
            plt.ylabel("Price")
            plt.title(f"Training Episode: {ep+1}| Option Value(Ct)| Replicated Portfolio(Phi)")
            plt.legend()
            plt.show()
            

        sampled_ep=self.sample_episodes(num_plots)
        for ep in sampled_ep:
            delta_data=self.predicted_delta_per_step_per_episode[ep]
            
            data=self.env.final_data["delta"].iloc[1:]
            data["pred_delta"]=delta_data
            data.index=time_step

            data.plot(figsize=(10,6),style=["g","c"])
            plt.xlabel("Time Steps")
            plt.ylabel("Weight")
            plt.title(f"Training Episode: {ep+1}| Time VS Delta(Stock Weight in Replication Portfolio)")
            plt.legend()
            plt.show()


        sampled_ep=self.sample_episodes(num_plots)
        for ep in sampled_ep:
            bond_data=self.bond_weight_per_step_per_episode[ep]

            data=pd.DataFrame(bond_data,columns=["Bond weight in Repl.Portfolio"],index=time_step)
            data.plot(figsize=(10,6),style=["g"])
            plt.xlabel("Time Steps")
            plt.ylabel("Weight")
            plt.title(f"Training Episode: {ep}| Time Vs Bond Weight in Replication Portfolio")
            plt.show()

    def episode_plots(self):
        episodes=list(range(1,self.training_episodes+1))

        plt.plot(episodes,self.trewards,lw=1.0,c="b")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title("Episode VS Total Reward Per Episode")
        plt.show()

        plt.plot(episodes,self.pl_avg_per_episode,lw=1.0,c="g")
        plt.xlabel("Episodes")
        plt.ylabel("Profit-Loss")
        plt.title("Episode VS Avg. Profit-Loss Between Option Value(Ct) and Repl.Portfolio(Phi)")
        plt.show()

        plt.plot(episodes,self.pl_percent_avg_per_episode,lw=1.0,c="c")
        plt.xlabel("Episodes")
        plt.ylabel("Profit-Loss%")
        plt.title("Episode VS Avg. Profit-Loss% Between Option Value(Ct) and Repl.Portfolio(Phi)")
        plt.show()

        plt.plot(episodes,self.mse_qvalues,lw=1.0,c="b")
        plt.xlabel("Episodes")
        plt.ylabel("Mean Squared Error")
        plt.title("Episode VS MSE between Real and Predicted Qvalues")
        plt.show()

        plt.plot(episodes,self.mse_deltas,lw=1.0,c="g")
        plt.xlabel("Episodes")
        plt.ylabel("Mean Squared Error")
        plt.title("Episode VS MSE between Theoretical and Predicted Delta")
        plt.show()

        plt.plot(episodes,self.mae_C_and_phi,lw=1.0,c="b")
        plt.xlabel("Episodes")
        plt.ylabel("Mean Absolute Error")
        plt.title("Episode VS MAE between Option Value(Ct) and Replicated Portfolio(Phi)")
        plt.show()

    def test_agent(self,verbose=True,plots=True):
        state,done=self.env.reset()
        total_reward=0
        while not done:
            action=self.optimal_action(state)
            next_state,reward,done,report=self.step(action)
            state=next_state

        if verbose:
            info=f"Testing| "
            for key,value in report.items():
                info+=f"{key}: {value}| "

            print(info)

        if plots:
            self.test_plots()
            
    def test_plots():
        time_step=list(range(1,self.env.steps))
                
        reward_data=self.env.reward_per_step
        plt.plot(time_step,reward_data,lw=1.0,c="b")
        plt.xlabel("Time Steps")
        plt.ylabel("Reward")
        plt.title(f"Testing | Time VS Reward")
        plt.show()
            
        
        phi_data=self.env.phi_value_per_step
        data=self.env.final_data["Ct"].iloc[1:]
        data["Phi"]=phi_data
        data.index=time_step
        data.plot(figsize=(10,6),style=["b","r"])
        plt.xlabel("Time Steps")
        plt.ylabel("Price")
        plt.title(f"Testing | Option Value(Ct)| Replicated Portfolio(Phi)")
        plt.legend()
        plt.show()
            
        
        delta_data=self.env.model_delta_per_step
        data=self.env.final_data["delta"].iloc[1:]
        data["pred_delta"]=delta_data
        data.index=time_step
        data.plot(figsize=(10,6),style=["g","c"])
        plt.xlabel("Time Steps")
        plt.ylabel("Weight")
        plt.title(f"Testing | Time VS Delta(Stock Weight in Replication Portfolio)")
        plt.legend()
        plt.show()

        
        bond_data=self.env.bond_weight_per_step
        data=pd.DataFrame(bond_data,columns=["Bond weight in Repl.Portfolio"],index=time_step)
        data.plot(figsize=(10,6),style=["g"])
        plt.xlabel("Time Steps")
        plt.ylabel("Weight")
        plt.title(f"Testing | Time Vs Bond Weight in Replication Portfolio")
        plt.show()
    
  