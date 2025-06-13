import os
import random
import numpy as np
import pandas as pd
from pylab import plt,mpl
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from trading_environment import TradingEnvironment

plt.style.use("seaborn-v0_8")
mpl.rcParams["figure.dpi"]=300
mpl.rcParams["savefig.dpi"]=300
mpl.rcParams["font.family"]="serif"


class TradingAgent:
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
        epsilon_decay,
        min_pred_accuracy,
        min_performance
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
        self.min_pred_accuracy=min_pred_accuracy
        self.min_performance=min_performance
        
        self.episodes=0
        self.trewards=list()
        self.max_reward=-np.inf
        self.acc_per_episode=list()
        self.perf_per_episode=list()
        self.memory=deque(maxlen=self.buffer_size)

        self.create_model()
        self.model_dir_path=r"/home/spiralmonster/Projects/ReinforcementLearningForFinance/AlogrithmicTrading/Models"


    def create_model(self):
        input=Input(shape=(self.env.n_features,),dtype=tf.float32)
        x=input
        
        hidden_layer_config=self.model_config["hidden_layers"]
        for config in hidden_layer_config:
            x=Dense(
                units=config["units"],
                activation=config["activation"],
                kernel_initializer=config["kernel_initializer"],
                kernel_regularizer=config["kernel_regularizer"]
                
            )(x)

        final_layer_config=self.model_config["final_layer"]
        out=Dense(
            units=final_layer_config["units"],
            activation=final_layer_config["activation"],
            kernel_initializer=final_layer_config["kernel_initializer"],
            kernel_regularizer=final_layer_config["kernel_regularizer"]
        )(x)

        model=Model(inputs=input,outputs=out)

        optimizer=Adam(
            learning_rate=self.optimizer_config['learning_rate'],
            beta_1=self.optimizer_config['beta_1'],
            beta_2=self.optimizer_config['beta_2']
        )

        model.compile(
            optimizer=optimizer,
            loss="mse"
        )

        self.model=model

    def get_model_architecture(self):
        self.model.summary()

    def act(self,state):
        if random.random()<self.epsilon():
            action=self.env.action_space().sample()
        else:
            inp=np.astype(state,dtype=np.float32)
            pred=self.model.predict(inp,batch_size=1,verbose=False)[0]
            action=np.amax(pred)
            
        return action

    def step(self,action):
        pred_action=action
        real_action=int(self.env.final_data['position'].iloc(self.env.index))

        self.env.pred_action.append(pred_action)
        self.env.real_action.append(real_action)

        if pred_action==real_action:
            reward=1
            
        else:
            reward=-1

        self.env.performance*=np.exp(self.env.final_data['returns'].iloc[self.env.index]*np.where(action>0,1,-1))
        send_acc_perf_report=False

        if self.env.index==len(self.env.final_data)-2:
            done=True

        elif self.env_index>15:
            send_acc_perf_report=True
            acc_score=accuracy_score(self.env.real_action,self.env_pred_action)
            
            if acc_score<self.min_pred_accuracy or self.performance<self.min_performance:
                done=True
        else:
            done=False

        self.env_index+=1
        next_state=self.env.get_state()

        if done:
            if send_acc_perf_report:
                report={}
                report["Prediction_accuracy"]=accuracy_score(self.env.real_action,self.env_pred_action)
                report["Performance"]=self.env.performance
                
            else:
                report={}
        else:
            report={}

        return next_state,reward,done,report

    def replay(self,steps_per_episode):
        data_X=[]
        data_Y=[]
        batch=random.sample(self.memory,self.batch_size)
        for (state,action,new_state,reward,done) in batch:
            if not done:
                qvalue_next=np.amax(self.model.predict(next_state,batch_size=1,verbose=False)[0])
                reward+=self.gamma*qvalue_next
                target=self.model.predict(state,batch_size=1,verbose=False)
                target[0,action]=reward
                data_X.append(state)
                data_Y.append(target[0])

        batch_size=len(data_X)
        data_X=np.array(data_X)
        data_Y=np.array(data_Y)
        self.model.fit(data_X,data_Y,batch_size=batch_size,epochs=steps_per_episode,verbose=False)
        

    
    
    def train_agent(self,episodes,steps_per_episode,training_version,verbose=True):
        for ep in range(1,episodes+1):
            state,done=self.env.reset()
            total_reward=0
            while not done:
                action=self.act(state)
                next_state,reward,done,report=self.step(action)
                total_reward+=reward
                self.memory.append(
                    [state,action,next_state,reward,done]
                )
                state=next_state

            self.trewards.append(total_reward)
            self.max_reward=max(self.max_reward,total_reward)
            if verbose:
                if ep%10==0:
                    info=f"Training Episode {ep}/{episodes}| Reward: {total_reward}|"
                    info+=f" Prediction Accuracy: {report['Prediction_accuracy']}| Performance: {report['Performance']}"
                    print(info)

            if report:
                self.acc_per_episode.append(report["Prediction_accuracy"])
                self.perm_per_episode.append(report["Performance"])
            else:
                self.acc_per_episode.append(0)
                self.perm_per_episode.append(1)
                
            if len(self.memory)>self.batch_size:
                self.replay(steps_per_episode)

            if ep==episodes:
                model_name=f"Model_version_{training_version}.h5"
                model_path=os.path.join(self.model_dir,model_name)
                
                try:
                    self.model.save(model_path)
                    print(f"Model saved successfully at: {model_path}")
                    
                except Exception as e:
                    print(e)
                
    def test_agent(self,episodes):
        self.test_episodes=episodes
        self.test_acc_per_episode=list()
        self.test_perf_per_episode=list()
        self.test_reward_per_episode=list()
        self.test_predictions_per_episode=list()

        for ep in range(1,episodes+1,verbose=True):
            state,done=self.env.reset()
            total_reward=0
            while not done:
                action=np.amax(self.model.predict(state,batch_size=1,verbose=False)[0])
                next_state,reward,done,report=self.step(action)
                total_reward+=reward
                state=next_state

            self.test_reward_per_episode.append(total_reward)
            self.test_predictions_per_episode.append(self.env.pred_action)
            if report:
                self.test_perf_per_episode.append(report["Performance"])
                self.test_acc_per_episode.append(report["Prediction_accuracy"])
                
            else:
                self.test_perf_per_episode.append(1)
                self.test_acc_per_episode.append(0)

            if verbose:
                if ep%10==0:
                    info=f"Testing Episode {ep}/{episodes}| Reward: {total_reward}|"
                    info+=f" Prediction Accuracy: {report['Prediction_accuracy']}| Performance: {report['Performance']}"
                    print(info)

    def generate_performance_plot(self,num_plots=5):
        sampled_indices=random.sample(range(self.test_episodes),num_plots)
        
        for ind in sampled_indices:
            episode_pred=self.test_predictions_per_episode[ind]
            
            bar=len(episode_pred)
            org_returns=pd.DataFrame(self.env.final_data["returns"].iloc[:bar])

            episode_pred=pd.DataFrame(episode_pred,columns=["Prediction"],index=org_returns.index)

            data=pd.concat([org_returns,episode_pred],axis=1)
            data["Predicted_position"]=np.where(data["Prediction"]>0,1,-1)
            data["strategy"]=data["returns"]*data["Predicted_position"]

            data[["returns","strategy"]].cumsum().apply(np.exp).plot(figsize=(10,6),style=["b","r--"])
            title=f"Strategy Performance| Test Episode:{ind}"
            plt.title(title)
            plt.show()

    def generate_episode_plots(self):
        episodes=list(range(1,self.test_episodes+1))
        number_steps_per_episode=[len(pred) for pred in self.test_predictions_per_episode]

        plt.plot(episoder,number_steps_per_episode,lw=1.0,c='b')
        plt.xlabel("Episodes")
        plt.ylabel("Number of Steps")
        plt.xlim(0,episodes[-1]+1)
        plt.ylim(min(number_steps_per_episode)-1,max(number_steps_per_episode)+1)
        plt.title("Testing Episodes VS Steps per Episode")
        plt.show()

        plt.plot(episodes,self.test_reward_per_episode,lw=1.0,c='b')
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.xlim(0,episodes[-1]+1)
        plt.ylim(min(self.test_reward_per_episode)-1,max(self.test_reward_per_episode)+1)
        plt.title("Testing Episodes VS Reward")
        plt.show()

        org_perf_per_episode=[]
        for step in number_steps_per_episode:
            perf=np.exp(self.env.final_data["returns"].iloc[:step].cumsum().iloc[-1])
            org_perf_per_episode.append(perf)

        plt.plot(episodes,org_perf_per_epsiode,lw=1.0,linestyle='-',color="blue",label="Original Returns")
        plt.plot(episodes,self.test_perf_per_episode,lw=1.0,linestyle="-",color="green",label="Strategy Returns")
        plt.xlabel("Episodes")
        plt.ylabel("Performance")
        plt.xlim(0,episodes[-1]+1)
        plt.ylim(min(self.test_perf_per_episode)-1,max(self.test_perf_per_episode)+1)
        plt.title("Test Episodes VS Performance")
        plt.legend()
        plt.show()

        plt.plot(episodes,self.test_acc_per_episode,lw=1.0,color="blue")
        plt.xlabel("Episodes")
        plt.ylabel("Accuracy")
        plt.xlim(0,episodes[-1]+1)
        plt.ylim(0,1)
        plt.title("Test Episodes VS Prediction Accuracy")
        plt.show()
        



    
