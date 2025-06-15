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
        input=Input(shape=(self.env.n_features,),dtype=tf.float64)
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
        if random.random()<self.epsilon:
            action=self.env.action_space.sample()
        else:
            inp=np.expand_dims(state,axis=0)
            pred=self.model.predict(inp,batch_size=1,verbose=False)[0]
            action=np.argmax(pred)
            
            
        return action

    def step(self,action,testing=False):
        real_action=self.env.final_data['position'].iloc[self.env.index]

        self.env.real_action.append(real_action)

        if action==real_action:
            reward=1
            
        else:
            reward=-1

        self.env.performance*=np.exp(self.env.final_data['returns'].iloc[self.env.index]*np.where(action>0,1,-1))
        
        if testing:
            send_acc_perf_report=True
        else:
            send_acc_perf_report=False
            

        if self.env.index==len(self.env.final_data)-2:
            done=True
            send_acc_perf_report=True

        elif self.env.index>15 and not testing:
            send_acc_perf_report=True
            acc_score=accuracy_score(self.env.real_action,self.env.pred_action)
            
            if acc_score<self.min_pred_accuracy or self.env.performance<self.min_performance:
                done=True
                
            else:
                done=False
        else:
            done=False

        self.env.index+=1
        next_state=self.env.get_state()

        if done:
            if send_acc_perf_report:
                report={}
                report["Prediction_accuracy"]=accuracy_score(self.env.real_action,self.env.pred_action)
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
        for (state,action,next_state,reward,done) in batch:
            if not done:
                qvalue_next=np.amax(self.model.predict(np.expand_dims(next_state,axis=0),verbose=False)[0])
                reward+=self.gamma*qvalue_next
                target=self.model.predict(np.expand_dims(state,axis=0),verbose=False)
                target[0,action]=reward
                data_X.append(state)
                data_Y.append(target[0])

        batch_size=len(data_X)
        data_X=np.array(data_X)
        data_Y=np.array(data_Y)
        self.model.fit(data_X,data_Y,batch_size=batch_size,epochs=steps_per_episode,verbose=False)

        if self.epsilon>self.epsilon_min:
            self.epsilon*=self.epsilon_decay
        

    
    
    def train_agent(self,episodes,steps_per_episode,training_version,verbose=True):
        self.train_episodes=episodes
        self.pred_per_episode=list()
        self.steps_per_episode=list()
        for ep in range(1,episodes+1):
            state,done=self.env.reset()
            total_reward=0
            step=0
            while not done:
                action=self.act(state)
                pred_action=np.argmax(self.model.predict(np.expand_dims(state,axis=0),verbose=False)[0])
                self.env.pred_action.append(pred_action)
                next_state,reward,done,report=self.step(action)
                total_reward+=reward
                self.memory.append(
                    [state,action,next_state,reward,done]
                )
                state=next_state
                step+=1

            self.trewards.append(total_reward)
            self.steps_per_episode.append(step)
            self.max_reward=max(self.max_reward,total_reward)
            
            if verbose:
                if ep%10==0:
                    info=f"Training Episode {ep}/{episodes}| Reward: {total_reward}|"
                    info+=f" Epsilon: {self.epsilon}| Steps: {step}|"
                    if report:
                        info+=f" Prediction Accuracy: {report['Prediction_accuracy']}|"
                        info+=f" Performance: {report['Performance']}"
                    else:
                        info+=f" Prediction Accuracy: NaN| Performance: NaN"
                        
                    print(info)

            if report:
                self.acc_per_episode.append(report["Prediction_accuracy"])
                self.perf_per_episode.append(report["Performance"])
            else:
                self.acc_per_episode.append(0)
                self.perf_per_episode.append(1)
                
            self.pred_per_episode.append(self.env.pred_action)
            
            if len(self.memory)>self.batch_size:
                self.replay(steps_per_episode)

            if ep==episodes:
                model_name1=f"Model_version_{training_version}.keras"
                model_name2=f"Model_version_{training_version}.h5"
                
                model_path1=os.path.join(self.model_dir_path,model_name1)
                model_path2=os.path.join(self.model_dir_path,model_name2)
                
                
                try:
                    self.model.save(model_path1)
                    self.model.save(model_path2)
                    
                    print(f"Model saved successfully at: {model_path1}")
                    print(f"Model saved successfully at: {model_path2}")
                    
                except Exception as e:
                    print(e)
                
    def test_agent(self,verbose=True,plots=True):
        total_reward=0
        state,done=self.env.reset()
        while not done:
            action=np.argmax(self.model.predict(np.expand_dims(state,axis=0),verbose=False))
            self.env.pred_action.append(action)
            next_state,reward,done,report=self.step(action,testing=True)
            state=next_state
            total_reward+=reward

        if verbose:
            info=f"Reward: {total_reward}| Performance: {report['Performance']}|"
            info+f"Prediction Accuracy: {report['Prediction_accuracy']}"
            print(info)
            
        if plots:
            self.test_plots()

    def test_plots(self):
        pred_pos=[1 if pred>0 else -1 for pred in self.env.pred_action]
        data=pd.DataFrame(self.env.final_data["returns"].iloc[:self.env.index])
        data["Predicted_position"]=pred_pos
        data["strategy"]=data["returns"]*data["Predicted_position"]
        data[["returns","strategy"]].cumsum().apply(np.exp).plot(figsize=(10,6),style=["b","g"])
        plt.legend()
        plt.title(f"{self.env.symbol}| Performance of Strategy")
        plt.show()


        
    def generate_performance_plots(self,num_plots=5):
        sampled_indices=random.sample(range(self.train_episodes),num_plots)
        
        for ind in sampled_indices:
            episode_pred=self.pred_per_episode[ind]
            
            bar=len(episode_pred)
            org_returns=pd.DataFrame(self.env.final_data["returns"].iloc[:bar])

            episode_pred=pd.DataFrame(episode_pred,columns=["Prediction"],index=org_returns.index)

            data=pd.concat([org_returns,episode_pred],axis=1)
            data["Predicted_position"]=np.where(data["Prediction"]>0,1,-1)
            data["strategy"]=data["returns"]*data["Predicted_position"]

            data[["returns","strategy"]].cumsum().apply(np.exp).plot(figsize=(10,6),style=["b","r--"])
            title=f"Strategy Performance| Train Episode:{ind}"
            plt.title(title)
            plt.show()

    def generate_episode_plots(self):
        episodes=list(range(1,self.train_episodes+1))

        plt.plot(episodes,self.steps_per_episode,lw=1.0,c='b')
        plt.xlabel("Episodes")
        plt.ylabel("Number of Steps")
        plt.xlim(0,episodes[-1]+1)
        plt.ylim(min(self.steps_per_episode)-1,max(self.steps_per_episode)+1)
        plt.title("Training Episodes VS Steps per Episode")
        plt.show()

        plt.plot(episodes,self.trewards,lw=1.0,c='b')
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.xlim(0,episodes[-1]+1)
        plt.ylim(min(self.trewards)-1,max(self.trewards)+1)
        plt.title("Training Episodes VS Reward")
        plt.show()

        org_perf_per_episode=[]
        for step in self.steps_per_episode:
            perf=np.exp(self.env.final_data["returns"].iloc[:step].cumsum().iloc[-1])
            org_perf_per_episode.append(perf)

        plt.plot(episodes,org_perf_per_episode,lw=1.0,linestyle='-',color="blue",label="Original Returns")
        plt.plot(episodes,self.perf_per_episode,lw=1.0,linestyle="-",color="green",label="Strategy Returns")
        plt.xlabel("Episodes")
        plt.ylabel("Performance")
        plt.xlim(0,episodes[-1]+1)
        plt.ylim(min(self.perf_per_episode)-1,max(self.perf_per_episode)+1)
        plt.title("Training Episodes VS Performance")
        plt.legend()
        plt.show()

        plt.plot(episodes,self.acc_per_episode,lw=1.0,color="blue")
        plt.xlabel("Episodes")
        plt.ylabel("Accuracy")
        plt.xlim(0,episodes[-1]+1)
        plt.ylim(0,1)
        plt.title("Training Episodes VS Prediction Accuracy")
        plt.show()
        
