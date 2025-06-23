import os
import random
import numpy as np
import pandas as pd
from pylab import plt,mpl
from collections import deque
from sklearn.metrics import mean_squared_error,mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,LSTM,Concatenate
from tensorflow.keras.optimizers import Adam


class Agent:
    def __init__(
        self,
        env,
        actor_model_config,
        critic_model_config,
        actor_optimizer_config,
        critic_optimizer_config,
        actor_model_loss,
        critic_model_loss,
        gamma,
        exploration_episodes,
        batch_size,
        buffer_size,
        epsilon,
        epsilon_min,
        epsilon_decay
    ):
        
        self.env=env
        
        self.actor_model_config=actor_model_config
        self.critic_model_config=critic_model_config
        self.actor_optimizer_config=actor_optimizer_config
        self.critic_optimizer_config=critic_optimizer_config
        self.actor_model_loss=actor_model_loss
        self.critic_model_loss=critic_model_loss

        self.exploration_episodes=exploration_episodes
        self.gamma=gamma
        self.batch_size=batch_size
        self.buffer_size=buffer_size
        self.epsilon=epsilon
        self.epsilon_min=epsilon_min
        self.epsilon_decay=epsilon_decay

        self.memory=deque(maxlen=self.buffer_size)
        self.model_dir=r"/home/spiralmonster/Projects/ReinforcementLearningForFinance/OptimalExecution/Models"

        self.actor_model=self.create_model(
            model_config=self.actor_model_config,
            optimizer_config=self.actor_optimizer_config,
            loss=self.actor_model_loss
        )
        
        self.critic_model=self.create_model(
            model_config=self.critic_model_config,
            optimizer_config=self.critic_optimizer_config,
            loss=self.critic_model_loss
        )
        

    def create_model(self,model_config,optimizer_config,loss):
        inp1=Input(shape=(self.env.trading_steps+1,1),dtype=tf.float32)
        inp2=Input(shape=(5,),dtype=tf.float32)

        x1=inp1
        x2=inp2

        model1_config=model_config["Model_1"]
        model2_config=model_config["Model_2"]
        final_model_config=config["Final_Model"]

        for config in model1_config:
            x1=LSTM(
                units=config["units"],
                activation=config["activation"],
                kernel_initializer=config["kernel_initializer"],
                kernel_regularizer=config["kernel_regularizer"],
                return_sequences=config["return_sequences"]
            )(x1)
            

        for config in model2_config:
            x2=Dense(
                units=config["units"],
                activation=config["activation"],
                kernel_initializer=config["kernel_initializer"],
                kernel_regularizer=config["kernel_regularizer"]
            )(x2)

        x=Concatenate(axis=-1)[(x1,x2)]

        for config in final_model_config:
            x=Dense(
                units=config["units"],
                activation=config["activation"],
                kernel_initializer=config["kernel_initializer"],
                kernel_regularizer=config["kernel_regularizer"]
            )(x)
            

        model=Model(inputs=[inp1,inp2],outputs=x)

        optimizer=Adam(
            learning_rate=optimizer_config["learning_rate"],
            beta_1=optimizer_config["beta_1"],
            beta_2=optimizer_config["beta_2"]
        )
        
        model.compile(
            optimizer=optimizer,
            loss=loss
        )

        return model

    def get_model_architecture(self,model):
        model.summary()


    def prepare_model_input(self,state):
        model_inp=[]

        def prepare(x):
            x=np.array(x)
            x=np.expand_dims(x,axis=0)
            return x

        for element in state:
            model_inp.append(prepare(element))

        return model_inp
        

    def act(self,state):
        if random.random()<self.epsilon() or self.env.training_episode<self.exploration_episodes:
            action=min(self.env.random_xt[self.env.index+1],self.env.remaining_share)

        else:
            model_inp=self.prepare_model_input(state)
            action=self.actor_model.predict(model_inp,verbose=False)[0][0]

        return action

        
    def step(self,action):
        state=self.get_state()
        
        model_inp=self.prepare_model_input(state)
        xt_learned=self.actor_model.predict(model_inp,verbose=False)[0][0]
        self.env.xt_learned_strategy_per_step.append(xt_learned)

        pred_state_value=self.critic_model.predict(model_inp,verbose=False)[0][0]
        self.env.predicted_state_value.append(pred_state_value)
        next_state_model_inp=self.prepare_model_input(state)
        real_state_value=reward+self.gamma*self.critic_model.predict(next_state_model_inp,verbose=False)[0][0]
        self.env.real_state_value.append(real_state_value)
    
        self.env.index+=1
        send_report=False
        
        if action>self.env.remaining_share:
            action_penalty=(action-self.env.remaining_share)**2
            self.env.xt[self.env.index]=self.env.remaining_share
            self.env.remaining_share=0.0
            
        else:
            action_penalty=0.0
            self.env.xt[self.env.index]=action
            self.env.remaining_share-=action

        self.env.permanent_impact=self.env.cal_permanent_impact()-self.env.permanent_impact
        self.env.temporary_impact=self.env.cal_temporary_impact()-self.env.temporary_impact
        self.env.execution_risk=self.env.cal_execution_risk()-self.env.execution_risk()
        self.env.total_execution_cost=self.env.permanent_impact+self.env.temporary_impact+self.env.execution_risk

        next_state=self.env.get_state()

        if self.env.index<self.env.trading_steps:
            if self.remaining_share<0.0001:
                done=True
                send_report=True
            else:
                done=False
                
        elif self.env.index==self.env.trading_steps:
            done=True
            send_report=True
            pen=self.remaining_share*10

        reward=-(pen+action_penalty+self.env.total_execution_cost)

        self.env.reward_per_step.append(reward)
        self.env.permanent_impact_per_step.append(self.env.permanent_impact)
        self.env.temporary_impact_per_step.append(self.env.temporary_impact)
        self.env.execution_risk_per_step.append(self.env.execution_risk)
        self.env.total_execution_cost_per_step.append(self.env.total_execution_cost)

        if send_report:
            xt_optimal=self.env.xt_optimal[1:]
            
            report={
                "Total Reward":sum(self.env.reward_per_step),
                "Average Reward":sum(self.env.reward_per_step)/self.env.trading_steps,
                "Average Permanent Impact":sum(self.env.permanent_impact_per_step)/self.env.trading_steps,
                "Average Temporary Impact":sum(self.env.temporary_impact_per_step)/self.env.trading_steps,
                "Average Execution Risk":sum(self.env.execution_risk_per_step)/self.env.trading_steps,
                "Average Total Execution Cost":sum(self.env.total_execution_cost)/self.env.trading_steps,
                "Mean Absolute Error between Optimal Execution Strategy and Learned Strategy":mean_absolute_error(
                    xt_optimal,
                    self.env.xt_learned_strategy_per_step
                ),
                "Mean Squared Error between Real and Predicted Value of State":mean_squared_error(
                    self.env.real_state_value,
                    self.env.predicted_state_value
                )
            }
            
        else:
            report={}

        return next_state,reward,done,report


    def replay(self):
        data_X1=[]
        data_X2=[]
        Y_critic=[]
        Y_actor=[]
        actor_sample_weight=[]
        
        batch_data=random.sample(self.memory,self.batch_size)
        for (state,action,next_state,reward,done) in batch_data:
            if not done:
                next_state_model_inp=self.prepare_model_input(next_state)
                next_value=self.critic_model.predict(next_state_model,verbose=False)[0][0]
                critic_target=reward+self.gamma*next_value
                Y_critic.append(critic_target)

                curr_model_inp=self.prepare_model_input(state)
                curr_value=self.critic_model.predict(curr_model_inp,verbose=False)[0][0]
                advantage=critic_target-curr_value

                Y_actor.append(action)
                actor_sample_weight.append(advantage)

                data_X1.append(state[0])
                data_X2.append(state[1])

        data_X1=np.array(data_X1)
        data_X2=np.array(data_X2)
        Y_critic=np.array(Y_critic)
        Y_actor=np.array(Y_actor)
        actor_sample_weight=np.array(actor_sample_weight)
        batch_size=len(Y_critic)

        self.critic_model.fit([data_X1,data_X2],Y_critic,epochs=1,batch_size=batch_size,verbose=False)
        self.actor_model.fit([data_X1,data_X2],Y_actor,sample_weight=actor_sample_weight,batch_size=batch_size,verbose=False)

        if self.epsilon>self.epsilon_min:
            self.epsilon*=self.epsilon_decay
                

    
    def train_agent(self,episodes,training_version,verbose=True):
        self.trewards=list()
        self.avg_reward_per_episode=list()
        self.avg_perm_impact_per_episode=list()
        self.avg_temp_impact_per_episode=list()
        self.avg_exec_risk_per_episode=list()
        self.avg_total_exec_cost_per_episode=list()
        self.mae_btw_strategy=list()
        self.mse_btw_state_value=list()

        self.reward_per_step_per_episode=list()
        self.perm_impact_per_step_per_episode=list()
        self.temp_impact_per_step_per_episode=list()
        self.exec_risk_per_step_per_episode=list()
        self.total_exec_cost_per_step_per_episode=list()
        self.learned_strategy_per_step_per_episode=list()
        
        for ep in range(1,episodes+1):
            state,done=self.env.reset()
            while not done:
                action=self.act(state)
                next_state,reward,done,report=self.step(action)
                
                self.memory.append(
                    [state,action,next_state,reward,done]
                )
                state=next_state

                
            self.trewards.append(report["Total Reward"])
            self.avg_reward_per_episode.append(report["Average Reward"])
            self.avg_perm_impact_per_episode.append(report["Average Permanent Impact"])
            self.avg_temp_impact_per_episode.append(report["Average Temporary Impact"])
            self.avg_exec_risk_per_episode.append(report["Average Execution Risk"])
            self.avg_total_exec_cost_per_episode.append(report["Average Total Execution Cost"])
            self.mae_btw_strategy.append(report[
            "Mean Absolute Error between Optimal Execution Strategy and Learned Strategy"       
            ])
            self.mse_btw_state_value.append(report[
            "Mean Squared Error between Real and Predicted Value of State"                      
            ])

            
            self.reward_per_step_per_episode.append(self.env.reward_per_step)
            self.perm_impact_per_step_per_episode.append(self.env.permanent_impact_per_step)
            self.temp_impact_per_step_per_episode.append(self.env.temporary_impact_per_step)
            self.exec_risk_per_step_per_episode.append(self.env.execution_risk_per_step)
            self.total_exec_cost_per_step_per_episode.append(self.env.total_execution_cost)
            self.learned_strategy_per_step_per_episode.append(self.env.xt_learned_strategy_per_step)

            
            if verbose:
                if (ep%50)==0:
                    info=f"Episode: {ep}/{episodes}| Epsilon: {self.epsilon}|"
                    for key,value in report.items():
                        info+=f" {key}: {value}|"

                    print(info)
                    
            if len(self.memory)>self.batch_size:
                self.replay()

            if ep==episodes:
                actor_model_name=f"actor_model_version_{training_version}.keras"
                critic_model_name=f"critic_model_version_{training_version}.keras"
                
                actor_model_path=os.path.join(self.model_dir,actor_model_name)
                critic_model_path=os.path.join(self.model_dir,critic_model_name)

                self.actor_model.save(actor_model_path)
                self.critic_model.save(critic_model_path)

                print(f"Actor Model saved at: { actor_model_path}")
                print(f"Critic Model saved at: {critic_model_path}")
                
                
    def sample_episodes(self,num_plots):
        return random.sample(range(self.exploration_episodes+1,self.env.training_episode))


    def training_plots(self,num_plots=5):
        time_step=list(range(1,self.env.trading_steps))

        sample_ep=self.sample_episodes(num_plots)
        for ep in sample_ep:
            perm_imp=self.perm_impact_per_step_per_episode[ep]
            data=pd.DataFrame(perm_imp,columns=["Permanent Impact"],index=time_step)
            data["Temporary Impact"]=self.temp_impact_per_step_per_episode[ep]
            data["Execution Risk"]=self.exec_risk_per_step_per_episode[ep]

            data.plot(figsize=(10,6),style=["b","g","r"])
            plt.xlabel("Trading Steps")
            plt.title(f"Training Episode: {ep}| Trading Steps VS Trading Impacts due to Trading Execution")
            plt.legend()
            plt.show()
            

        sample_ep=self.sample_episodes(num_plots)
        for ep in sample_ep:
            total_exec_cost=self.total_exec_cost_per_step_per_episode[ep]
            data=pd.DataFrame(total_exec_cost,columns=["Total Execution Cost"],index=time_step)

            data.plot(figsize=(10,6),style=["c"])
            plt.xlabel("Trading Steps")
            plt.title(f"Training Episode: {ep}| Trading Steps VS Total Execution Cost")
            plt.legend()
            plt.show()
            

        sample_ep=self.sample_episodes(num_plots)
        optimal_strategy=self.env.xt_optimal[1:]
        for ep in sample_ep:
            data=pd.DataFrame(optimal_strategy,columns=["Optimal Strategy"],index=time_step)
            learned_strategy=self.learned_strategy_per_step_per_episode[ep]
            data["Learned Strategy"]=learned_strategy

            data.plot(figsize=(10,6),style=["b","r"])
            plt.xlabel("Trading Steps")
            plt.ylabel("Shares Traded")
            plt.title(f"Training Episode: {ep}| Trading Strategies")
            plt.legend()
            plt.show()
            

        sample_ep=self.sample_episodes(num_plots)
        optimal_strategy=np.array(self.env.xt_optimal[1:]).cumsum()[::-1]
        for ep in sample_ep:
            data=pd.DataFrame(optimal_strategy,columns=["Optimal Strategy"],index=time_step)
            learned_strategy=np.array(self.learned_strategy_per_step_per_episode[ep]).cumsum()[::-1]
            data["Learned Strategy"]=learned_strategy

            data.plot(figsize=(10,6),style=["g","c"])
            plt.xlabel("Trading Steps")
            plt.ylabel("Shares")
            plt.title(f"Training Episode: {ep}| Trading Strategies| Trading Steps Vs Share at Trading Step")
            plt.legend()
            plt.show()


        sample_ep=self.sample_episodes(num_plots)
        for ep in sample_ep:
            reward_data=self.reward_per_step_per_episode[ep]
            plt.plot(time_step,reward_data,lw=1.0,c="b")
            plt.xlabel("Trading Steps")
            plt.ylabel("Reward")
            plt.title(f"Training Episode: {ep}| Trading Steps VS Reward received")
            plt.show()
            

    def episode_plots(self):
        episodes=list(range(1,self.env.training_episode+1))

        plt.plot(episodes,self.avg_perm_impact_per_episode,lw=1.0,c="b",label="Average Permanent Impact")
        plt.plot(episodes,self.avg_temp_impact_per_episode,lw=1.0,c="g",label="Average Temporary Impact")
        plt.plot(episodes,self.avg_exec_risk_per_episode,lw=1.0,c="r",label="Average Execution Risk")
        plt.xlabel("Episode")
        plt.title("Episode VS Average Trading Impacts Per Episode")
        plt.legend()
        plt.show()

        plt.plot(episodes,self.avg_total_exec_cost_per_episode,lw=1.0,c="c")
        plt.xlabel("Episode")
        plt.title("Episode VS Average Total Execution Cost Per Episode")
        plt.show()

        plt.plot(episodes,self.trewards,lw=1.0,c="b")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Episode VS Total Reward Per Episode")
        plt.show()

        plt.plot(episodes,self.avg_reward_per_episode,lw=1.0,c="g")
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.title("Episode VS Average Reward Per Episode")
        plt.show()

        plt.plot(episodes,self.mae_btw_strategy,lw=1.0,c="c")
        plt.xlabel("Episode")
        plt.ylabel("Mean Absolute Error")
        plt.title("Episode VS MAE between Optimal and Learned Strategy")
        plt.show()

        plt.plot(episodes,self.mse_btw_state_value,lw=1.0,c="r")
        plt.xlabel("Episode")
        plt.ylabel("Mean Squared Error")
        plt.title("Episode VS MSE between Real and Predicted State Value")
        plt.show()


    def test_agent(self,verbose=True,plots=True):
        state,done=self.env.reset()
        while not done:
            model_inp=self.prepare_model_input(state)
            action=self.actor_model.predict(model_inp,verbose=False)[0][0]

            next_state,reward,done,report=self.step(action)
            state=next_state

        if verbose:
            info="Testing| "
            for key,value in report.items():
                info+=f" {key}: {value}|"

            print(info)

        if plots:
            self.test_plots()

    def test_plots(self):
        time_step=list(range(1,self.env.trading_steps))

        perm_imp= self.env.permanent_impact_per_step
        data=pd.DataFrame(perm_imp,columns=["Permanent Impact"],index=time_step)
        data["Temporary Impact"]=self.env.temporary_impact_per_step
        data["Execution Risk"]=self.env.execution_risk_per_step
        data.plot(figsize=(10,6),style=["b","g","r"])
        plt.xlabel("Trading Steps")
        plt.title(f"Testing| Trading Steps VS Trading Impacts due to Trading Execution")
        plt.legend()
        plt.show()
        
        total_exec_cost=self.env.total_execution_cost_per_step
        data=pd.DataFrame(total_exec_cost,columns=["Total Execution Cost"],index=time_step)
        data.plot(figsize=(10,6),style=["c"])
        plt.xlabel("Trading Steps")
        plt.title(f"Testing| Trading Steps VS Total Execution Cost")
        plt.legend()
        plt.show()

        
        optimal_strategy=self.env.xt_optimal[1:]
        data=pd.DataFrame(optimal_strategy,columns=["Optimal Strategy"],index=time_step)
        learned_strategy=self.env.xt_learned_strategy_per_step
        data["Learned Strategy"]=learned_strategy
        data.plot(figsize=(10,6),style=["b","r"])
        plt.xlabel("Trading Steps")
        plt.ylabel("Shares Traded")
        plt.title(f"Testing| Trading Strategies")
        plt.legend()
        plt.show()

        