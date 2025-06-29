import os
import random
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,LSTM
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from collections import deque
from pylab import plt,mpl
from positional_embedding_layer import PositionalEmbeddingLayer

plt.style.use("seaborn-v0_8")
mpl.rcParams["figure.dpi"]=300
mpl.rcParams["savefig.dpi"]=300
mpl.rcParams["font.family"]="serif"

model_dir_path=r"/home/spiralmonster/Projects/ReinforcementLearningForFinance/StockPicker/Models"


class Actor_Critic_Agent:
    def __init__(
        self,
        env,
        actor_model_config,
        critic_model_config,
        actor_optimizer_config,
        critic_optimizer_config,
        actor_model_loss,
        critic_model_loss,
        batch_size,
        buffer_size,
        exploration_episode,
        gamma,
        epsilon,
        epsilon_min,
        epsilon_decay,
        model_trained=False
    ):

        self.env=env
        self.actor_model_config=actor_model_config
        self.critic_model_config=critic_model_config
        self.actor_optimizer_config=actor_optimizer_config
        self.critic_optimizer_config=critic_optimizer_config
        self.actor_model_loss=actor_model_loss
        self.critic_model_loss=critic_model_loss

        self.batch_size=batch_size
        self.exploration_episode=exploration_episode
        self.buffer_size=buffer_size
        self.memory=deque(maxlen=self.buffer_size)

        self.gamma=gamma
        self.epsilon=epsilon
        self.epsilon_min=epsilon_min
        self.epsilon_decay=epsilon_decay
        self.model_trained=model_trained
        self.model_dir_path=model_dir_path
        

        if not self.model_trained:
            self.actor_model=self.create_model(
                model_config=self.actor_model_config,
                optimizer_config=self.actor_optimizer_config,
                model_loss=self.actor_model_loss
            )
    
            self.critic_model=self.create_model(
                model_config=self.critic_model_config,
                optimizer_config=self.critic_optimizer_config,
                model_loss=self.critic_model_loss
            )

        


    def create_model(self,model_config,optimizer_config,model_loss):
        lstm_config=model_config["LSTM"]
        dense_config=model_config["Dense"]
        final_model_config=model_config["Final_Model"]

        model_inputs=[]
        model_outputs=[]
        
        lstm_model_names=list(lstm_config["Models"].keys())
        lstm_model_config=lstm_config["Models"]
        lstm_input_config=lstm_config["Input_Shape"]
        
        for model_name in lstm_model_names:
            inp_shape=lstm_input_config[model_name]
            inp=Input(shape=inp_shape,dtype=tf.float32)
            model_inputs.append(inp)

            x=PositionalEmbeddingLayer(seqlen=inp_shape[1],embedding_dim=inp_shape[0])(inp)
            configs=lstm_model_config[model_name]

            for config in configs:
                x=LSTM(
                    units=config["units"],
                    activation=config["activation"],
                    kernel_initializer=config["kernel_initializer"],
                    kernel_regularizer=config["kernel_regularizer"],
                    return_sequences=config["return_sequences"]
                )(x)

            model_outputs.append(x)

        dense_model_names=list(dense_config["Models"].keys())
        dense_input_config=dense_config["Input_Shape"]
        dense_model_config=dense_config["Models"]

        for model_name in dense_model_names:
            inp_shape=dense_input_config[model_name]
            inp=Input(shape=inp_shape,dtype=tf.float32)
            model_inputs.append(inp)

            x=inp
            configs=dense_model_config[model_name]
            for config in configs:
                x=Dense(
                    units=config["units"],
                    activation=config["activation"],
                    kernel_initializer=config["kernel_initializer"],
                    kernel_regularizer=config["kernel_regularizer"]
                )(x)

            model_outputs.append(x)
            

        out=Concatenate(axis=-1)(model_outputs)
        for config in final_model_config:
            out=Dense(
                  units=config["units"],
                  activation=config["activation"],
                  kernel_initializer=config["kernel_initializer"],
                  kernel_regularizer=config["kernel_regularizer"]
                )(out)
            

        model=Model(inputs=model_inputs,outputs=out)

        optimizer=Adam(
            learning_rate=optimizer_config["learning_rate"],
            beta_1=optimizer_config["beta_1"],
            beta_2=optimizer_config["beta_2"],
            clipnorm=optimizer_config["clipnorm"]
        )

        model.compile(
            optimizer=optimizer,
            loss=model_loss
        )

        return model
        

    
    def get_model_architecture(self,model):
        model.summary()


    
    def prepare_model_input(self,state):
        model_inp=[]

        def prepare(element):
            element=np.array(element)
            element=np.expand_dims(element,axis=0)
            return element

        for elem in state:
            inp=prepare(elem)
            model_inp.append(inp)

        return model_inp
        
    

    def get_action_from_model(self,state):
        model_inp=self.prepare_model_input(state)
        action_pred=self.actor_model.predict(model_inp,verbose=False)[0]
        action_pred=np.array(action_pred)

        action=np.argpartition(action_pred,self.env.total_stock)
        action=list(action)[-self.env.num_stock_to_picked:]

        return action
    
    
    
    def act(self,state):
        if random.random()<self.epsilon or self.env.training_episode<self.exploration_episode:
            action=self.env.action_space.sample()

        else:
            action=self.get_action_from_model(state)

        return action
            

    
    def step(self,action):
        state=self.env.get_state()
        self.env.index+=1
        send_report=False

        new_stocks=self.env.get_stock_from_index(action)
        old_stocks=self.env.get_stock_from_index(self.env.current_stocks)

        total_return_old=self.env.cal_total_return_for_stocks(old_stocks)
        total_return_new=self.env.cal_total_return_for_stocks(new_stocks)

        reward=total_return_old-total_return_new
        self.env.current_stocks=action

        next_state=self.env.get_state()

        self.env.stock_per_step.append(self.env.get_stock_from_index(action))
        
        self.env.reward_per_step.append(reward)
        self.env.return_per_step.append(total_return_new)

        self.env.set1_stocks_returns_per_step.append(self.env.cal_total_return_for_stocks(self.env.set1_stocks))
        self.env.set2_stocks_returns_per_step.append(self.env.cal_total_return_for_stocks(self.env.set2_stocks))
        self.env.set3_stocks_returns_per_step.append(self.env.cal_total_return_for_stocks(self.env.set3_stocks))

        model_inp_state=self.prepare_model_input(state)
        pred_state_value=self.critic_model.predict(model_inp_state,verbose=False)[0][0]
        self.env.pred_state_value_per_step.append(pred_state_value)

        model_inp_next_state=self.prepare_model_input(next_state)
        next_state_value=self.critic_model.predict(model_inp_next_state,verbose=False)[0][0]
        real_state_value=reward+self.gamma*next_state_value
        self.env.real_state_value_per_step.append(real_state_value)

        if self.env.index==self.env.steps-1:
            done=True
            send_report=True

        else:
            done=False

        if send_report:
            report={
                "Total Reward":sum(self.env.reward_per_step),
                "Average Reward Per Step":sum(self.env.reward_per_step)/self.env.steps,
                "Total Return":np.exp(np.array(self.env.total_return_per_step).sum()),
                "Average Return Per Step":sum(self.env.total_return_per_step)/self.steps,
                "Mean Squared Error between Real and Predicted State Value":mean_squared_error(
                    self.env.real_state_value_per_step,
                    self.env.pred_state_value_per_step
                )
    
            }

            stock_picker_report={
                "Total Returns By (Stock Picker)":np.exp(np.array(self.env.total_return_per_step).sum()),
                f"Total Returns By({'|'.join(self.env.set1_stocks)})":np.exp(np.array(self.env.set1_stocks_returns_per_step).sum()),
                f"Total Returns By({'|'.join(self.env.set2_stocks)})":np.exp(np.array(self.env.set2_stocks_returns_per_step).sum()),
                f"Total Returns By ({'|'.join(self.env.set3_stocks)})":np.exp(np.array(self.env.set3_stocks_returns_per_step).sum())
            }

        else:
            report={}
            stock_picker_report={}
        
        

        return next_state,reward,done,report,stock_picker_report

    
    def replay(self):
        data_X1=[]
        data_X2=[]
        data_X3=[]
        data_X4=[]
        data_X5=[]
        data_X6=[]
        data_X7=[]
        data_X8=[]

        actor_Y=[]
        critic_Y=[]
        actor_sample_weight=[]

        batch_data=random.sample(self.memory,self.batch_size)
        for (state,action,next_state,reward,done) in batch_data:
            if not done:
                model_inp_next_state=self.prepare_model_input(next_state)
                next_state_value=self.critic_model.predict(model_inp_next_state,verbose=False)[0][0]

                model_inp_state=self.prepare_model_input(state)
                state_value=self.critic_model.predict(model_inp_state,verbose=False)[0][0]
    
                critic_target=reward+self.gamma*next_state_value
                critic_Y.append(critic_target)

                advantage=critic_target-state_value
                actor_sample_weight.append(advantage)

                action_real=self.env.total_stock*[0.0]
                for act in action:
                    action_real[act]=1.0

                actor_Y.append(action_real)
                data_X1.append(state[0])
                data_X2.append(state[1])
                data_X3.append(state[2])
                data_X4.append(state[3])
                data_X5.append(state[4])
                data_X6.append(state[5])
                data_X7.append(state[6])
                data_X8.append(state[7])

        batch_size=len(actor_Y
                      )
        data_X1=np.array(data_X1)
        data_X2=np.array(data_X2)
        data_X3=np.array(data_X3)
        data_X4=np.array(data_X4)
        data_X5=np.array(data_X5)
        data_X6=np.array(data_X6)
        data_X7=np.array(data_X7)
        data_X8=np.array(data_X8)

        actor_Y=np.array(actor_Y)
        critic_Y=np.array(critic_Y)
        actor_sample_weight=np.array(actor_sample_weight)

        self.critic_model.fit(
            [data_X1,data_X2,data_X3,data_X4,data_X5,data_X6,data_X7,data_X8],
            critic_Y,
            batch_size=batch_size,
            epochs=1,
            verbose=False
        )

        self.actor_model.fit(
            [data_X1,data_X2,data_X3,data_X4,data_X5,data_X6,data_X7,data_X8],
            actor_Y,
            sample_weight=actor_sample_weight,
            batch_size=batch_size,
            epochs=1,
            verbose=False
        )

        if self.epsilon>self.epsilon_min:
            self.epsilon*=self.epsilon_decay
        
                
    
    def train_agent(self,episodes,training_version,verbose=True):
        self.stock_per_step_per_episode=[]
        
        self.treward=[]
        self.avg_reward_per_episode=[]
        self.avg_total_return_per_episode=[]
        self.total_return_per_episode=[]
        self.mse_state_value=[]

        self.reward_per_step_per_episode=[]
        self.return_per_step_per_episode=[]

        self.set1_stock=[]
        self.set2_stock=[]
        self.set3_stock=[]

        self.set1_return_per_step_per_episode=[]
        self.set2_return_per_step_per_episode=[]
        self.set3_return_per_step_per_episode=[]


        for ep in range(1,episodes+1):
            state,done=self.env.reset()
            while not done:
                action=self.act(state)
                next_state,reward,done,report,stock_picker_report=self.step(action)

                self.memory.append([
                    state,
                    action,
                    next_state,
                    reward,
                    done
                ])

                state=next_state

            self.stock_per_step_per_episode.append(self.env.stock_per_step)
            
            self.treward.append(report["Total Reward"])
            self.avg_reward_per_episode.append(report["Average Reward Per Step"])
            self.avg_total_return_per_episode.append(report["Average Return Per Step"])
            self.total_return_per_episode.append(report["Total Return"])
            self.mse_state_value.append(report["Mean Squared Error between Real and Predicted State Value"])

            self.reward_per_step_per_episode.append(self.env.reward_per_step)
            self.return_per_step_per_episode.append(self.env.return_per_step)

            self.set1_stock.append(self.env.set1_stocks)
            self.set2_stock.append(self.env.set2_stocks)
            self.set3_stock.append(self.env.set1_stocks)
            

            self.set1_return_per_step_per_episode.append(self.env.set1_stocks_returns_per_step)
            self.set2_return_per_step_per_episode.append(self.env.set2_stocks_returns_per_step)
            self.set3_return_per_step_per_episode.append(self.env.set3_stocks_returns_per_step)
            
                
            if verbose:
                if (ep%100)==0:
                    info=f"Episode: {ep}/{episodes}"
                    info+="\n"
                    info+=f"Epsilon: {self.epsilon}"
                    for key,value in report.items():
                        info+=f"{key}: {value}"
                        info+="\n"

                    info+="***"
                    info+="Return Comparison: Varying Stocks Selected by StockPicker vs. Holding the Same Stock Throughout"
                    info+="\n"
                    for key,value in stock_picker_report.items():
                        info+=f"{key}:{value}"
                        info+="\n"

                    info+=130*"*"
                    info+="\n\n"
                    print(info)

            if len(self.memory)>self.batch_size:
                self.replay()

            if ep==episodes:
                print("** Training Completed **")
                actor_model_name=f"actor_model_version_{training_version}.keras"
                critic_model_name=f"critic_model_version_{training_version}.keras"

                actor_model_path=os.path.join(self.model_dir_path,actor_model_name)
                critic_model_path=os.path.join(self.model_dir_path,critic_model_name)

                self.actor_model.save(actor_model_path)
                self.critic_model.save(critic_model_path)

                print(f"Actor Model save at: {actor_model_path}")
                print(f"Critic Model save at: {critic_model_path}")
                
                
    def sample_indices(self,start,end,k):
        return random.sample(range(start,end),k)
    
    
    def training_plots(self,num_plots=5):
        time_step=list(range(1,self.env.steps-1))
        
        sampled_episodes=self.sample_indices(start=self.exploration_episode,end=self.env.training_episode,k=num_plots)
        for episode in sampled_episodes:
            picked_stocks=self.stock_per_step_per_episode[episode]
            picked_stocks=["|".join(stock) for stock in picked_stocks]

            data_return=self.return_per_step_per_episode[episode]
            data=pd.DataFrame(data_return,columns=["Stock Returns"])
            data["Stocks"]=picked_stocks
            data.index=time_step

            print("Stock Returns and Stock Picked By Agent at a Time Step")
            print(130*"*")
            print("\n")
            print(f"Training Episode: {episode}")
            
            time_step_index=self.sample_indices(start=0,end=self.env.steps-1,k=3)
            for index in time_step_index:
                data_index=data.iloc[index:index+5]
                index=list(data_index.index)
                returns=list(data_index["Stock Returns"])
                labels=list(data_index["Stocks"])

                plt.scatter(index,returns,color="black")
                for i in range(5):
                    plt.annotate(labels[i],
                                 (index[i],returns[i]),
                                 textcoords="offset points",
                                 xytext=(5,5), 
                                 ha='center',
                                 color="black")

                plt.plot(index,returns,lw=1.0,c="b")
                plt.xlabel("Step")
                plt.ylabel("Stock Returns")
                plt.title(f"Time Step: {index}-{index+5}| Stock Picked By Agent| Stock Returns")
                plt.legend()
                plt.show()

            print(130*"*")
            print("\n")


        print("Return Comparison: Varying Stocks Selected by Agent vs. Holding the Same Stock Throughout")
        sampled_episodes=self.sample_indices(start=self.exploration_episode,end=self.env.training_episode,k=num_plots)
        for episode in sampled_episodes:
            set1_stock=self.set1_stock[episode]
            set2_stock=self.set2_stock[episode]
            set3_stock=self.set3.stock[episode]

            set1_stock="|".join(set1_stock)
            set2_stock="|".join(set2_stock)
            set3_stock="|".join(set3_stock)

            set1_stock_return=self.set1_return_per_step_per_episode[episode]
            set2_stock_return=self.set2_return_per_step_per_episode[episode]
            set3_stock_return=self.set3_return_per_step_per_episode[episode]

            stock_return=self.return_per_step_per_episode[episode]

            data=pd.DataFrame(stock_return,columns=["Stock Returns of stock picked by Agent"],index=time_step)
            data[f"Stock Returns of {set1_stock}"]=set1_stock_return
            data[f"Stock Returns of {set2_stock}"]=set2_stock_return
            data[f"Stock Returns of {set3_stock}"]=set3_stock_return

            data.plot(figsize=(10,6),style=["r","b","g","c"])
            plt.xlabel("Time Steps")
            plt.ylabel("Stock Returns")
            plt.title(f"Training Episode: {episode}")
            plt.legend()
            plt.show()

        print(130*"*")
        print("\n")

        print("Rewards during Agent Execution")
        sampled_episodes=self.sample_indices(start=self.exploration_episode,end=self.env.training_episode,k=num_plots)
        for episode in sampled_episodes:
            reward_data=self.reward_per_step_per_episode[episode]
            data=pd.DataFrame(reward_data,columns=["Reward"],index=time_step)

            data.plot(figsize=(10,6),style=["g"])
            plt.xlabel("Time Step")
            plt.ylabel("Reward")
            plt.title(f"Training Episode: {episode}| Time Step VS Reward")
            plt.show()

        print(130*"*")
        print("\n")


    def episode_plots(self):
        episodes=list(range(1,self.env.training_episode+1))

        data=self.total_return_per_episode
        plt.plot(episodes,data,lw=1.0,c="b")
        plt.xlabel("Episodes")
        plt.ylabel("Returns")
        plt.title("Episodes VS Returns of stocks picked By Agent")
        plt.show()

        data=self.avg_total_return_per_episode
        plt.plot(episodes,data,lw=1.0,c="g")
        plt.xlabel("Episodes")
        plt.ylabel("Returns")
        plt.title("Episodes VS Average Returns Per Episode of stock picked By Agent")
        plt.show()

        data=self.treward
        plt.plot(episodes,data,lw=1.0,c="c")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title("Episode VS Total Reward received by Agent")
        plt.show()

        data=self.avg_reward_per_episode
        plt.plot(episodes,data,lw=1.0,c="c")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title("Episode VS Average Reward Per Episode received by Agent")
        plt.show()

        data=self.mse_state_value
        plt.plot(episodes,data,lw=1.0,c="r")
        plt.xlabel("Episodes")
        plt.ylabel("MSE")
        plt.title("Episodes VS Mean Squared Error between Real and Predicted State Value")
        plt.show()


    def test_agent(self,verbose=True,plotting=True):
        state,done=self.env.reset()
        while not done:
            action=self.get_action_from_model(state)
            next_state,reward,done,report,stock_picker_report=self.step(action)

            state=next_state

        if verbose:
            info=""
            for key,value in report.items():
                info+=f"{key}: {value}"
                info+="\n"

            info+="***"
            info+="Return Comparison: Varying Stocks Selected by StockPicker vs. Holding the Same Stock Throughout"
            info+="\n"
            for key,value in stock_picker_report.items():
                info+=f"{key}:{value}"
                info+="\n"

            info+=130*"*"
            info+="\n\n"
            print(info)

        if plotting:
            self.test_plots()


    def test_plots(self):
        time_step=list(range(1,self.env.steps-1))

        picked_stocks=self.env.stock_per_step
        picked_stocks=["|".join(stock) for stock in picked_stocks]

        returns_data=self.env.return_per_step
        data=pd.DataFrame(returns_data,columns=["Returns"])
        data["Stocks"]=picked_stocks
        data["Steps"]=time_step

        print("(Testing) Stock Returns and Stock Picked By Agent at a Time Step")
        print(130*"*")
        print("\n")
        
        sampled_indices=self.sample_indices(start=0,end=self.env.steps-1,k=5)
        for ind in sampled_indices:
            data_ind=data.iloc[ind:ind+5]
            
            x=list(data_ind["Steps"])
            y=list(data_ind["Returns"])
            labels=list(data_ind["Stocks"])

            plt.scatter(x,y,color="black")

            for i in range(5):
                plt.annotate(labels[i],
                             (x[i],y[i]),
                             textcoords="offset points",
                             xytext=(5,5),
                             ha="center",
                             color="black")

            plt.plot(x,y,lw=1.0,c="b")
            plt.xlabel("Step")
            plt.ylabel("Stock Returns")
            plt.title(f"Time Step: {ind}-{ind+5}| Stock Picked By Agent| Stock Returns")
            plt.legend()
            plt.show()
            
        
        print(130*"*")
        print("\n")

        print("Return Comparison (Testing): Varying Stocks Selected by Agent vs. Holding the Same Stock Throughout")
        returns_data=self.env.returns_per_step

        set1_stock=self.env.set1_stocks
        set2_stock=self.env.set2_stocks
        set3_stock=self.env.set3_stocks

        set1_stock="|".join(set1_stock)
        set2_stock="|".join(set2_stock)
        set3_stock="|".join(set3_stock)

        set1_stock_return=self.env.set1_stocks_returns_per_step
        set2_stock_return=self.env.set2_stocks_returns_per_step
        set3_stock_return=self.env.set3_stocks_returns_per_step

        stock_return=self.env.return_per_step

        data=pd.DataFrame(stock_return,columns=["Stock Returns of stock picked by Agent"],index=time_step)
        data[f"Stock Returns of {set1_stock}"]=set1_stock_return
        data[f"Stock Returns of {set2_stock}"]=set2_stock_return
        data[f"Stock Returns of {set3_stock}"]=set3_stock_return

        data.plot(figsize=(10,6),style=["r","b","g","c"])
        plt.xlabel("Time Steps")
        plt.ylabel("Stock Returns")
        plt.legend()
        plt.show()

        print(130*"*")
        print("\n")

        plt.plot(time_step,self.env.reward_per_step,lw=1.0,c="g")
        plt.xlabel("Time Step")
        plt.ylabel("Reward")
        plt.title("Time Step VS Reward Per Step received by Agent")
        plt.show()

        
