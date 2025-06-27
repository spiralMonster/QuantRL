import random
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,LSTM
from tensorflow.keras.layers import Permute
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from collections import deque
from pylab import plt,mpl
from positional_embedding_layer import PositionalEmbeddingLayer

plt.style.use("seaborn-v0_8")
mpl.rcParams["figure.dpi"]=300
mpl.rcParams["savefig.dpi"]=300
mpl.rcParams["font.family"]="serif"


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
        exploration_episodes,
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
        

        if not self.model_trained:
            self.actor_model=create_model(
                model_config=self.actor_model_config,
                optimizer_config=self.actor_optimizer_config,
                model_loss=self.actor_model_loss
            )
    
            self.critic_model=create_model(
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
        

    def cal_total_return_for_stocks(stocks):
        total_return=0.0
        data=self.env.final_data.iloc[self.env.index]

        for sym in stocks:
            data_sym=np.array(data[[f"{sym}_lag_{lag}" for lag in range(1,self.env.execution_gap)]])
            ret=np.exp(data_sym.sum(axis=1))
            total_return+=ret

        total_return/=self.env.num_stock_to_picked
        return total_return
    
    
    def act(self,state):
        if random.random()<self.epsilon or self.env.training_episode<self.exploration_episode:
            action=self.env.action_space.sample()

        else:
            model_inp=self.prepare_model_input(state)
            action_pred=self.actor_model.predict(model_inp,verbose=False)[0]
            action_pred=np.array(action_pred)

            action=np.argpartition(action_pred,self.env.total_stock)
            action=list(action)[-self.env.num_stock_to_picked:]
            return action
            

    def step(self,action):
        self.env.index+=1
        send_report=False

        new_stocks=[self.env.index_to_stock[act] for act in action]
        old_stocks=[self.env.index_to_stock[ind] for ind in self.env.current_stocks]

        total_return_old=self.cal_total_return_for_stocks(old_stocks)
        total_return_new=self.cal_total_return_for_stocks(new_stocks)

        reward=total_return_old-total_return_new
        self.env.current_stocks=action

        self.env.reward_per_step.append(reward)
        self.env.total_return_per_step.append(total_return_new)

        if self.env.index==self.env.steps-1:
            done=True
            send_report=True

        else:
            done=False

        if send_report:
            report={}

        else:
            report={}
        
        
        next_state=self.env.get_state()

        return next_state,reward,done,report
            

    

        

            

        
    


