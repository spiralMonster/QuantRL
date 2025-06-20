import os
import random
import math
from scipy import stats
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pylab import plt,mpl
from alpha_vantage.timeseries import TimeSeries
from observation_space import Observation_Space
from action_space import Action_Space
load_dotenv()

plt.style.use("seaborn-v0_8")
mpl.rcParams["figure.dpi"]=300
mpl.rcParams["savefig.dpi"]=300
mpl.rcParams["font.family"]="serif"

class Environment:
    def __init__(
        self,
        symbol,
        short_rate,
        maturity,
        steps,
        number_lags,
        window,
        env_type="normal"
    ):

        self.symbol=symbol
        self.short_rate=short_rate
        self.maturity=maturity
        self.steps=steps
        self.dt=self.maturity/self.steps
        self.number_lags=number_lags
        self.window=window
        self.env_type=env_type
        self.index=0
        self.stock=0
        self.bond=0

        self.ts=TimeSeries(key=os.environ["ALPHA_VANTAGE_KEY"],indexing_type="date",output_format="pandas")

        self.action_space=Action_Space()
        
        self.get_raw_data()
        self.prepare_data()
        

    def get_option_value(self,st,t):
        K=self.strike_price
        sig=self.volatility
        r=self.short_rate
        T=self.maturity
        t=t*self.dt

        d1=(np.log(st/K)+(r+0.5*(sig**2))*(T-t))/(sig*((T-t)**0.5))
        d2=(np.log(st/K)+(r-0.5*(sig**2))*(T-t))/(sig*((T-t)**0.5))

        N_d1=stats.norm.cdf(d1,0,1)
        N_d2=stats.norm.cdf(d2,0,1)

        option_value=st*N_d1-np.exp(-r*(T-t))*K*N_d2

        return option_value
        

    def get_delta(self,st,t):
        K=self.strike_price
        sig=self.volatility
        r=self.short_rate
        T=self.maturity
        t*=self.dt

        d1=(np.log(st/K)+(r+0.5*(sig**2))*(T-t))/(sig*((T-t)**0.5))
        delta=stats.norm.cdf(d1,0,1)

        return delta
        


    def get_raw_data(self):
        raw_data,_=self.ts.get_daily(symbol=self.symbol,outputsize="full")
        raw_data=pd.DataFrame(raw_data["4. close"])
        raw_data=raw_data.iloc[:self.steps]
        raw_data=raw_data.iloc[::-1]
        raw_data.rename(columns={"4. close":"Xt"},inplace=True)
        raw_data.dropna(inplace=True)
        
        raw_data["Xt_returns"]=np.log(raw_data["Xt"]/raw_data["Xt"].shift(1))
        self.volatility=raw_data["Xt_returns"].std()*np.sqrt(252)
        
        self.initial_value=raw_data["Xt"].iloc[0]
        self.strike_price=self.initial_value

        bond_values=self.initial_value*np.exp(self.short_rate*np.arange(self.steps)*self.dt)
        raw_data["Yt"]=bond_values
        
        raw_data["steps"]=list(range(self.steps))
        raw_data["time_to_mature"]=self.maturity-raw_data["steps"]*self.dt

        raw_data["Ct"]=self.get_option_value(raw_data["Xt"],raw_data["steps"])
        raw_data["Ct_returns"]=np.log(raw_data["Ct"]/raw_data["Ct"].shift(1))

        raw_data["delta"]=self.get_delta(raw_data["Xt"],raw_data["steps"])
        
        raw_data.drop(columns=["steps"],axis=1,inplace=True)
        raw_data.dropna(inplace=True)


        self.raw_data=raw_data


    def prepare_data(self):
        data=self.raw_data

        self.xt_features=["Xt"]
        self.yt_features=["Yt"]
        self.ct_features=["Ct"]

        for lag in range(1,self.number_lags+1):
            data[f"Xt_lag_{lag}"]=data["Xt"].shift(lag)
            data[f"Yt_lag_{lag}"]=data["Yt"].shift(lag)
            data[f"Ct_lag_{lag}"]=data["Ct"].shift(lag)

            self.xt_features.append(f"Xt_lag_{lag}")
            self.yt_features.append(f"Yt_lag_{lag}")
            self.ct_features.append(f"Ct_lag_{lag}")
            

        self.xt_return_features=["Xt_returns"]
        self.ct_return_features=["Ct_returns"]

        for lag in range(1,self.number_lags+1):
            data[f"Xt_returns_lag_{lag}"]=data["Xt_returns"].shift(lag)
            data[f"Ct_returns_lag_{lag}"]=data["Ct_returns"].shift(lag)

            self.xt_return_features.append(f"Xt_returns_lag_{lag}")
            self.ct_return_features.append(f"Ct_returns_lag_{lag}")
            
        
        data["Xt_distance"]=data["Xt"]-data["Xt"].rolling(self.window).mean()
        data["Ct_distance"]=data["Ct"]-data["Ct"].rolling(self.window).mean()

        self.xt_dis_features=["Xt_distance"]
        self.ct_dis_features=["Ct_distance"]

        data["Xt_vol"]=data["Xt_returns"].rolling(self.window).std()
        data["Ct_vol"]=data["Ct_returns"].rolling(self.window).std()

        self.xt_vol_features=["Xt_vol"]
        self.ct_vol_features=["Ct_vol"]

        for lag in range(1,self.number_lags+1):
            data[f"Xt_distance_lag_{lag}"]=data["Xt_distance"].shift(lag)
            data[f"Ct_distance_lag_{lag}"]=data["Ct_distance"].shift(lag)

            self.xt_dis_features.append(f"Xt_distance_lag_{lag}")
            self.ct_dis_features.append(f"Ct_distance_lag_{lag}")

            data[f"Xt_vol_lag_{lag}"]=data["Xt_vol"].shift(lag)
            data[f"Ct_vol_lag_{lag}"]=data["Ct_vol"].shift(lag)

            self.xt_vol_features.append(f"Xt_vol_lag_{lag}")
            self.ct_vol_features.append(f"Ct_vol_lag_{lag}")

        data.dropna(inplace=True)

        self.steps=len(data)
        data.index=list(range(self.steps))
        self.final_data=data

    def get_state(self):
        state=[]

        xt_data=list(self.final_data[self.xt_features].iloc[self.index])
        yt_data=list(self.final_data[self.yt_features].iloc[self.index])
        ct_data=list(self.final_data[self.ct_features].iloc[self.index])

        state.append(list(zip(xt_data,yt_data,ct_data)))

        xt_return_data=list(self.final_data[self.xt_return_features].iloc[self.index])
        ct_return_data=list(self.final_data[self.ct_return_features].iloc[self.index])

        state.append(list(zip(xt_return_data,ct_return_data)))

        xt_dis_data=list(self.final_data[self.xt_dis_features].iloc[self.index])
        ct_dis_data=list(self.final_data[self.ct_dis_features].iloc[self.index])

        state.append(list(zip(xt_dis_data,ct_dis_data)))

        xt_vol_data=list(self.final_data[self.xt_vol_features].iloc[self.index])
        ct_vol_data=list(self.final_data[self.ct_vol_features].iloc[self.index])

        state.append(list(zip(xt_vol_data,ct_vol_data)))

        other_data=[
            self.final_data["Xt"].iloc[self.index],
            self.final_data["Yt"].iloc[self.index],
            self.final_data["Ct"].iloc[self.index],
            self.strike_price,
            self.short_rate,
            self.final_data["time_to_mature"].iloc[self.index],
            self.stock,
            self.bond
        ]

        state.append(other_data)

        return state

    def reset(self):
        self.index=0
        self.stock=0
        self.bond=0

        self.phi_value_per_step=[]
        self.reward_per_step=[]
        self.pl_per_step=[]
        self.pl_percent_per_step=[]
        self.model_delta_per_step=[]
        self.bond_weight_per_step=[]
        self.predicted_qvalue_per_step=[]
        self.real_qvalue_per_step=[]

        state=self.get_state()

        return state,False

    def plots(self):

        self.final_data[["Xt","Yt"]].plot(figsize=(10,6),style=["b","g"])
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.title(f"Risky Asset(Xt): {self.symbol}| Risk free Asset(Yt): Bond")
        plt.legend()
        plt.show()

        self.final_data[["Ct"]].plot(figsize=(10,6),style=["c"])
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.title(f"Option (Ct)| Time Vs Price")
        plt.show()

        self.final_data[["delta"]].plot(figsize=(10,6),style=["r"])
        plt.xlabel("Time")
        plt.title("Time VS Delta")
        plt.show()

        plt.plot(list(self.final_data["Xt"]),list(self.final_data["delta"]),lw=1.0,c="g")
        plt.ylim(0,1)
        plt.xlabel("Stock Price")
        plt.ylabel("Delta")
        plt.title("Stock Price VS Delta")
        plt.show()

        self.final_data[["Xt_distance"]].plot(figsize=(10,6),style=["b"])
        plt.xlabel("Time")
        plt.title(f"Risk Asset(Xt): {self.symbol}| Time Vs Index Drift")
        plt.show()

        self.final_data[["Xt_vol"]].plot(figsize=(10,6),style=["g"])
        plt.xlabel("Time")
        plt.title(f"Risk Asset(Xt): {self.symbol}| Time Vs Index Volatility")
        plt.show()

        self.final_data[["Ct_distance"]].plot(figsize=(10,6),style=["c"])
        plt.xlabel("Time")
        plt.title(f"Option(Ct) | Time Vs Option Drift")
        plt.show()

        self.final_data[["Ct_vol"]].plot(figsize=(10,6),style=["r"])
        plt.xlabel("Time")
        plt.title(f"Option(Ct) | Time Vs Option Volatility")
        plt.show()

