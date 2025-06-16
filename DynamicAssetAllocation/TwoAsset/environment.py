import os
import random
import numpy as np
import pandas as pd
from pylab import plt,mpl
from dotenv import load_dotenv
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
        sma_period,
        window,
        env_type="normal"
        
    ):
        self.symbol=symbol
        self.short_rate=short_rate
        self.maturity=maturity
        self.steps=steps
        self.dt=self.maturity/self.steps
        self.number_lags=number_lags
        self.sma_period=sma_period
        self.window=window
        self.index=0
        self.xt=0
        self.yt=0
        self.xt_values=list()
        self.env_type=env_type
        
        self.ts=TimeSeries(key=os.environ["ALPHA_VANTAGE_KEY"],output_format="pandas",indexing_type="date")

        self.get_raw_data()
        self.prepare_data()

        self.initial_value=self.final_data["Xt"].iloc[0]
        self.action_space=Action_Space()


    def get_raw_data(self):
        raw_data,_=self.ts.get_daily(symbol=self.symbol,outputsize="full")
        raw_data=pd.DataFrame(raw_data["4. close"]).dropna()
        raw_data.rename(columns={"4. close":"Xt"},inplace=True)
        raw_data=raw_data.iloc[:self.steps]
        raw_data=raw_data.iloc[::-1]

        initial_value=raw_data["Xt"].iloc[0]
        risk_free_asset_data=initial_value*np.exp(self.short_rate*np.arange(self.steps)*self.dt)
        raw_data["Yt"]=risk_free_asset_data
        
        self.raw_data=raw_data

    def prepare_data(self):
        data=self.raw_data
        
        data["Xt_returns"]=np.log(data["Xt"]/data["Xt"].shift(1))

        self.xt_features=["Xt"]
        self.yt_features=["Yt"]
        self.xt_return_features=["Xt_returns"]
        
        for lag in range(1,self.number_lags+1):
            data[f"Xt_lag{lag}"]=data["Xt"].shift(lag)
            data[f"Yt_lag{lag}"]=data["Yt"].shift(lag)
            data[f"Xt_returns_lag{lag}"]=data["Xt_returns"].shift(lag)
            
            self.xt_features.append(f"Xt_lag{lag}")
            self.yt_features.append(f"Yt_lag{lag}")
            self.xt_return_features.append(f"Xt_returns_lag{lag}")

        data["Xt_momentum"]=data["Xt_returns"].rolling(self.window).mean()
        data["Xt_volatility"]=data["Xt_returns"].rolling(self.window).std()

        data["Xt_distance"]=data["Xt"]-data["Xt"].rolling(self.sma_period).mean()
        data["Xt_min"]=data["Xt"].rolling(self.sma_period).min()
        data["Xt_max"]=data["Xt"].rolling(self.sma_period).max()

        data.dropna(inplace=True)

        self.final_data=data

    def get_state(self):
        state=[]
        data=self.final_data.iloc[self.index]

        data_xt=list(data[self.xt_features])
        data_yt=list(data[self.yt_features])

        state.append(list(zip(data_xt,data_yt))[::-1])

        data_xt_returns=list(data[self.xt_return_features])
        data_xt_returns=[[v] for v in data_xt_returns]

        state.append(data_xt_returns[::-1])

        state.append(list(data[["Xt","Yt"]]))
        state[2].append(self.xt)
        state[2].append(self.yt)

        state.append(list(data[["Xt_distance","Xt_min","Xt_max","Xt_momentum","Xt_volatility"]]))

        return state

    
    def reset(self):
        self.index=0
        self.xt=0
        self.yt=0
        self.pl=list()
        self.predicted_pl=list()
        self.pl_percent=list()
        self.pvalue=list()
        
        state=self.get_state()
        return state,False

    def plots(self):

        self.final_data[["Xt"]].plot(figsize=(10,6),style=["b"])
        plt.xlabel("Time steps")
        plt.ylabel("Price")
        plt.title(f"Risky Asset ({self.symbol})| Time VS Price")
        plt.show()

        
        self.final_data[["Yt"]].plot(figsize=(10,6),style=["g"])
        plt.xlabel("Time steps")
        plt.ylabel("Price")
        plt.title(f"Risk Free Asset| Short Rate: {self.short_rate}| Time VS Price")
        plt.show()

        
        self.final_data[["Xt","Xt_min","Xt_max"]].plot(figsize=(10,6),style=["b","c","r"])
        plt.xlabel("Time steps")
        plt.ylabel("Price")
        plt.title(f"Risky Asset ({self.symbol})| Stock Price| Min| Max")
        plt.show()

        self.final_data[["Xt_distance"]].plot(figsize=(10,6),style=["g"])
        plt.xlabel("Time steps")
        plt.ylabel("Distance")
        plt.title(f"Risky Asset ({self.symbol})| Distance")
        plt.show()

        self.final_data[["Xt_returns"]].cumsum().apply(np.exp).plot(figsize=(10,6),style=["b"])
        plt.xlabel("Time Steps")
        plt.ylabel("Returns")
        plt.title(f"Risky Asset ({self.symbol})| Returns Vs Time")
        plt.show()

        self.final_data[["Xt_momentum"]].plot(figsize=(10,6),style=["g"])
        plt.xlabel("Time Steps")
        plt.ylabel("Momentum")
        plt.title(f"Risky Asset ({self.symbol})| Momentum Vs Time")
        plt.show()

        self.final_data[["Xt_volatility"]].plot(figsize=(10,6),style=["c"])
        plt.xlabel("Time Steps")
        plt.ylabel("Volatility")
        plt.title(f"Risky Asset ({self.symbol})| Volatility Vs Time")
        plt.show()
