import random
import math
import numpy as np
import pandas as pd
from pylab import plt,mpl
from action_space import Action_Space
from environment import Environment

plt.style.use("seaborn-v0_8")
mpl.rcParams["figure.dpi"]=300
mpl.rcParams["savefig.dpi"]=300
mpl.rcParams["font.family"]="serif"

class Simulated_Environment(Environment):
    def __init__(
        self,
        initial_value,
        short_rate:list,
        index_drift:list,
        index_volatility:list,
        maturity,
        steps,
        number_lags,
        sma_period,
        window
    ):
        self.initial_value=initial_value
        self.short_rate=short_rate
        self.index_drift=index_drift
        self.index_volatility=index_volatility
        self.maturity=maturity
        self.steps=steps
        self.dt=self.maturity/self.steps
        self.number_lags=number_lags
        self.sma_period=sma_period
        self.window=window
        self.index=0
        self.xt=0
        self.yt=0

        self.action_space=Action_Space()

        self.curr_index_drift=random.choice(self.index_drift)
        self.curr_volatility=random.choice(self.index_volatility)
        self.curr_short_rate=random.choice(self.short_rate)

        self.get_raw_data()
        self.prepare_data()

        
    def get_raw_data(self):
        S=[self.initial_value]
        
        for t in range(1,self.steps):
            st=S[t-1]*math.exp((self.curr_index_drift-      (self.curr_volatility**2)/2)*self.dt+self.curr_volatility*math.sqrt(self.dt)*random.gauss(0,1))
            S.append(st)

        raw_data=pd.DataFrame(S,columns=["Xt"])

        risk_free_asset_data=self.initial_value*np.exp(self.curr_short_rate*np.arange(self.steps)*self.dt)
        raw_data["Yt"]=risk_free_asset_data
        self.raw_data=raw_data

    def reset(self):
        self.index=0
        self.xt=0
        self.yt=0
        self.trewards=list()

        self.curr_index_drift=random.choice(self.index_drift)
        self.curr_volatility=random.choice(self.index_volatility)
        self.curr_short_rate=random.choice(self.short_rate)

        self.get_raw_data()
        self.prepare_data()

        state=self.get_state()
        return state,False

    def plots(self):
        self.final_data[["Xt"]].plot(figsize=(10,6),style=["b"])
        plt.xlabel("Time steps")
        plt.ylabel("Price")
        plt.title(f"Risky Asset| Index Drift: {self.curr_index_drift}| Index Volatility: {self.curr_volatility}")
        plt.show()

        
        self.final_data[["Yt"]].plot(figsize=(10,6),style=["g"])
        plt.xlabel("Time steps")
        plt.ylabel("Price")
        plt.title(f"Risk Free Asset| Short Rate: {self.curr_short_rate}| Time VS Price")
        plt.show()

        
        self.final_data[["Xt","Xt_min","Xt_max"]].plot(figsize=(10,6),style=["b","c","r"])
        plt.xlabel("Time steps")
        plt.ylabel("Price")
        plt.title(f"Risky Asset| Index Drift: {self.curr_index_drift}| Index Volatility: {self.curr_volatility}")
        plt.show()

        self.final_data[["Xt_distance"]].plot(figsize=(10,6),style=["g"])
        plt.xlabel("Time steps")
        plt.ylabel("Distance")
        plt.title(f"Risky Asset| Index Drift: {self.curr_index_drift}| Index Volatility: {self.curr_volatility}")
        plt.show()

        self.final_data[["Xt_returns"]].cumsum().apply(np.exp).plot(figsize=(10,6),style=["b"])
        plt.xlabel("Time Steps")
        plt.ylabel("Returns")
        plt.title(f"Risky Asset| Index Drift: {self.curr_index_drift}| Index Volatility: {self.curr_volatility}")
        plt.show()

        self.final_data[["Xt_momentum"]].plot(figsize=(10,6),style=["g"])
        plt.xlabel("Time Steps")
        plt.ylabel("Momentum")
        plt.title(f"Risky Asset| Index Drift: {self.curr_index_drift}| Index Volatility: {self.curr_volatility}")
        plt.show()

        self.final_data[["Xt_volatility"]].plot(figsize=(10,6),style=["c"])
        plt.xlabel("Time Steps")
        plt.ylabel("Volatility")
        plt.title(f"Risky Asset| Index Drift: {self.curr_index_drift}| Index Volatility: {self.curr_volatility}")
        plt.show()

        