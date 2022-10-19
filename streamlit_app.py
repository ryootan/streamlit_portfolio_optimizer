import streamlit as st
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier

assets = ['A','B']
mu = pd.Series([1.0,20.0],index=assets)
S = pd.DataFrame({'A':[0.01,0.05],'B':[0.05,1.00]},index=assets)
ef = EfficientFrontier(mu, S)
ef.max_sharpe()
weights = ef.clean_weights()
weights_df = pd.DataFrame.from_dict(weights, orient = 'index')
weights_df.columns = ['weights']

ef.portfolio_performance(verbose=True)

x = 10
'x: ', x 

st.subheader("Optimized Max Sharpe Portfolio Weights")
st.dataframe(weights_df)
