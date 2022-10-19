import streamlit as st
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier

target_return = 3.0
assets = ['A','B']
score = [1.0,1.4]
max_wgtavg_score = 2.0

mu = pd.Series([1.0,20.0],index=assets)
S = pd.DataFrame({'A':[0.01,0.05],'B':[0.05,1.00]},index=assets)
ef = EfficientFrontier(mu, S)
ef.add_constraint(lambda w: score @ w <= max_wgtavg_score)
ef.efficient_return(target_return)
# ef.max_sharpe()
weights = ef.clean_weights()
weights_df = pd.DataFrame.from_dict(weights, orient = 'index')
weights_df.columns = ['Optimal Weights']

expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(verbose=True)
portfolio_performance_df = pd.DataFrame([(expected_annual_return*100).round(2),(annual_volatility*100).round(2)],
                                     index=['Expected Annual Return (%)', 'Annual Volatility (%)'],
                                     columns=['Portfolio Performance'])


# x = 10
# 'x: ', x 

st.subheader("Optimization Result")
st.dataframe(portfolio_performance_df)

# st.subheader("Optimized Max Sharpe Portfolio Weights")
st.dataframe(weights_df.reindex(assets)

# st.subheader('Expected annual return: {}%'.format((expected_annual_return*100).round(2)))
# st.subheader('Annual volatility: {}%'.format((annual_volatility*100).round(2)))
# st.subheader('Sharpe Ratio: {}'.format(sharpe_ratio.round(2)))
