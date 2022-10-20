import streamlit as st
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier

# input_txt = st.sidebar.text_area('Input')
# input_df = pd.read_table(input_txt)

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  input_df = pd.read_excel(uploaded_file)
  st.write(input_df)

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
  weights_df = pd.DataFrame.from_dict(weights, orient = 'index')*100.0
  weights_df.columns = ['Optimal Weight (%)']

  expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(verbose=True)
  portfolio_performance_df = pd.DataFrame([expected_annual_return*100.0,annual_volatility*100.0],
                                       index=['Expected Annual Return (%)', 'Annual Volatility (%)'],
                                       columns=['Portfolio Performance'])


  # x = 10
  # 'x: ', x 

#   st.subheader("Input")
#   st.write(input_txt)
  # st.dataframe(input_df)

  st.subheader("Optimization Result")
  st.dataframe(portfolio_performance_df.style.format(precision=2))

  # st.subheader("Optimized Max Sharpe Portfolio Weights")
  st.dataframe(weights_df.reindex(assets).style.format(precision=2))
