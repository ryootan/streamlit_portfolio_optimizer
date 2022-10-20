import streamlit as st
import pandas as pd
from io import StringIO
from pypfopt.efficient_frontier import EfficientFrontier

st.subheader("Input")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  risk_contribution_asset_class_df = pd.read_excel(uploaded_file,sheet_name = 'Risk Contribution - Asset Class').dropna()
  risk_contribution_asset_class_df['Asset'] = [ele['Asset'] + ('' if ele['FX Hedged'] == 'No' else ' (Hedged)') for _,ele in risk_contribution_asset_class_df.iterrows()]
  risk_contribution_asset_class_df = risk_contribution_asset_class_df.set_index('Asset')
  risk_asset_class_corr_mtx_df = pd.read_excel(uploaded_file,sheet_name = 'Risk - Asset Class Corr Mtx').dropna().set_index('Asset Classes')
    
  vol = risk_contribution_asset_class_df['Asset Volatility']
  corr = risk_asset_class_corr_mtx_df
  S = corr.mul(vol,axis='columns').mul(vol,axis='index')
  st.write(S)
  
  target_return = st.number_input('Insert your Target Return (% p.a.)',value=3.5)
  input_format_df = pd.DataFrame({'Asset':vol.index})
  input_format_df['Expected Return (% p.a.)'] = 0.0
  input_format_df['Lower Bound (%)'] = 0.0
  input_format_df['Upper Bound (%)'] = 100.0
  input_format_df['Risk Weight (%)'] = 0.0
  
  input_txt = st.text_area('Insert optimization inputs',value=input_format_df.to_csv(sep='\t',index=False),height=30)
  input_df = pd.read_csv(StringIO(input_txt),sep='\t')
  
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

#   st.write(input_txt)
  # st.dataframe(input_df)

  st.subheader("Optimization Result")
  st.dataframe(portfolio_performance_df.style.format(precision=2))

  # st.subheader("Optimized Max Sharpe Portfolio Weights")
  st.dataframe(weights_df.reindex(assets).style.format(precision=2))
