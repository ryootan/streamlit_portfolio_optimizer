import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier

assets = ['A','B']
mu = pd.Series([1.0,2.0],index=assets)
S = pd.DataFrame({'A':[0.01,0.05],'B':[0.05,1.00]},index=assets)
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()



x = 10
'x: ', x 
