# python
import sys
from binance.client import Client
import pandas as pd
import statsmodels.formula.api as smf
import sklearn.metrics as sm
import numpy as np
import datetime
import math

def measure(a, b):
	# calculating the mean and std deviation for a and b 
    meanOfA, meanOfB = np.mean(a), np.mean(b)
    stdOfA, stdOfB = np.std(a), np.std(b)
    n, result = len(a), 0
    for i in range(0,n):
        numer = ((meanOfA - a[i]) * (meanOfB - b[i]))
        denom = float(stdOfA * stdOfB * n)
        result += (numer/denom)
    return result
def computeDelta(wt, X, Xi):
    """
    This function computes equation 6 of the paper, but with the euclidean distance 
    replaced by the similarity function given in Equation 9.

    Parameters
    ----------
    wt : int
        This is the constant c at the top of the right column on page 4.
    X : A row of Panda Dataframe
        Corresponds to (x, y) in Equation 6.
    Xi : Panda Dataframe
        Corresponds to a dataframe of (xi, yi) in Equation 6.

    Returns
    -------
    float
        The output of equation 6, a prediction of the average price change.
    """

    numerator, denominator = 0, 0
    n = len(X) - 1
    matrx = Xi.values
    for i in range(0,len(matrx)):
        numerator +=  (matrx[i][n] * math.exp(wt * measure(X[0:n], matrx[i][0:n])))
        denominator += math.exp(wt * measure(X[0:n], matrx[i][0:n]))
    E_emp = float(numerator)/denominator
    return E_emp

# Main class
class TradingBot:
    # Read env
    def __read_env_dict(path: str) -> dict:
        with open(path, 'r') as f:
            return dict(tuple(line.replace('\n', '').split('=')) for line
                    in f.readlines() if not line.startswith('#'))

    # Start the binance client
    def __start_client(self):
        self.__api_key = self.env.get("API_KEY","")
        self.__api_secret = self.env.get("API_SECRET","")
        self.client=Client(self.__api_key,self.__api_secret)

    # Initialize the bot
    def __init__(self, env_path=".env"):
        self.env = TradingBot.__read_env_dict(env_path)
        self.__start_client()
        self.model=None

    # Returns three arays containing differences in price over interval of 1 Min, with Total diff at the end
    def __get_market_data(self,start,end):
        # Market
        asset="ETHEUR"
        # Intervals docs
        # https://python-binance.readthedocs.io/en/latest/constants.html?highlight=Binance%20Kline%20interval#binance-constants
        # Granularity
        timeframe="1m"
        df= pd.DataFrame(self.client.get_historical_klines(symbol=asset, interval=timeframe,start_str=start,end_str=end))
        df=df.iloc[:,:6]
        df.columns=["Date","Open","High","Low","Close","Volume"]
        df=df.set_index("Date")
        df.index=pd.to_datetime(df.index,unit="ms")
        df= df.astype("float")
        # Retreive the differences in three sets
        data_full = []
        Yi_full = None
        data_half = []
        Yi_half = None
        data_quart = []
        Yi_quart = None
        last = None
        counter = 0
        for it in df.iterrows():
            counter+=1
            if counter == math.floor(len(df) / 2):
                Yi_half = it[1].values[3]
            if counter == math.floor(len(df) / 4):
                Yi_quart = it[1].values[3]
            if last:
                data_full.append(it[1].values[3]-last)
                data_half.append(it[1].values[3]-last)
                data_quart.append(it[1].values[3]-last)
            else:
                Yi_full= it[1].values[3]
            last = it[1].values[3]
        Yi_full = Yi_full - last
        Yi_half = Yi_half - last
        Yi_quart = Yi_quart - last
        data_full.append(Yi_full)
        data_half.append(Yi_half)
        data_quart.append(Yi_quart)
        return data_full,data_half,data_quart

    # TODO Change to fit our set + create the training set here
    def __train_model(self):
        # Retreive training datasets
        end=str(datetime.datetime.utcnow()-datetime.timedelta(days=365))
        start=str(datetime.datetime.utcnow()-datetime.timedelta(days=365,minutes=361))
        self.train360a,self.train180a,self.train90a = self.__get_market_data(start,end)
        end=str(datetime.datetime.utcnow()-datetime.timedelta(days=65))
        start=str(datetime.datetime.utcnow()-datetime.timedelta(days=65,minutes=361))
        train360b,train180b,train90b = self.__get_market_data(start,end)

        self.train360a=pd.DataFrame(self.train360a)
        self.train180a=pd.DataFrame(self.train180a)
        self.train90a=pd.DataFrame(self.train90a)
        train360b=pd.DataFrame(train360b)
        train180b=pd.DataFrame(train180b)
        train90b=pd.DataFrame(train90b)

        # Perform the Bayesian Regression to predict the average price change for each dataset of train2 using train1 as input. 
        # These will be used to estimate the coefficients (w0, w1, w2, and w3) in equation 8.
        weight = 2  # This constant was not specified in the paper, but we will use 2.
        trainDeltaP90 = np.empty(0)
        trainDeltaP180 = np.empty(0)
        trainDeltaP360 = np.empty(0)
        
        
        for i in range(0,len(self.train90a.index)) :
            trainDeltaP90 = np.append(trainDeltaP90, computeDelta(weight,train90b.iloc[i],self.train90a))
        for i in range(0,len(self.train180a.index)) :
            trainDeltaP180 = np.append(trainDeltaP180, computeDelta(weight,train180b.iloc[i],self.train180a))
        for i in range(0,len(self.train360a.index)) :
            trainDeltaP360 = np.append(trainDeltaP360, computeDelta(weight,train360b.iloc[i],self.train360a))


        # Actual deltaP values for the train2 data.
        trainDeltaP = np.asarray(train360b[len(train360b)-1])
        trainDeltaP = np.reshape(trainDeltaP, -1)


        # Combine all the training data
        d = {'deltaP': trainDeltaP,
            'deltaP90': trainDeltaP90,
            'deltaP180': trainDeltaP180,
            'deltaP360': trainDeltaP360 }
        trainData = pd.DataFrame(d)


        # Feed the data: [deltaP, deltaP90, deltaP180, deltaP360] to train the linear model. 
        # Use the statsmodels ols function.
        # Use the variable name model for your fitted model
        # YOUR CODE HERE
        self.model = smf.ols(formula = 'deltaP ~ deltaP90 + deltaP180 + deltaP360', data = trainData).fit()

    def __predict(self,test_360,test_180,test_90):
        if self.model==None:
            self.__train_model()
        test_360=pd.DataFrame(test_360)
        test_180=pd.DataFrame(test_180)
        test_90=pd.DataFrame(test_90)

        weight = 2 
        testDeltaP90 = testDeltaP180 = testDeltaP360 = np.empty(0)
        for i in range(0,len(self.train90a.index)) :
            testDeltaP90 = np.append(testDeltaP90, computeDelta(weight,test_90.iloc[i],self.train90a))
        for i in range(0,len(self.train180a.index)) :
            testDeltaP180 = np.append(testDeltaP180, computeDelta(weight,test_180.iloc[i],self.train180a))
        for i in range(0,len(self.train360a.index)) :
            testDeltaP360 = np.append(testDeltaP360, computeDelta(weight,test_360.iloc[i],self.train360a))

        # Actual deltaP values for test data.
        # YOUR CODE HERE (use the right variable names so the below code works)
        testDeltaP = np.asarray(test_360[len(test_360)-1])
        testDeltaP = np.reshape(testDeltaP, -1)


        # Combine all the test data
        d = {'deltaP': testDeltaP,
            'deltaP90': testDeltaP90,
            'deltaP180': testDeltaP180,
            'deltaP360': testDeltaP360}
        testData = pd.DataFrame(d)
        result = self.model.predict(testData)
        return result


    def start(self):
        # Predict current
        start=str(datetime.datetime.utcnow()-datetime.timedelta(minutes=361))
        end=str(datetime.datetime.utcnow())
        data360,data180,data90 = self.__get_market_data(start,end)
        #print(data360)
        #print(len(data360))
        #print(data90)
        prediction = self.__predict(data360,data180,data90)
        print(prediction)
        
        
def main(argv):
    current = TradingBot()
    current.start()
    pass


if __name__ == '__main__':
    main(sys.argv)