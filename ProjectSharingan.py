import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time


pd.set_option('display.width', 400)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


class GetKlinesDataOfMultipleCoins:
    def __init__(self, symbol_names, interval):
        self.symbol_names = symbol_names
        self.interval = interval
        self.coins_data_df = []

        self.get_candlestick_data_from_binance()
        self.calculate_percent_change_and_log_return()

    def get_candlestick_data_from_binance(self):

        for i in range(len(self.symbol_names)):
            url = requests.get("https://api.binance.com/api/v3/klines?symbol=" + self.symbol_names[i] + "&interval=" + self.interval + "&limit=1000").json()
            df = pd.DataFrame(url)

            df = df.rename(columns={0: self.symbol_names[i]+": Open time", 1: self.symbol_names[i]+": Open",
                                    2: self.symbol_names[i]+": High", 3: self.symbol_names[i]+": Low",
                                    4: self.symbol_names[i]+": Close", 5: self.symbol_names[i]+": Volume",
                                    6: self.symbol_names[i]+": Close time", 7: self.symbol_names[i]+": Quote asset volume",
                                    8: self.symbol_names[i]+": Number of trades", 9: self.symbol_names[i]+": Taker buy base asset volume",
                                    10: self.symbol_names[i]+": Taker buy quote asset volume", 11: self.symbol_names[i]+": Ignore"})

            df = df.astype(float)
            df[self.symbol_names[i] + ": Open time"] = pd.to_datetime(df[self.symbol_names[i] + ": Open time"], unit="ms")
            df[self.symbol_names[i] + ": Close time"] = pd.to_datetime(df[self.symbol_names[i] + ": Close time"], unit="ms")
            self.coins_data_df.append(df)

    def calculate_percent_change_and_log_return(self):

        for i in range(len(self.symbol_names)):
            self.coins_data_df[i][self.symbol_names[i] + ": Percent Change"] = self.coins_data_df[i][self.symbol_names[i]+": Close"]/self.coins_data_df[i][self.symbol_names[i]+": Open"]
            self.coins_data_df[i][self.symbol_names[i] + ": Log Returns"] = np.log(self.coins_data_df[i][self.symbol_names[i] + ": Percent Change"])

    def print_all_data_heads(self):
        print("----------")
        print("print_all_data_heads:")
        for i in range(len(self.coins_data_df)):
            print(self.coins_data_df[i].head(5), "\n")


class CalculateATR:
    def __init__(self, data, atr_rolling_window):
        self.TrueRange = pd.DataFrame()
        self.AverageTrueRange = pd.DataFrame()
        self.ATR_rolling_window = atr_rolling_window

        for i in range(len(data.symbol_names)):
            high_low = data.coins_data_df[i][data.symbol_names[i] + ": High"] - data.coins_data_df[i][
                data.symbol_names[i] + ": Low"]
            high_cp = abs(data.coins_data_df[i][data.symbol_names[i] + ": High"] - data.coins_data_df[i][
                data.symbol_names[i] + ": Close"].shift(1))
            low_cp = abs(data.coins_data_df[i][data.symbol_names[i] + ": Low"] - data.coins_data_df[i][
                data.symbol_names[i] + ": Close"].shift(1))

            df = pd.concat([high_low, high_cp, low_cp], axis=1)
            self.TrueRange = pd.concat([self.TrueRange, pd.Series(np.max(df, axis=1))], axis=1)
            self.TrueRange.rename(columns={0: data.symbol_names[i] + ": TR"}, inplace=True)

        self.AverageTrueRange = pd.concat(
            [self.AverageTrueRange, self.TrueRange.rolling(self.ATR_rolling_window).mean()], axis=1)
        for i in range(len(data.symbol_names)):
            self.AverageTrueRange.rename(columns={data.symbol_names[i] + ": TR": data.symbol_names[i] + ": ATR"},
                                         inplace=True)

    def print_average_true_range(self):
        print("----------")
        print("print_AverageTrueRange:")
        print(self.AverageTrueRange.head(self.ATR_rolling_window + 5))


class CalculatePortfolioReturn:
    def __init__(self, data, portfolio_weights):
        self.portfolio_weights = portfolio_weights
        self.assets_log_returns_df = pd.DataFrame
        self.assets_log_returns_weighted_df = pd.DataFrame
        self.portfolio_log_returns = pd.DataFrame
        self.log_return_of_portfolio = float
        self.volatility = float
        self.interval = data.interval
        self.annual_factor = 0

        if self.interval == "12h":
            self.annual_factor = 365*2
        elif self.interval == "1d":
            self.annual_factor = 365
        elif self.interval == "1w":
            self.annual_factor = 52
        elif self.interval == "1M":
            self.annual_factor = 12

        tmp = []
        tmp2 = []
        for i in range(len(data.symbol_names)):
            tmp.append(data.coins_data_df[i][data.symbol_names[i] + ": Log Returns"])

            tmp2.append(data.coins_data_df[i][data.symbol_names[i] + ": Log Returns"]*self.portfolio_weights[i])
            tmp2[-1].rename(data.symbol_names[i] + ": Log Returns - Weighted", inplace=True)

        self.assets_log_returns_df = pd.concat(tmp, axis=1)
        self.assets_log_returns_weighted_df = pd.concat(tmp2, axis=1)

        self.portfolio_log_returns = pd.DataFrame({"Portfolio Log Returns": np.sum(self.assets_log_returns_weighted_df, axis=1)})

        self.log_return_of_portfolio = float(np.sum(self.assets_log_returns_weighted_df.mean()))*self.annual_factor

        self.volatility = np.sqrt(np.dot(self.portfolio_weights.T, np.dot(self.assets_log_returns_df.cov()*self.annual_factor, self.portfolio_weights)))


class MakeMontecarloSimulation:

    def __init__(self, data, iterations):
        self.symbol_names = data.symbol_names

        n = iterations

        self.weights = np.zeros((n, len(data.symbol_names)))
        self.exp_rtns = np.zeros(n)
        self.exp_vols = np.zeros(n)
        self.sharpe_ratios = np.zeros(n)
        self.best_portfolio = np.array([])

        for i in range(n):
            weight = np.random.random(len(data.symbol_names))
            weight /= weight.sum()
            self.weights[i] = weight

            portfolio_mc = CalculatePortfolioReturn(data, weight)

            self.exp_rtns[i] = portfolio_mc.log_return_of_portfolio
            self.exp_vols[i] = portfolio_mc.volatility
            self.sharpe_ratios[i] = self.exp_rtns[i] / self.exp_vols[i]

        self.best_portfolio = self.weights[self.sharpe_ratios.argmax()]

    def visualize_montecarlo_portfolios(self):
        fig, ax = plt.subplots()
        ax.scatter(self.exp_vols, self.exp_rtns,
                   c=self.sharpe_ratios)
        ax.scatter(self.exp_vols[self.sharpe_ratios.argmax()],
                   self.exp_rtns[self.sharpe_ratios.argmax()], c="r")
        ax.set_xlabel("Expected Volatility")
        ax.set_ylabel("Expected Return")
        plt.show()

    def print_best_portfolio_params(self):
        print("Portfolio:", self.symbol_names)
        print("Sharp Ratio of Portfolio:", self.sharpe_ratios[self.sharpe_ratios.argmax()])
        print("Volatility of Portfolio:", self.exp_vols[self.sharpe_ratios.argmax()])
        print("Log Return of Portfolio:", self.exp_rtns[self.sharpe_ratios.argmax()])
        print("Weights of Portfolio:", self.weights[self.sharpe_ratios.argmax()])


class CalculateLinearRegressionBetweenCoins:
    def __init__(self, data):
        self.assets_log_returns_df = pd.DataFrame

        tmp = []
        for i in range(len(data.symbol_names)):
            tmp.append(data.coins_data_df[i][data.symbol_names[i] + ": Log Returns"])
        self.assets_log_returns_df = pd.concat(tmp, axis=1)

        for i in range(len(data.symbol_names)):
            for n in range(len(data.symbol_names)):
                if n > i:
                    X = self.assets_log_returns_df[data.symbol_names[i] + ": Log Returns"].to_numpy().reshape(-1, 1)
                    Y = self.assets_log_returns_df[data.symbol_names[n] + ": Log Returns"].to_numpy().reshape(-1, 1)

                    lin_regr = LinearRegression()
                    lin_regr.fit(X, Y)

                    Y_pred = lin_regr.predict(X)

                    alpha = lin_regr.intercept_[0]
                    beta = lin_regr.coef_[0, 0]

                    fig, ax = plt.subplots()
                    ax.set_title("Alpha: " + str(round(alpha, 5)) + ", Beta: " + str(round(beta, 3)))
                    ax.set_xlabel(data.symbol_names[i])
                    ax.set_ylabel(data.symbol_names[n])
                    ax.scatter(X, Y)
                    ax.plot(X, Y_pred, c="r")
                    plt.show()


class CalculateBetas:
    def __init__(self, data, benchmark_coin):
        self.benchmark_coin = benchmark_coin
        self.assets_log_returns_df = pd.DataFrame
        self.beta_values = pd.DataFrame

        tmp = []
        for i in range(len(data.symbol_names)):
            tmp.append(data.coins_data_df[i][data.symbol_names[i] + ": Log Returns"])
        self.assets_log_returns_df = pd.concat(tmp, axis=1)

        cov = self.assets_log_returns_df.cov()
        var = self.assets_log_returns_df[self.benchmark_coin + ": Log Returns"].var()
        self.beta_values = cov.loc[self.benchmark_coin + ": Log Returns"]/var

        self.beta_values = self.beta_values.rename(self.benchmark_coin + " as Benchmark to calculate Beta")
        for i in range(len(data.symbol_names)):
            self.beta_values = self.beta_values.rename(index={data.symbol_names[i] + ": Log Returns": data.symbol_names[i] + ": Beta"})

    def print_beta_values(self):
        print(self.beta_values)



class CalculateExpectedReturnsCAPM:
    def __init__(self, data, beta_values, benchmark_coin, risk_free_rate=0.0):
        self.expected_returns_CAPM = pd.DataFrame
        self.assets_log_returns = pd.DataFrame
        self.benchmark_coin = benchmark_coin
        self.market_return = 0
        self.risk_free_rate = risk_free_rate

        tmp = []
        for i in range(len(data.symbol_names)):
            tmp.append(data.coins_data_df[i][data.symbol_names[i] + ": Log Returns"])
            tmp[-1].rename(data.symbol_names[i] + ": Log Return", inplace=True)

        self.assets_log_returns = pd.concat(tmp, axis=1)
        self.assets_log_returns = pd.DataFrame(np.sum(self.assets_log_returns, axis=0)).T

        self.market_return = float(self.assets_log_returns[self.benchmark_coin + ": Log Return"])

        self.expected_returns_CAPM = self.risk_free_rate + beta_values * (self.market_return - self.risk_free_rate)

        for i in range(len(data.symbol_names)):
            self.expected_returns_CAPM = self.expected_returns_CAPM.rename(index={data.symbol_names[i] + ": Beta": data.symbol_names[i] + ": Expected Return"})
        self.expected_returns_CAPM = self.expected_returns_CAPM.rename(self.benchmark_coin + " as Benchmark to calculate Expected Return with CAPM")

    def print_expected_returns_capm(self):
        print(self.expected_returns_CAPM)


if __name__ == "__main__":
    start_time = time.time()
    market = "BTCUSDT"  # Coin which represents the Market as a Benchmark
    riskFreeRate = 0.0167
    coins_data = GetKlinesDataOfMultipleCoins(["BTCUSDT", "ETHUSDT", "BNBUSDT", "IOTAUSDT"], "1d")

    montecarlo_simulation = MakeMontecarloSimulation(data=coins_data, iterations=5000)
    print("--- %s seconds ---" % (time.time() - start_time))
    montecarlo_simulation.print_best_portfolio_params()
    montecarlo_simulation.visualize_montecarlo_portfolios()

    optimal_portfolio = CalculatePortfolioReturn(data=coins_data, portfolio_weights=montecarlo_simulation.best_portfolio)

    coins_betas = CalculateBetas(data=coins_data, benchmark_coin=market)
    coins_betas.print_beta_values()
    CAPMs = CalculateExpectedReturnsCAPM(data=coins_data, beta_values=coins_betas.beta_values, benchmark_coin=market, risk_free_rate=riskFreeRate)
    CAPMs.print_expected_returns_capm()
