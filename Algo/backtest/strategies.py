from backtesting import Strategy
from backtesting.lib import crossover, resample_apply, barssince, plot_heatmaps
import talib
import pandas as pd
import numpy as np
from helper_functions import chandelier_exit, slope, convexity, slope_diff, conv_diff, vwma, momentum_state


class TrailingStrategy(Strategy):
    """
    A strategy with automatic trailing stop-loss, trailing the current
    price at distance of some multiple of average true range (ATR). Call
    `TrailingStrategy.set_trailing_sl()` to set said multiple
    (`6` by default). See [tutorials] for usage examples.

    [tutorials]: index.html#tutorials

    Remember to call `super().init()` and `super().next()` in your
    overridden methods.
    """
    n_atr = 10
    __atr = None
    atr_period = 100
    stop_loss_short = 1.01
    stop_loss_long = 0.99

    def init(self):
        super().init()
        self.set_atr_periods()

    def set_atr_periods(self, periods: int = 100):
        """
        Set the lookback period for computing ATR. The default value
        of 100 ensures a _stable_ ATR.
        """
        h, l, c_prev = self.data.High, self.data.Low, pd.Series(self.data.Close).shift(1)
        tr = np.max([h - l, (c_prev - h).abs(), (c_prev - l).abs()], axis=0)
        atr = pd.Series(tr).rolling(periods).mean().bfill().values
        self.__atr = atr

    def set_trailing_sl(self, n_atr: float = 6):
        """
        Sets the future trailing stop-loss as some multiple (`n_atr`)
        average true bar ranges away from the current price.
        """
        self.n_atr = n_atr

    def next(self):
        super().next()
        # Can't use index=-1 because self.__atr is not an Indicator type
        index = len(self.data)-1
        for trade in self.trades:
            if trade.is_long:
                trade.sl = max([trade.sl or -np.inf,
                               self.data.Close[index] - self.__atr[index] * self.n_atr,
                                self.data.Close[index] * self.stop_loss_long])
            else:
                trade.sl = min(trade.sl or np.inf,
                               self.data.Close[index] + self.__atr[index] * self.n_atr,
                               self.data.Close[index] * self.stop_loss_short)


class BasicTrailingStrategy(Strategy):
    """
    A strategy with automatic trailing stop-loss, trailing the current
    price at distance stop_loss_short and stop_loss_long respectively

    Remember to call `super().init()` and `super().next()` in your
    overridden methods.
    """
    stop_loss_short = 1.015
    stop_loss_long = 0.985

    def init(self):
        super().init()

    def next(self):
        super().next()
        index = len(self.data)-1
        for trade in self.trades:
            if trade.is_long:
                trade.sl = max([trade.sl or -np.inf,
                                self.data.Close[index] * self.stop_loss_long])
            else:
                trade.sl = min(trade.sl or np.inf,
                               self.data.Close[index] * self.stop_loss_short)


class CHDLStrategy(Strategy):
    atr_length = 14
    multiplier = 3  # use higher value when volatility is higher
    high_length = 4
    low_length = 4
    size = 0.99

    def init(self):
        super().init()
        self.exit = self.I(
            chandelier_exit, self.data.df, self.atr_length, self.multiplier, self.high_length, self.low_length)

    def next(self):
        super().next()
        if self.exit == 1 and not self.position.is_long:
            if self.position.is_short:
                self.position.close()
            self.buy(size=self.size)
        elif self.exit == -1 and not self.position.is_short:
            if self.position.is_long:
                self.position.close()
            self.sell(size=self.size)


class BBStrategy(Strategy):
    # mean reversion strategy
    bband_length = 5
    bband_avg_type = 2  # 1 = ema, 0 = sma, 2 = wma
    size = 0.99
    bars_knockout = 5
    take_profit = 1.04

    stddev_long_entry_cutoff = 660  # 1117 df2
    stddev_short_entry_cutoff = 416  # 451 df2
    stddev_exit_cutoff = 417  # 602 df2
    stddev_regime_switch = 1800  # 1351 df2
    stddev_ema_delta_cutoff = 7
    stop_loss_short = 1.01
    stop_loss_long = 0.99

    def init(self):
        super().init()
        self.upper, self.middle, self.lower = self.I(
            talib.BBANDS, self.data.Close, timeperiod=self.bband_length, nbdevup=2.0, nbdevdn=2.0,
            matype=self.bband_avg_type)
        self.long_ema = self.I(talib.EMA, self.data.Close, timeperiod=150)

        self.stddev_short = self.I(talib.STDDEV, self.data.Close, timeperiod=5)
        self.stddev = self.I(talib.STDDEV, self.data.Close, timeperiod=20)
        self.stddev_long = self.I(talib.STDDEV, self.data.Close, timeperiod=100)
        self.stddev_very_long = self.I(talib.STDDEV, self.data.Close, timeperiod=300)

        self.stddev_ema = self.I(talib.EMA, self.stddev, timeperiod=20)
        self.stddev_ema_long = self.I(talib.EMA, self.stddev_long, timeperiod=20)

        self.volume_avg = self.I(talib.SMA, self.data.Volume.astype(float), timeperiod=40)
        # self.short_angle = self.I(talib.LINEARREG_ANGLE, 0.5*(self.data.Close + self.data.Open) / 50, timeperiod=14)
        # self.long_angle = self.I(talib.LINEARREG_ANGLE, 0.5*(self.data.Close + self.data.Open) / 50, timeperiod=24)

    def update_regime(self, price, stddev, direction='buy'):
        self.entry_price = price
        self.entry_stddev = stddev

    def next(self):
        super().next()
        price = self.data.Close[-1]
        volume = self.data.Volume[-1]
        while self.stddev[-1] > self.stddev_regime_switch:
            # crash regime -> no mean reversion, trade momentum
            if self.position:
                self.position.close()
            return
        if not self.position:
            if self.stddev_long > self.stddev_long_entry_cutoff and self.stddev_ema[-2] - self.stddev_ema[-1] > self.stddev_ema_delta_cutoff:
                # we are in a high volatility regime and the stddev moving average is receding -> time to get in
                if price < self.long_ema[-1]:
                    # price is low, volatility high -> is gonna go back up
                    self.buy(size=self.size, tp=self.take_profit*price, sl=price*self.stop_loss_long)
            elif self.stddev > self.stddev_short_entry_cutoff and self.stddev_ema[-2] - self.stddev_ema[-1] > self.stddev_ema_delta_cutoff:
                if price > self.long_ema[-1]:
                    self.sell(size=self.size, tp=(2-self.take_profit) * price, sl=price*self.stop_loss_short)
                    pass

        elif self.position and (crossover(self.stddev_long, self.stddev_ema)):
            entry_price = [t.entry_price for t in self.trades][0]
            if self.position.is_long and price > entry_price:
                self.position.close()
            elif self.position.is_short and price < entry_price:
                self.position.close()
        """price = self.data.Close[-1]
        volume = self.data.Volume[-1]
        if self.position:
            if self.position.is_long and (crossover(self.long_ema, self.lower)):
                self.position.close()
            elif self.position.is_short and (crossover(self.upper, self.long_ema)):
                self.position.close()

        else:
            if ((crossover(self.data.Close, self.upper) and self.middle[-1] > self.middle[-2])

            ):
                self.buy(size=self.size, tp=self.take_profit*price)
            elif crossover(self.lower, self.data.Close) and self.middle[-1] < self.middle[-2]:
                self.sell(size=self.size, tp=(2 - self.take_profit) * price)"""


class EMASlopeStrategy(TrailingStrategy):
    """
    Idea: compare short running EMA slope vs long running EMA slope
    Condition (Entry Long): ShortEMA > LongEMA and ShortSlope > LongSlope and SecondDeriv(Short) > 0
    Condition (Exit Long): ShortSlope < 0
    Condition (Entry Short): ShortEMA < LongEMA and ShortSlope < LongSlope and SecondDeriv(Short) < 0
    Condition (Exit Short): ShortSlope > 0
    """
    size = 0.99

    long_length = 120
    short_length = 30  # convexity is less local than simple ema -> smooth it out more!
    very_short_length = 20

    slope_smoothing_length = 5
    convexity_smoothing_length = 5
    slope_accuracy = 3
    convexity_accuracy = 3

    exit_slope_cutoff = 1
    relative_slope_diff_cutoff = 0.01
    absolute_slope_diff_cutoff = 5
    absolute_ema_diff_cutoff = 5

    def init(self):
        super().init()
        # short stuff
        self.short = self.I(talib.EMA, self.data.Close, timeperiod=self.short_length)
        self.raw_slope_short = slope(self.short, accuracy=self.slope_accuracy)
        self.slope_short = self.I(talib.EMA, self.raw_slope_short, timeperiod=self.slope_smoothing_length)

        self.raw_convexity_short = slope(self.slope_short, accuracy=self.convexity_accuracy)
        self.convexity_short = talib.EMA(self.raw_convexity_short, timeperiod=self.convexity_smoothing_length)

        # long stuff
        self.long = self.I(talib.EMA, self.data.Close, timeperiod=self.long_length)
        self.raw_slope_long = slope(self.long, accuracy=self.slope_accuracy)
        self.slope_long = self.I(talib.EMA, self.raw_slope_long, timeperiod=self.slope_smoothing_length)

        self.raw_convexity_long = slope(self.slope_long, accuracy=self.convexity_accuracy)
        self.convexity_long = talib.EMA(self.raw_convexity_long, timeperiod=self.convexity_smoothing_length)

        self.very_short = self.I(talib.EMA, self.data.Close, timeperiod=self.very_short_length)
        self.raw_slope_vs = slope(self.very_short, accuracy=self.slope_accuracy)
        self.slope_very_short = self.I(talib.EMA, self.raw_slope_vs, timeperiod=self.slope_smoothing_length)

        self.relative_slope = self.I(slope_diff, self.slope_short, self.slope_long)
        # self.relative_convexity = self.I(conv_diff, self.convexity_short, self.convexity_long)

    def next(self):
        super().next()
        price = self.data.Close[-1]
        volume = self.data.Volume[-1]

        short_ema = self.short[-1]
        short_slope = self.slope_short[-1]
        # short_convexity = self.convexity_short[-1]
        long_ema = self.long[-1]
        long_slope = self.slope_long[-1]
        # long_convexity = self.convexity_long[-1]

        if not self.position:
            if (short_ema > long_ema + self.absolute_ema_diff_cutoff and short_slope > long_slope and
                    (short_slope - long_slope) / short_slope > self.relative_slope_diff_cutoff
                and self.relative_slope[-1] > self.absolute_slope_diff_cutoff):
                self.buy(size=self.size, sl=self.stop_loss_long*price)
            elif (short_ema < long_ema - self.absolute_ema_diff_cutoff and short_slope < long_slope and
                  (long_slope - short_slope) / short_slope > self.relative_slope_diff_cutoff and
                  self.relative_slope[-1] < self.absolute_slope_diff_cutoff):
                self.sell(size=self.size, sl=self.stop_loss_short*price)
        else:
            if ((self.position.is_long and short_slope < self.exit_slope_cutoff) or
                (self.position.is_short and short_slope > - self.exit_slope_cutoff)):
                self.position.close()


class MACDStrategy(TrailingStrategy):
    volume_range = 18
    macd_long_range = 60
    macd_short_range = 7
    macd_signal_range = 32
    size = 0.99
    long_ema = 200
    short_ema = 30
    overbought = 0
    oversold = 0

    def init(self):
        super().init()
        self.macd, self.macds, _ = self.I(
            talib.MACD, self.data.Close, fastperiod=self.macd_short_range, slowperiod=self.macd_long_range,
            signalperiod=self.macd_signal_range)  # entry signals
        self.avg_volume = self.I(talib.SMA, self.data.Volume.astype(float), timeperiod=self.volume_range)
        self.long_running_avg = self.I(talib.EMA, self.data.Close, timeperiod=self.long_ema)
        self.short_running_avg = self.I(talib.EMA, self.data.Close, timeperiod=self.short_ema)

    def next(self):
        super().next()
        volume = self.data.Volume[-1]  # most recent volume
        price = self.data.Close[-1]
        if not self.position and volume > self.avg_volume[-1]:
            if crossover(self.macd, self.macds) and self.macd > self.oversold and price > self.long_running_avg[-1]:
                self.buy(size=self.size, sl=self.stop_loss_long*price)
            elif crossover(self.macds, self.macd) and self.macd < self.overbought and price < self.long_running_avg[-1]:
                self.sell(size=self.size, sl=self.stop_loss_short*price)


class MACDPullbackStrategy(BasicTrailingStrategy):
    volume_range = 18
    macd_long_range = 26
    macd_short_range = 12
    macd_signal_range = 9
    size = 0.05
    momentum_threshold = 250
    momentum_bars = 3
    pullback_bars = 2

    def init(self):
        super().init()
        self.macd, self.macds, _ = self.I(
            talib.MACD, self.data.Close, fastperiod=self.macd_short_range,
            slowperiod=self.macd_long_range,
            signalperiod=self.macd_signal_range)  # entry signals
        self.avg_volume = self.I(talib.SMA, self.data.Volume.astype(float),
                                 timeperiod=self.volume_range)
        self.momentum = self.I(talib.MOM, self.data.Close, timeperiod=10)
        self.momentum_state = self.I(momentum_state, self.momentum, self.momentum_threshold)

    def next(self):
        super().next()
        volume = self.data.Volume[-1]  # most recent volume
        price = self.data.Close[-1]
        # self.momentum_state = momentum_state(self.momentum, self.momentum_threshold, self.momentum_bars)
        if not self.position and volume > self.avg_volume[-1]:
            if (barssince(self.momentum_state != 1.0) > self.momentum_bars and  # we are in an upward momentum state
                    barssince(self.data.Close > self.data.Open) >= self.pullback_bars):  # we are in a pullback
                self.buy(size=self.size, sl=self.stop_loss_long * price)
        elif self.position:
            if (crossover(self.macd, self.macds) or crossover(self.macds, self.macd)) or volume < self.avg_volume:
                self.position.close()
            elif self.momentum_state != 1.0:
                self.position.close()
