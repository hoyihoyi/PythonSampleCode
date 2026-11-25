// IndexRegimeSwing cBot  - research rig sleeves A (pullback), B (turn-of-month), D (regime long)
// Attach to a DAILY chart of an index (US500, NAS100, US30, GER40, ...).
// One instance per index. Positions are sized so a full position carries
// `TargetVolPct` annualised volatility on account equity (vol targeting),
// using the broker's own tick value -> works for spread-bet (£/pt) accounts.
//
// Logic (all evaluated on the CLOSED daily bar; orders sent at the next bar open,
// exactly like the Python backtest):
//   Regime   bull  = close > SMA(200)   and 20d vol not in the top 20% of its 3y history
//   Sleeve D regime long : long `RegimeWeight` while bull regime holds
//   Sleeve A pullback    : +`PullbackWeight` when bull & RSI(2) < 10; exit when close > SMA(5)
//   Sleeve B turn-of-month: +`TomWeight` from last trading day of month to 3rd trading day
// Target position = sum of active sleeve weights x vol-targeted size. Trades only
// when the target differs from the current holding by more than `RebalanceBuffer`.
using System;
using System.Linq;
using cAlgo.API;
using cAlgo.API.Indicators;

namespace TradingRig
{
    [Robot(AccessRights = AccessRights.None, TimeZone = TimeZones.UTC)]
    public class IndexRegimeSwing : Robot
    {
        [Parameter("Target vol % (full position)", DefaultValue = 15, MinValue = 1, MaxValue = 60)]
        public double TargetVolPct { get; set; }

        [Parameter("Regime long weight (D)", DefaultValue = 0.34)]
        public double RegimeWeight { get; set; }

        [Parameter("Pullback weight (A)", DefaultValue = 0.33)]
        public double PullbackWeight { get; set; }

        [Parameter("Turn-of-month weight (B)", DefaultValue = 0.33)]
        public double TomWeight { get; set; }

        [Parameter("RSI(2) entry level", DefaultValue = 10)]
        public double RsiEntry { get; set; }

        [Parameter("Rebalance buffer (fraction)", DefaultValue = 0.25)]
        public double RebalanceBuffer { get; set; }

        [Parameter("Max drawdown kill-switch %", DefaultValue = 25)]
        public double KillSwitchDdPct { get; set; }

        [Parameter("Label", DefaultValue = "IRS")]
        public string BotLabel { get; set; }

        private RelativeStrengthIndex _rsi;
        private SimpleMovingAverage _sma200, _sma5;
        private bool _inPullback;
        private double _peakEquity;
        private bool _killed;

        protected override void OnStart()
        {
            if (TimeFrame != TimeFrame.Daily)
                Print("WARNING: designed for the Daily timeframe; current = {0}", TimeFrame);
            _rsi = Indicators.RelativeStrengthIndex(Bars.ClosePrices, 2);
            _sma200 = Indicators.SimpleMovingAverage(Bars.ClosePrices, 200);
            _sma5 = Indicators.SimpleMovingAverage(Bars.ClosePrices, 5);
            _peakEquity = Account.Equity;
            _inPullback = false;
        }

        protected override void OnBar()
        {
            if (Bars.Count < 800 || _killed) return;

            // ---- risk kill-switch (portfolio-level protection) ----
            _peakEquity = Math.Max(_peakEquity, Account.Equity);
            if (Account.Equity < _peakEquity * (1 - KillSwitchDdPct / 100.0))
            {
                foreach (var p in Positions.FindAll(BotLabel, SymbolName)) ClosePosition(p);
                _killed = true;
                Print("Kill-switch: drawdown > {0}% from peak. Bot stopped trading; review before restarting.", KillSwitchDdPct);
                return;
            }

            double close = Bars.ClosePrices.Last(1);

            // ---- regime ----
            bool trendUp = close > _sma200.Result.Last(1);
            bool calm = Vol20Percentile() <= 0.8;
            bool bull = trendUp && calm;

            // ---- sleeve A: pullback state machine ----
            if (!_inPullback && trendUp && _rsi.Result.Last(1) < RsiEntry) _inPullback = true;
            else if (_inPullback && close > _sma5.Result.Last(1)) _inPullback = false;

            // ---- sleeve B: turn of month (today = Bars.OpenTimes.Last(0)) ----
            bool tom = InTurnOfMonth(Bars.OpenTimes.Last(0));

            double weight = (bull ? RegimeWeight : 0) + (_inPullback ? PullbackWeight : 0) + (tom ? TomWeight : 0);
            double targetUnits = weight * VolTargetUnits();
            Rebalance(targetUnits);
        }

        // 20-day realised vol percentile vs the last 750 days
        private double Vol20Percentile()
        {
            int n = 750;
            var vols = new double[n];
            for (int k = 0; k < n; k++) vols[k] = RealisedVol(1 + k, 20);
            double now = vols[0];
            return vols.Count(v => v <= now) / (double)n;
        }

        private double RealisedVol(int endShift, int len)
        {
            double sum = 0, sum2 = 0;
            for (int i = endShift; i < endShift + len; i++)
            {
                double r = Bars.ClosePrices.Last(i) / Bars.ClosePrices.Last(i + 1) - 1;
                sum += r; sum2 += r * r;
            }
            double m = sum / len;
            return Math.Sqrt(Math.Max(sum2 / len - m * m, 1e-12));
        }

        // units such that a full position has TargetVolPct annualised vol on equity
        private double VolTargetUnits()
        {
            double dailyVol = RealisedVol(1, 36);                       // fraction
            double priceMovePerDay = dailyVol * Bars.ClosePrices.Last(1); // price units
            double ticksPerDay = priceMovePerDay / Symbol.TickSize;
            double valuePerUnitPerDay = ticksPerDay * Symbol.TickValue;  // account ccy per unit volume
            double targetDailyRisk = Account.Equity * TargetVolPct / 100.0 / Math.Sqrt(252);
            return valuePerUnitPerDay > 0 ? targetDailyRisk / valuePerUnitPerDay : 0;
        }

        // Last trading day of month (approx. last weekday) through 3rd trading day of next month
        private static bool InTurnOfMonth(DateTime d)
        {
            if (d.Day <= 5)
            {
                int tradingDaysSoFar = 0;
                for (var x = new DateTime(d.Year, d.Month, 1); x <= d.Date; x = x.AddDays(1))
                    if (x.DayOfWeek != DayOfWeek.Saturday && x.DayOfWeek != DayOfWeek.Sunday) tradingDaysSoFar++;
                if (tradingDaysSoFar <= 3) return true;
            }
            var next = d.Date.AddDays(1);
            while (next.DayOfWeek == DayOfWeek.Saturday || next.DayOfWeek == DayOfWeek.Sunday) next = next.AddDays(1);
            return next.Month != d.Month; // today is the last weekday of the month
        }

        private void Rebalance(double targetUnits)
        {
            var mine = Positions.FindAll(BotLabel, SymbolName);
            double current = mine.Sum(p => p.TradeType == TradeType.Buy ? p.VolumeInUnits : -p.VolumeInUnits);
            double target = Symbol.NormalizeVolumeInUnits(Math.Abs(targetUnits), RoundingMode.Down) * Math.Sign(targetUnits);
            if (Math.Abs(target) < Symbol.VolumeInUnitsMin) target = 0;

            bool flip = Math.Sign(target) != Math.Sign(current);
            bool drift = current != 0 && Math.Abs(target - current) > RebalanceBuffer * Math.Abs(current);
            if (!(target == 0 && current != 0) && !(current == 0 && target != 0) && !flip && !drift) return;

            if (target == 0 || flip)
            {
                foreach (var p in mine) ClosePosition(p);
                current = 0;
            }
            double delta = target - current;
            if (Math.Abs(delta) < Symbol.VolumeInUnitsMin) return;
            if (Math.Sign(delta) == Math.Sign(current) || current == 0)
            {
                var res = ExecuteMarketOrder(delta > 0 ? TradeType.Buy : TradeType.Sell, SymbolName,
                    Symbol.NormalizeVolumeInUnits(Math.Abs(delta)), BotLabel);
                if (!res.IsSuccessful) Print("Order failed: {0}", res.Error);
            }
            else
            {
                // reduce: close part of the largest position
                double reduce = Math.Abs(delta);
                foreach (var p in mine.OrderByDescending(p => p.VolumeInUnits))
                {
                    if (reduce <= 0) break;
                    double cut = Math.Min(reduce, p.VolumeInUnits);
                    cut = Symbol.NormalizeVolumeInUnits(cut, RoundingMode.Down);
                    if (cut >= p.VolumeInUnits) ClosePosition(p); else if (cut > 0) ClosePosition(p, cut);
                    reduce -= cut;
                }
            }
        }
    }
}
