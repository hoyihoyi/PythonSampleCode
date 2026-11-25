// FxCarryRegime cBot - research rig sleeve C.
// Attach to a DAILY chart of an FX pair (one instance per pair).
// Direction = the side with the better overnight swap, as published by the broker
// (Symbol.SwapLong / Symbol.SwapShort) -> this is the real, after-markup carry
// you will receive. Trades only if that swap is positive (you get paid to hold).
// Risk-off regime switch from equities (default US500 daily):
//     risk-off if US500 < SMA(200) OR US500 20d vol is in the top 20% of 3y history
// -> carry positions are closed in risk-off (carry crashes cluster in equity stress).
// Positions are vol-targeted like the backtest.
using System;
using System.Linq;
using cAlgo.API;
using cAlgo.API.Internals;

namespace TradingRig
{
    [Robot(AccessRights = AccessRights.None, TimeZone = TimeZones.UTC)]
    public class FxCarryRegime : Robot
    {
        [Parameter("Target vol % (full position)", DefaultValue = 10, MinValue = 1, MaxValue = 50)]
        public double TargetVolPct { get; set; }

        [Parameter("Risk regime symbol", DefaultValue = "US500")]
        public string RegimeSymbol { get; set; }

        [Parameter("Min positive swap (pips/day equivalent)", DefaultValue = 0.0)]
        public double MinSwap { get; set; }

        [Parameter("Rebalance buffer (fraction)", DefaultValue = 0.25)]
        public double RebalanceBuffer { get; set; }

        [Parameter("Max drawdown kill-switch %", DefaultValue = 20)]
        public double KillSwitchDdPct { get; set; }

        [Parameter("Label", DefaultValue = "FXC")]
        public string BotLabel { get; set; }

        private Bars _risk;
        private double _peakEquity;
        private bool _killed;

        protected override void OnStart()
        {
            _risk = MarketData.GetBars(TimeFrame.Daily, RegimeSymbol);
            while (_risk.Count < 1000 && _risk.LoadMoreHistory() > 0) { }
            _peakEquity = Account.Equity;
            Print("SwapLong={0} SwapShort={1} type={2}", Symbol.SwapLong, Symbol.SwapShort, Symbol.SwapCalculationType);
        }

        protected override void OnBar()
        {
            if (Bars.Count < 60 || _risk.Count < 800 || _killed) return;

            _peakEquity = Math.Max(_peakEquity, Account.Equity);
            if (Account.Equity < _peakEquity * (1 - KillSwitchDdPct / 100.0))
            {
                foreach (var p in Positions.FindAll(BotLabel, SymbolName)) ClosePosition(p);
                _killed = true;
                Print("Kill-switch hit. Review before restarting.");
                return;
            }

            int dir = 0;
            if (Symbol.SwapLong > MinSwap && Symbol.SwapLong >= Symbol.SwapShort) dir = 1;
            else if (Symbol.SwapShort > MinSwap) dir = -1;

            if (RiskOff()) dir = 0;

            double target = dir * VolTargetUnits();
            Rebalance(target);
        }

        private bool RiskOff()
        {
            var c = _risk.ClosePrices;
            double sma = 0;
            for (int i = 1; i <= 200; i++) sma += c.Last(i);
            sma /= 200;
            if (c.Last(1) < sma) return true;
            int n = 750;
            double now = Vol(c, 1, 20);
            int below = 0;
            for (int k = 0; k < n; k++) if (Vol(c, 1 + k, 20) <= now) below++;
            return below / (double)n > 0.8;
        }

        private static double Vol(DataSeries c, int endShift, int len)
        {
            double s = 0, s2 = 0;
            for (int i = endShift; i < endShift + len; i++)
            {
                double r = c.Last(i) / c.Last(i + 1) - 1;
                s += r; s2 += r * r;
            }
            double m = s / len;
            return Math.Sqrt(Math.Max(s2 / len - m * m, 1e-12));
        }

        private double VolTargetUnits()
        {
            double dailyVol = Vol(Bars.ClosePrices, 1, 36);
            double ticks = dailyVol * Bars.ClosePrices.Last(1) / Symbol.TickSize;
            double valuePerUnit = ticks * Symbol.TickValue;
            double risk = Account.Equity * TargetVolPct / 100.0 / Math.Sqrt(252);
            return valuePerUnit > 0 ? risk / valuePerUnit : 0;
        }

        private void Rebalance(double targetUnits)
        {
            var mine = Positions.FindAll(BotLabel, SymbolName);
            double current = mine.Sum(p => p.TradeType == TradeType.Buy ? p.VolumeInUnits : -p.VolumeInUnits);
            double target = Symbol.NormalizeVolumeInUnits(Math.Abs(targetUnits), RoundingMode.Down) * Math.Sign(targetUnits);
            if (Math.Abs(target) < Symbol.VolumeInUnitsMin) target = 0;

            bool change = (target == 0) != (current == 0) || Math.Sign(target) != Math.Sign(current)
                          || (current != 0 && Math.Abs(target - current) > RebalanceBuffer * Math.Abs(current));
            if (!change) return;

            foreach (var p in mine) ClosePosition(p);
            if (target != 0)
            {
                var r = ExecuteMarketOrder(target > 0 ? TradeType.Buy : TradeType.Sell, SymbolName, Math.Abs(target), BotLabel);
                if (!r.IsSuccessful) Print("Order failed: {0}", r.Error);
            }
        }
    }
}
