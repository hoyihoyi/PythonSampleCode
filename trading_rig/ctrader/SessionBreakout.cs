// SessionBreakout cBot - EXPERIMENTAL intraday research bot (DEMO / BACKTEST ONLY).
// The Python rig found a consistent GROSS edge in Asian-range breakouts on FX/gold
// but it did not survive conservative costs on 2 years of Yahoo hourly data.
// Use this bot in the cTrader backtester with Pepperstone TICK data and real
// spreads to settle the question on 10+ years of history before risking money.
//
// Rules (times in UTC): Asian range = high/low from 00:00 to AsiaEndHour.
// Between AsiaEndHour and EntryEndHour: stop-entry above/below the range
// (one trade per day). Stop = opposite side of the range. Flat at ExitHour.
// Risk per trade = RiskPct of equity at the stop.
using System;
using System.Linq;
using cAlgo.API;

namespace TradingRig
{
    [Robot(AccessRights = AccessRights.None, TimeZone = TimeZones.UTC)]
    public class SessionBreakout : Robot
    {
        [Parameter("Asia end hour (UTC)", DefaultValue = 7)] public int AsiaEndHour { get; set; }
        [Parameter("Entry end hour (UTC)", DefaultValue = 16)] public int EntryEndHour { get; set; }
        [Parameter("Exit hour (UTC)", DefaultValue = 20)] public int ExitHour { get; set; }
        [Parameter("Risk % per trade", DefaultValue = 0.5)] public double RiskPct { get; set; }
        [Parameter("Only after narrow Asian range", DefaultValue = false)] public bool NarrowOnly { get; set; }
        [Parameter("Label", DefaultValue = "SBO")] public string BotLabel { get; set; }

        private DateTime _day;
        private double _hi, _lo;
        private bool _placed;
        private readonly System.Collections.Generic.Queue<double> _ranges = new System.Collections.Generic.Queue<double>();

        protected override void OnStart()
        {
            if (TimeFrame != TimeFrame.Hour && TimeFrame != TimeFrame.Minute15 && TimeFrame != TimeFrame.Minute5)
                Print("Use an intraday chart (M5/M15/H1).");
        }

        protected override void OnBar()
        {
            var t = Bars.OpenTimes.Last(0);
            if (t.Date != _day)
            {
                _day = t.Date; _hi = double.MinValue; _lo = double.MaxValue; _placed = false;
            }
            var lastT = Bars.OpenTimes.Last(1);
            if (lastT.Date == _day && lastT.Hour < AsiaEndHour)
            {
                _hi = Math.Max(_hi, Bars.HighPrices.Last(1));
                _lo = Math.Min(_lo, Bars.LowPrices.Last(1));
            }

            if (t.Hour >= ExitHour)
            {
                foreach (var p in Positions.FindAll(BotLabel, SymbolName)) ClosePosition(p);
                foreach (var o in PendingOrders.Where(o => o.Label == BotLabel && o.SymbolName == SymbolName)) CancelPendingOrder(o);
                return;
            }
            if (t.Hour >= EntryEndHour)
            {
                foreach (var o in PendingOrders.Where(o => o.Label == BotLabel && o.SymbolName == SymbolName)) CancelPendingOrder(o);
                return;
            }
            if (!_placed && t.Hour >= AsiaEndHour && _hi > _lo && _hi != double.MinValue)
            {
                double range = _hi - _lo;
                bool narrowOk = true;
                if (_ranges.Count >= 10)
                {
                    var sorted = _ranges.OrderBy(x => x).ToArray();
                    narrowOk = range < sorted[sorted.Length / 2];
                }
                _ranges.Enqueue(range); if (_ranges.Count > 20) _ranges.Dequeue();
                _placed = true;
                if (NarrowOnly && !narrowOk) return;

                double stopPips = range / Symbol.PipSize + Symbol.Spread / Symbol.PipSize;
                double riskMoney = Account.Equity * RiskPct / 100.0;
                double units = riskMoney / (stopPips * Symbol.PipValue);
                units = Symbol.NormalizeVolumeInUnits(units, RoundingMode.Down);
                if (units < Symbol.VolumeInUnitsMin) return;
                var expiry = _day.AddHours(EntryEndHour);
                PlaceStopOrder(TradeType.Buy, SymbolName, units, _hi + Symbol.Spread, BotLabel, stopPips, null, expiry);
                PlaceStopOrder(TradeType.Sell, SymbolName, units, _lo, BotLabel, stopPips, null, expiry);
            }
        }

        // one-cancels-other: when one side fills, cancel the other
        protected override void OnTick()
        {
            if (Positions.FindAll(BotLabel, SymbolName).Any())
                foreach (var o in PendingOrders.Where(o => o.Label == BotLabel && o.SymbolName == SymbolName)) CancelPendingOrder(o);
        }
    }
}
