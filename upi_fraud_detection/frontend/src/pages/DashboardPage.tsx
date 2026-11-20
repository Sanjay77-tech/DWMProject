import React, { useEffect, useState } from 'react';
import { 
  ShieldCheck, 
  AlertTriangle, 
  Activity, 
  TrendingUp, 
  CreditCard, 
  ArrowUpRight, 
  ChevronRight
} from 'lucide-react';
import { 
  PieChart, 
  Pie, 
  Cell, 
  ResponsiveContainer, 
  Tooltip, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid 
} from 'recharts';
import { api } from '../api/client';
import { MetricCard } from '../components/MetricCard';

interface DashboardPageProps {
  onNavigateToTab: (tab: string) => void;
}

export const DashboardPage: React.FC<DashboardPageProps> = ({ onNavigateToTab }) => {
  const [data, setData] = useState<any | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    const fetchDashboard = async () => {
      setLoading(true);
      try {
        const stats = await api.getDashboardStats();
        if (mounted) {
          setData(stats);
          setError(null);
        }
      } catch (err: any) {
        if (mounted) {
          setError(err.message || "Failed to load dashboard data. Please make sure a dataset is ingested.");
        }
      } finally {
        if (mounted) setLoading(false);
      }
    };

    fetchDashboard();
    return () => { mounted = false; };
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[500px] space-y-4">
        <div className="w-10 h-10 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin" />
        <p className="text-sm text-slate-400 font-medium">Aggregating telemetry and fraud risk metrics...</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="bg-navy-800/80 border border-navy-700 rounded-2xl p-8 text-center max-w-lg mx-auto mt-12 space-y-4">
        <AlertTriangle className="w-10 h-10 text-amber-400 mx-auto" />
        <h3 className="text-base font-bold text-white">Dataset Ingestion Required</h3>
        <p className="text-xs text-slate-400">
          {error || "No active transaction dataset is currently loaded in memory."}
        </p>
        <button
          onClick={() => onNavigateToTab('upload')}
          className="px-4 py-2 bg-emerald-500 hover:bg-emerald-400 text-navy-950 font-bold text-xs rounded-xl transition-all"
        >
          Go to Dataset Upload
        </button>
      </div>
    );
  }

  const kpis = data.kpis;
  const classificationData = data.classification_breakdown || [];
  const riskDistribution = data.risk_distribution || [];
  const volumeByType = data.volume_by_type || [];
  const alertStream = data.alert_stream || [];

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Page Title & Status */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold text-emerald-400 uppercase tracking-wider mb-1">
            <Activity className="w-3.5 h-3.5" />
            <span>Real-Time Telemetry & Threat Surveillance</span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Executive Threat Dashboard</h1>
          <p className="text-slate-400 text-sm mt-1">
            Autonomous multi-model risk assessment across {kpis.total_transactions?.toLocaleString()} ingest records.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => onNavigateToTab('prediction')}
            className="px-3.5 py-2 bg-navy-800 hover:bg-navy-700 border border-navy-600 rounded-xl text-xs font-medium text-slate-200 transition-all flex items-center gap-1.5"
          >
            <span>Live Transaction Predictor</span>
            <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
          </button>
        </div>
      </div>

      {/* 6 Key Animated Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        <MetricCard
          label="Total Transactions"
          value={kpis.total_transactions}
          icon={<CreditCard className="w-4 h-4 text-emerald-400" />}
        />
        <MetricCard
          label="Legitimate Volume"
          value={kpis.legitimate_transactions}
          icon={<ShieldCheck className="w-4 h-4 text-emerald-400" />}
        />
        <MetricCard
          label="Suspicious Flags"
          value={kpis.suspicious_transactions}
          icon={<AlertTriangle className="w-4 h-4 text-amber-400" />}
        />
        <MetricCard
          label="Confirmed Frauds"
          value={kpis.potential_frauds}
          icon={<AlertTriangle className="w-4 h-4 text-red-400" />}
        />
        <MetricCard
          label="Fraud Rate"
          value={kpis.fraud_rate}
          isDecimal={true}
          suffix="%"
          icon={<TrendingUp className="w-4 h-4 text-red-400" />}
        />
        <MetricCard
          label="Avg Transaction"
          value={kpis.average_amount}
          isDecimal={true}
          prefix="₹"
          icon={<Activity className="w-4 h-4 text-emerald-400" />}
        />
      </div>

      {/* Interactive Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Animated Donut Chart: Transaction Classification */}
        <div className="lg:col-span-5 bg-navy-800/80 border border-navy-700 rounded-2xl p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-1">
              <h3 className="text-sm font-bold text-white tracking-tight flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-400" />
                Transaction Classification
              </h3>
              <span className="text-[10px] text-slate-400 font-mono">Animated Arc Donut</span>
            </div>
            <p className="text-xs text-slate-400 mb-4">
              Distribution between verified clean, behavioral anomalies, and fraudulent transactions.
            </p>

            {/* Donut Container with smoothly sweeping animation */}
            <div className="h-60 relative flex items-center justify-center">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0a0f1d',
                      borderColor: '#1e293b',
                      borderRadius: '0.75rem',
                      color: '#f8fafc',
                      fontSize: '12px',
                    }}
                    formatter={(val: any, name: any) => [`${Number(val).toLocaleString()} txns`, name]}
                  />
                  <Pie
                    data={classificationData}
                    cx="50%"
                    cy="50%"
                    innerRadius={65}
                    outerRadius={95}
                    paddingAngle={4}
                    dataKey="value"
                    animationBegin={200}
                    animationDuration={1500}
                    animationEasing="ease-out"
                  >
                    {classificationData.map((entry: any, index: number) => (
                      <Cell key={`cell-${index}`} fill={entry.color} stroke="#0f172a" strokeWidth={2} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>

              {/* Center Donut Label */}
              <div className="absolute flex flex-col items-center justify-center pointer-events-none">
                <span className="text-2xl font-black text-white tracking-tight">
                  {kpis.fraud_rate.toFixed(1)}%
                </span>
                <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">
                  Threat Ratio
                </span>
              </div>
            </div>
          </div>

          {/* Classification Legend / Breakdowns */}
          <div className="grid grid-cols-3 gap-2 pt-4 border-t border-navy-700">
            {classificationData.map((item: any, idx: number) => (
              <div key={idx} className="bg-navy-900/60 p-2.5 rounded-xl border border-navy-700/60 text-center">
                <div className="flex items-center justify-center gap-1.5 mb-1">
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: item.color }} />
                  <span className="text-[10px] font-semibold text-slate-300 truncate">{item.name}</span>
                </div>
                <div className="text-sm font-bold text-white">{item.value?.toLocaleString()}</div>
                <div className="text-[10px] text-slate-400">{item.pct}%</div>
              </div>
            ))}
          </div>
        </div>

        {/* Animated Bar Chart: Volume by Transaction Modality */}
        <div className="lg:col-span-7 bg-navy-800/80 border border-navy-700 rounded-2xl p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-1">
              <h3 className="text-sm font-bold text-white tracking-tight flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-cyan-400" />
                Transaction Volume by Modality
              </h3>
              <span className="text-[10px] text-slate-400 font-mono">Animated Progressive Rise</span>
            </div>
            <p className="text-xs text-slate-400 mb-4">
              Aggregate payment channel distribution across TRANSFER, CASH_OUT, PAYMENT, and DEPOSIT.
            </p>

            <div className="h-60">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={volumeByType} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                  <XAxis 
                    dataKey="type" 
                    stroke="#64748b" 
                    fontSize={11} 
                    tickLine={false} 
                    axisLine={{ stroke: '#334155' }} 
                  />
                  <YAxis 
                    stroke="#64748b" 
                    fontSize={11} 
                    tickLine={false} 
                    axisLine={{ stroke: '#334155' }}
                    tickFormatter={(val) => val >= 1000 ? `${(val / 1000).toFixed(0)}k` : val}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0a0f1d',
                      borderColor: '#1e293b',
                      borderRadius: '0.75rem',
                      color: '#f8fafc',
                      fontSize: '12px',
                    }}
                    cursor={{ fill: 'rgba(255, 255, 255, 0.05)' }}
                    formatter={(val: any) => [`${Number(val).toLocaleString()} txns`, 'Volume']}
                  />
                  <Bar 
                    dataKey="count" 
                    fill="#10b981" 
                    radius={[6, 6, 0, 0]}
                    animationDuration={1500}
                    animationEasing="ease-out"
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Quick Summary Strip */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 pt-4 border-t border-navy-700">
            <div className="bg-navy-900/60 p-2.5 rounded-xl border border-navy-700/60">
              <span className="text-[10px] text-slate-400 font-medium block">Unique Senders</span>
              <span className="text-xs font-bold text-white mt-0.5 block">{kpis.unique_senders?.toLocaleString()}</span>
            </div>
            <div className="bg-navy-900/60 p-2.5 rounded-xl border border-navy-700/60">
              <span className="text-[10px] text-slate-400 font-medium block">Peak Transaction</span>
              <span className="text-xs font-bold text-white mt-0.5 block">₹{kpis.max_amount?.toLocaleString()}</span>
            </div>
            <div className="bg-navy-900/60 p-2.5 rounded-xl border border-navy-700/60">
              <span className="text-[10px] text-slate-400 font-medium block">Feature Vector</span>
              <span className="text-xs font-bold text-emerald-400 mt-0.5 block">{kpis.features_count} Dimensions</span>
            </div>
            <div className="bg-navy-900/60 p-2.5 rounded-xl border border-navy-700/60">
              <span className="text-[10px] text-slate-400 font-medium block">Inspection Status</span>
              <span className="text-xs font-bold text-emerald-400 mt-0.5 block">Valid Ingest</span>
            </div>
          </div>
        </div>
      </div>

      {/* Risk Tier Breakdown & Live Alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Risk Tiers */}
        <div className="lg:col-span-5 bg-navy-800/80 border border-navy-700 rounded-2xl p-6">
          <h3 className="text-sm font-bold text-white tracking-tight mb-1">
            Calibrated Risk Tier Segments
          </h3>
          <p className="text-xs text-slate-400 mb-4">
            Dynamic threshold classification across all historical transaction scores.
          </p>

          <div className="space-y-3">
            {riskDistribution.map((item: any, idx: number) => (
              <div key={idx} className="bg-navy-900/60 p-3 rounded-xl border border-navy-700/60">
                <div className="flex justify-between items-center text-xs mb-1.5">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: item.color }} />
                    <span className="font-semibold text-slate-200">{item.tier}</span>
                  </div>
                  <div className="font-mono text-slate-300">
                    <span className="text-white font-bold">{item.count?.toLocaleString()}</span>
                    <span className="text-slate-500 text-[11px] ml-1.5">({item.pct}%)</span>
                  </div>
                </div>
                <div className="w-full bg-navy-800 h-2 rounded-full overflow-hidden">
                  <div 
                    className="h-full rounded-full transition-all duration-1000 ease-out"
                    style={{ 
                      width: `${Math.max(item.pct, 1)}%`, 
                      backgroundColor: item.color 
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Flagged Alert Stream */}
        <div className="lg:col-span-7 bg-navy-800/80 border border-navy-700 rounded-2xl p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-1">
              <h3 className="text-sm font-bold text-white tracking-tight flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                <span>High-Risk Transaction Alert Feed</span>
              </h3>
              <span className="text-[10px] bg-red-500/10 text-red-400 border border-red-500/20 px-2 py-0.5 rounded font-mono">
                FLAGGED
              </span>
            </div>
            <p className="text-xs text-slate-400 mb-4">
              Recent anomalies flagged by ensemble behavioral anomaly scoring.
            </p>

            <div className="space-y-2">
              {alertStream.slice(0, 5).map((alert: any, idx: number) => (
                <div 
                  key={idx}
                  className="bg-navy-900/70 border border-navy-700/60 hover:border-navy-600 rounded-xl p-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg bg-red-500/10 border border-red-500/20 flex items-center justify-center text-red-400 font-bold shrink-0">
                      !
                    </div>
                    <div>
                      <div className="font-medium text-slate-200">
                        {alert.sender_id || `User_${idx + 100}`} → {alert.receiver_id || 'Mule_Node'}
                      </div>
                      <div className="text-[10px] text-slate-400 mt-0.5">
                        Type: <span className="font-mono text-slate-300">{alert.transaction_type || 'TRANSFER'}</span>
                        {alert.location && ` • Loc: ${alert.location}`}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-4 justify-between sm:justify-end">
                    <div className="text-right">
                      <div className="font-mono font-bold text-white text-sm">
                        ₹{Number(alert.amount || 0).toLocaleString()}
                      </div>
                      <div className="text-[10px] font-mono text-red-400">
                        {alert.risk_score ? `Score: ${alert.risk_score}` : 'High Risk'}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-navy-700 flex justify-end">
            <button
              onClick={() => onNavigateToTab('analytics')}
              className="text-xs text-emerald-400 hover:text-emerald-300 flex items-center gap-1 font-semibold transition-colors"
            >
              <span>View In-Depth Fraud Analytics & Mule Graphs</span>
              <ArrowUpRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
