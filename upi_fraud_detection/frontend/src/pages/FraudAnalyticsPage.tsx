import React, { useEffect, useState } from 'react';
import { 
  Activity, 
  Clock, 
  BarChart2, 
  Share2, 
  AlertOctagon
} from 'lucide-react';
import { 
  AreaChart, 
  Area, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer 
} from 'recharts';
import { api } from '../api/client';

export const FraudAnalyticsPage: React.FC = () => {
  const [analytics, setAnalytics] = useState<any | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    const fetchAnalytics = async () => {
      setLoading(true);
      try {
        const data = await api.getAnalytics();
        if (mounted) {
          setAnalytics(data);
          setError(null);
        }
      } catch (err: any) {
        if (mounted) {
          setError(err.message || "Failed to load analytics data.");
        }
      } finally {
        if (mounted) setLoading(false);
      }
    };

    fetchAnalytics();
    return () => { mounted = false; };
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[500px] space-y-4">
        <div className="w-10 h-10 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin" />
        <p className="text-sm text-slate-400 font-medium">Computing temporal curves and topological mule graphs...</p>
      </div>
    );
  }

  if (error || !analytics) {
    return (
      <div className="bg-navy-800/80 border border-navy-700 rounded-2xl p-8 text-center max-w-lg mx-auto mt-12 space-y-3">
        <AlertOctagon className="w-10 h-10 text-amber-400 mx-auto" />
        <h3 className="text-base font-bold text-white">Dataset Ingestion Required</h3>
        <p className="text-xs text-slate-400">{error || "Please ingest a dataset to access fraud analytics."}</p>
      </div>
    );
  }

  const { hourly_pattern, amount_distribution, modality_vulnerability, network_graph, archetypes } = analytics;

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-xs font-semibold text-emerald-400 uppercase tracking-wider mb-1">
          <Activity className="w-3.5 h-3.5" />
          <span>Advanced Forensic Analytics</span>
        </div>
        <h1 className="text-2xl font-bold text-white tracking-tight">Temporal & Topological Fraud Analytics</h1>
        <p className="text-slate-400 text-sm mt-1 max-w-3xl">
          Deep behavioral telemetry across diurnal cycles, log-normal financial density, transaction channel vulnerability, and NetworkX money mule routing networks.
        </p>
      </div>

      {/* Row 1: Hourly Temporal Pattern & Transaction Amount Density */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Diurnal Hourly Attack Curve */}
        <div className="bg-navy-800/80 border border-navy-700 rounded-2xl p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-1">
              <h3 className="text-sm font-bold text-white tracking-tight flex items-center gap-2">
                <Clock className="w-4 h-4 text-emerald-400" />
                <span>Diurnal Temporal Attack Curve (24-Hour)</span>
              </h3>
              <span className="text-[10px] text-slate-400 font-mono">Animated Spline</span>
            </div>
            <p className="text-xs text-slate-400 mb-4">
              Normal diurnal consumer activity peaks during daytime; automated siphon attacks spike between 1:00 AM – 4:00 AM.
            </p>

            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={hourly_pattern} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
                  <defs>
                    <linearGradient id="normalGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.4}/>
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="fraudGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ef4444" stopOpacity={0.7}/>
                      <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                  <XAxis dataKey="label" stroke="#64748b" fontSize={10} tickLine={false} />
                  <YAxis stroke="#64748b" fontSize={10} tickLine={false} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0a0f1d',
                      borderColor: '#1e293b',
                      borderRadius: '0.75rem',
                      color: '#f8fafc',
                      fontSize: '12px',
                    }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="normal_count" 
                    stroke="#10b981" 
                    fillOpacity={1} 
                    fill="url(#normalGrad)" 
                    name="Legitimate"
                    animationDuration={1500}
                    animationEasing="ease-out"
                  />
                  <Area 
                    type="monotone" 
                    dataKey="fraud_count" 
                    stroke="#ef4444" 
                    fillOpacity={1} 
                    fill="url(#fraudGrad)" 
                    name="Fraud Incident"
                    animationDuration={1800}
                    animationEasing="ease-out"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="flex items-center justify-between text-xs pt-4 border-t border-navy-700 text-slate-400">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" /> Legitimate Volume
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-red-400" /> Fraud Incidents
            </span>
            <span className="font-mono text-[10px] text-slate-500">24 Bins</span>
          </div>
        </div>

        {/* Transaction Amount Density (Log10) */}
        <div className="bg-navy-800/80 border border-navy-700 rounded-2xl p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-1">
              <h3 className="text-sm font-bold text-white tracking-tight flex items-center gap-2">
                <BarChart2 className="w-4 h-4 text-cyan-400" />
                <span>Transaction Value Spectrum (Log₁₀ Bins)</span>
              </h3>
              <span className="text-[10px] text-slate-400 font-mono">₹10 → ₹1M+</span>
            </div>
            <p className="text-xs text-slate-400 mb-4">
              Micro-payments (₹10–₹500) show high consumer density; illicit cash-outs cluster in upper brackets (₹30,000+).
            </p>

            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={amount_distribution} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                  <XAxis dataKey="range" stroke="#64748b" fontSize={9} tickLine={false} interval={0} />
                  <YAxis stroke="#64748b" fontSize={10} tickLine={false} />
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
                  <Bar 
                    dataKey="normal" 
                    stackId="a"
                    fill="#10b981" 
                    name="Legitimate"
                    animationDuration={1500}
                    animationEasing="ease-out"
                  />
                  <Bar 
                    dataKey="fraud" 
                    stackId="a"
                    fill="#ef4444" 
                    radius={[4, 4, 0, 0]}
                    name="Fraud Incident"
                    animationDuration={1800}
                    animationEasing="ease-out"
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="flex items-center justify-between text-xs pt-4 border-t border-navy-700 text-slate-400">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" /> Legitimate Volume
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-red-400" /> Fraud Incidents
            </span>
            <span className="font-mono text-[10px] text-slate-500">Log-Normal Spectrum</span>
          </div>
        </div>
      </div>

      {/* Row 2: Modality Vulnerability Analysis */}
      <div className="bg-navy-800/80 border border-navy-700 rounded-2xl p-6">
        <h3 className="text-sm font-bold text-white tracking-tight mb-1">
          Payment Modality Vulnerability Spectrum
        </h3>
        <p className="text-xs text-slate-400 mb-4">
          Comparative empirical risk and fraud prevalence across each transaction modality.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {modality_vulnerability.map((mod: any, idx: number) => {
            const isHigh = mod.fraud_rate > 5;
            const isMed = mod.fraud_rate > 1;
            return (
              <div 
                key={idx}
                className="bg-navy-900/60 p-4 rounded-xl border border-navy-700/60 flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-mono font-bold text-xs text-white">{mod.type}</span>
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                      isHigh ? 'bg-red-500/20 text-red-400' : isMed ? 'bg-amber-500/20 text-amber-400' : 'bg-emerald-500/20 text-emerald-400'
                    }`}>
                      {isHigh ? 'HIGH RISK' : isMed ? 'ELEVATED' : 'STABLE'}
                    </span>
                  </div>
                  <div className="text-xl font-bold text-white mb-1">
                    {mod.fraud_rate.toFixed(2)}%
                  </div>
                  <p className="text-[10px] text-slate-400">
                    Fraud Incident Rate
                  </p>
                </div>

                <div className="mt-4 pt-3 border-t border-navy-700/60 flex justify-between text-[11px] text-slate-400">
                  <span>Volume: {mod.total_volume?.toLocaleString()}</span>
                  <span className="text-red-400 font-semibold">{mod.fraud_volume} Frauds</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Row 3: Real-World Fraud Archetypes */}
      <div>
        <h3 className="text-sm font-bold text-white tracking-tight mb-1">
          Cataloged Fraud Archetypes & Exploitation Vectors
        </h3>
        <p className="text-xs text-slate-400 mb-4">
          Structural typology detected in UPI payment networks based on empirical behavioral signals.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {archetypes.map((arch: any) => (
            <div 
              key={arch.id}
              className="bg-navy-800/80 border border-navy-700/80 rounded-2xl p-5 flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] font-mono text-slate-400">ARCHETYPE #{arch.id}</span>
                  <span 
                    className="text-[10px] font-bold px-2 py-0.5 rounded"
                    style={{ backgroundColor: `${arch.color}20`, color: arch.color }}
                  >
                    {arch.severity}
                  </span>
                </div>
                <h4 className="text-xs font-bold text-white mb-2">{arch.title}</h4>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  {arch.description}
                </p>
              </div>

              <div className="mt-4 pt-3 border-t border-navy-700/60 flex items-center justify-between text-[10px] text-slate-400">
                <span>Ensemble Trigger</span>
                <span className="font-mono text-emerald-400">Active Monitor</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Row 4: NetworkX Topological Mule Graph Analysis */}
      <div className="bg-navy-800/80 border border-navy-700 rounded-2xl p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
          <div>
            <div className="flex items-center gap-2">
              <Share2 className="w-4 h-4 text-emerald-400" />
              <h3 className="text-sm font-bold text-white tracking-tight">
                NetworkX Topological Money Mule Hubs
              </h3>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              High in-degree accounts receiving sudden multi-source transfers followed by rapid liquidation.
            </p>
          </div>

          <div className="flex items-center gap-3 text-xs">
            <div className="bg-navy-900/80 px-3 py-1.5 rounded-lg border border-navy-700">
              <span className="text-slate-400">Sub-Graph Nodes: </span>
              <span className="font-mono text-white font-bold">{network_graph?.nodes?.length || 0}</span>
            </div>
            <div className="bg-navy-900/80 px-3 py-1.5 rounded-lg border border-navy-700">
              <span className="text-slate-400">Directed Edges: </span>
              <span className="font-mono text-white font-bold">{network_graph?.links?.length || 0}</span>
            </div>
          </div>
        </div>

        {/* Top Mule Nodes Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-navy-900/80 text-slate-400 uppercase text-[10px] font-semibold border-b border-navy-700">
              <tr>
                <th className="py-2.5 px-4">Account / Node ID</th>
                <th className="py-2.5 px-4">In-Degree (Sources)</th>
                <th className="py-2.5 px-4">Out-Degree (Targets)</th>
                <th className="py-2.5 px-4">Total Flow (₹ INR)</th>
                <th className="py-2.5 px-4">Topological Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-navy-700/60 font-mono text-[11px]">
              {network_graph?.top_nodes?.map((node: any, idx: number) => {
                const isSuspect = node.in_degree >= 2 || node.flow_volume > 50000;
                return (
                  <tr key={idx} className="hover:bg-navy-700/30 transition-colors">
                    <td className="py-3 px-4 text-white font-medium">{node.node_id}</td>
                    <td className="py-3 px-4 text-cyan-400">{node.in_degree} incoming</td>
                    <td className="py-3 px-4 text-slate-300">{node.out_degree} outgoing</td>
                    <td className="py-3 px-4 text-white font-bold">₹{Number(node.flow_volume).toLocaleString()}</td>
                    <td className="py-3 px-4 font-sans">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        isSuspect ? 'bg-red-500/20 text-red-400 border border-red-500/30' : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                      }`}>
                        {isSuspect ? 'MULE AGGREGATION HUB' : 'NORMAL PARTICIPANT'}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
