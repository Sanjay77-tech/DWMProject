import React, { useState, useEffect } from 'react';
import { 
  SearchCheck, 
  Sparkles, 
  Zap,
  AlertOctagon
} from 'lucide-react';
import { api } from '../api/client';
import { RiskScoreGauge } from '../components/RiskScoreGauge';

export const TransactionPredictionPage: React.FC = () => {
  const [presets, setPresets] = useState<any[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState<number | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [predictionResult, setPredictionResult] = useState<any | null>(null);

  // Form State
  const [formData, setFormData] = useState({
    amount: 350.0,
    transaction_type: 'PAYMENT',
    hour_of_day: 9,
    sender_id: 'rahul.sharma@okhdfcbank',
    receiver_id: 'chai_corner@paytm',
    sender_avg_amt: 450.0,
    tx_count_5m: 0,
    tx_count_30m: 0,
    tx_count_24h: 1,
    time_since_prev_min: 180.0,
    is_new_recipient: 0,
    is_device_changed: 0,
    is_location_changed: 0,
    failed_attempts: 0,
    old_balance: 14500.0,
    new_balance: 14150.0,
  });

  const handleApplyPreset = async (preset: any) => {
    setSelectedPresetId(preset.id);
    const presetData = preset.data || preset.payload || {};
    const updatedForm = {
      ...formData,
      ...presetData,
    };
    setFormData(updatedForm);
    setAnalyzing(true);
    setError(null);
    try {
      const response = await api.analyzeTransaction(updatedForm);
      setPredictionResult(response.report);
    } catch (err: any) {
      setError(err.message || "Failed to analyze transaction.");
    } finally {
      setAnalyzing(false);
    }
  };

  useEffect(() => {
    const fetchPresets = async () => {
      try {
        const data = await api.getPresets();
        const pList = data.presets || [];
        setPresets(pList);
        if (pList.length > 0) {
          const first = pList[0];
          setSelectedPresetId(first.id);
          const firstData = first.data || first.payload || {};
          const initialForm = { ...formData, ...firstData };
          setFormData(initialForm);
          api.analyzeTransaction(initialForm).then(res => {
            setPredictionResult(res.report);
          }).catch(() => {});
        }
      } catch (err) {
        console.error("Failed to load presets", err);
      }
    };
    fetchPresets();
  }, []);

  const handleInputChange = (field: string, value: any) => {
    setSelectedPresetId(null);
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    setAnalyzing(true);
    setError(null);

    try {
      const response = await api.analyzeTransaction(formData);
      setPredictionResult(response.report);
    } catch (err: any) {
      setError(err.message || "Failed to analyze transaction.");
    } finally {
      setAnalyzing(false);
    }
  };

  const report = predictionResult;

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-xs font-semibold text-emerald-400 uppercase tracking-wider mb-1">
          <SearchCheck className="w-3.5 h-3.5" />
          <span>Real-Time Inference Engine</span>
        </div>
        <h1 className="text-2xl font-bold text-white tracking-tight">Transaction Risk Inference & XAI</h1>
        <p className="text-slate-400 text-sm mt-1 max-w-3xl">
          Simulate or evaluate live UPI transactions against our trained ensemble (Random Forest, Isolation Forest, and Self-Training) with full Explainable AI attribution.
        </p>
      </div>

      {/* 4 Attack Scenario Presets */}
      <div>
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-300 mb-3">
          <Sparkles className="w-4 h-4 text-emerald-400" />
          <span>Demonstration Scenario Presets</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          {presets.map((preset) => (
            <button
              key={preset.id}
              onClick={() => handleApplyPreset(preset)}
              type="button"
              className={`p-3.5 rounded-xl text-left border transition-all ${
                selectedPresetId === preset.id
                  ? 'bg-emerald-500/10 border-emerald-500 text-white shadow-md shadow-emerald-500/10'
                  : 'bg-navy-800/80 border-navy-700/80 hover:border-navy-600 text-slate-300 hover:text-white'
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="font-semibold text-xs tracking-tight">{preset.title}</span>
                <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                  preset.risk === 'LOW' 
                    ? 'bg-emerald-500/20 text-emerald-400' 
                    : preset.risk === 'CRITICAL' 
                    ? 'bg-red-500/20 text-red-400' 
                    : 'bg-amber-500/20 text-amber-400'
                }`}>
                  {preset.risk}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 line-clamp-2 mt-1">
                {preset.description}
              </p>
            </button>
          ))}
        </div>
      </div>

      {/* Error alert if any */}
      {error && (
        <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 flex items-start gap-3 text-red-300 text-xs animate-fadeIn">
          <AlertOctagon className="w-4 h-4 shrink-0 mt-0.5 text-red-400" />
          <div>
            <span className="font-semibold text-red-200">Prediction Error: </span>
            {error}
          </div>
        </div>
      )}

      {/* Main Grid: Form & Real-Time Assessment */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Form Inputs (Left) */}
        <div className="lg:col-span-7 bg-navy-800/80 border border-navy-700 rounded-2xl p-6">
          <form onSubmit={handleAnalyze} className="space-y-6">
            {/* Section 1: Financial & Channel Parameters */}
            <div>
              <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-3 flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                1. Financial & Channel Parameters
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div>
                  <label className="block text-[11px] font-medium text-slate-400 mb-1">
                    Amount (₹ INR)
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    min="1"
                    value={formData.amount}
                    onChange={(e) => handleInputChange('amount', parseFloat(e.target.value) || 0)}
                    className="w-full bg-navy-900 border border-navy-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-emerald-500 font-mono"
                    required
                  />
                </div>

                <div>
                  <label className="block text-[11px] font-medium text-slate-400 mb-1">
                    Transaction Type
                  </label>
                  <select
                    value={formData.transaction_type}
                    onChange={(e) => handleInputChange('transaction_type', e.target.value)}
                    className="w-full bg-navy-900 border border-navy-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-emerald-500"
                  >
                    <option value="PAYMENT">PAYMENT (P2M)</option>
                    <option value="TRANSFER">TRANSFER (P2P)</option>
                    <option value="CASH_OUT">CASH_OUT (Atm/Agent)</option>
                    <option value="DEPOSIT">DEPOSIT</option>
                  </select>
                </div>

                <div>
                  <label className="block text-[11px] font-medium text-slate-400 mb-1">
                    Hour of Day (0–23)
                  </label>
                  <input
                    type="number"
                    min="0"
                    max="23"
                    value={formData.hour_of_day}
                    onChange={(e) => handleInputChange('hour_of_day', parseInt(e.target.value) || 0)}
                    className="w-full bg-navy-900 border border-navy-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-emerald-500 font-mono"
                  />
                </div>
              </div>
            </div>

            {/* Section 2: Account VPAs & Balance Depletion */}
            <div>
              <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-3 flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                2. Balances & Identifiers
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-[11px] font-medium text-slate-400 mb-1">
                    Sender UPI ID (VPA)
                  </label>
                  <input
                    type="text"
                    value={formData.sender_id}
                    onChange={(e) => handleInputChange('sender_id', e.target.value)}
                    className="w-full bg-navy-900 border border-navy-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-emerald-500 font-mono"
                  />
                </div>
                <div>
                  <label className="block text-[11px] font-medium text-slate-400 mb-1">
                    Receiver UPI ID (VPA)
                  </label>
                  <input
                    type="text"
                    value={formData.receiver_id}
                    onChange={(e) => handleInputChange('receiver_id', e.target.value)}
                    className="w-full bg-navy-900 border border-navy-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-emerald-500 font-mono"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div>
                  <label className="block text-[11px] font-medium text-slate-400 mb-1">
                    Sender Old Balance (₹)
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    value={formData.old_balance}
                    onChange={(e) => handleInputChange('old_balance', parseFloat(e.target.value) || 0)}
                    className="w-full bg-navy-900 border border-navy-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-emerald-500 font-mono"
                  />
                </div>
                <div>
                  <label className="block text-[11px] font-medium text-slate-400 mb-1">
                    Sender New Balance (₹)
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    value={formData.new_balance}
                    onChange={(e) => handleInputChange('new_balance', parseFloat(e.target.value) || 0)}
                    className="w-full bg-navy-900 border border-navy-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-emerald-500 font-mono"
                  />
                </div>
                <div>
                  <label className="block text-[11px] font-medium text-slate-400 mb-1">
                    Sender 30-Day Avg (₹)
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    value={formData.sender_avg_amt}
                    onChange={(e) => handleInputChange('sender_avg_amt', parseFloat(e.target.value) || 0)}
                    className="w-full bg-navy-900 border border-navy-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-emerald-500 font-mono"
                  />
                </div>
              </div>
            </div>

            {/* Section 3: Velocity & Contextual Anomalies */}
            <div>
              <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-3 flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                3. Velocity & Contextual Behavioral Anomaly Flags
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
                <div>
                  <label className="block text-[10px] text-slate-400 mb-1">Tx Count (5m)</label>
                  <input
                    type="number"
                    min="0"
                    value={formData.tx_count_5m}
                    onChange={(e) => handleInputChange('tx_count_5m', parseInt(e.target.value) || 0)}
                    className="w-full bg-navy-900 border border-navy-700 rounded-lg p-2 text-xs text-white font-mono"
                  />
                </div>
                <div>
                  <label className="block text-[10px] text-slate-400 mb-1">Tx Count (30m)</label>
                  <input
                    type="number"
                    min="0"
                    value={formData.tx_count_30m}
                    onChange={(e) => handleInputChange('tx_count_30m', parseInt(e.target.value) || 0)}
                    className="w-full bg-navy-900 border border-navy-700 rounded-lg p-2 text-xs text-white font-mono"
                  />
                </div>
                <div>
                  <label className="block text-[10px] text-slate-400 mb-1">Tx Count (24h)</label>
                  <input
                    type="number"
                    min="0"
                    value={formData.tx_count_24h}
                    onChange={(e) => handleInputChange('tx_count_24h', parseInt(e.target.value) || 0)}
                    className="w-full bg-navy-900 border border-navy-700 rounded-lg p-2 text-xs text-white font-mono"
                  />
                </div>
                <div>
                  <label className="block text-[10px] text-slate-400 mb-1">Min Since Prev Tx</label>
                  <input
                    type="number"
                    min="0"
                    value={formData.time_since_prev_min}
                    onChange={(e) => handleInputChange('time_since_prev_min', parseFloat(e.target.value) || 0)}
                    className="w-full bg-navy-900 border border-navy-700 rounded-lg p-2 text-xs text-white font-mono"
                  />
                </div>
              </div>

              {/* Toggles */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <label className="flex items-center gap-2 p-2.5 bg-navy-900/60 rounded-xl border border-navy-700/60 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={formData.is_new_recipient === 1}
                    onChange={(e) => handleInputChange('is_new_recipient', e.target.checked ? 1 : 0)}
                    className="rounded border-navy-600 text-emerald-500 focus:ring-0"
                  />
                  <span className="text-xs text-slate-300">New Recipient</span>
                </label>

                <label className="flex items-center gap-2 p-2.5 bg-navy-900/60 rounded-xl border border-navy-700/60 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={formData.is_device_changed === 1}
                    onChange={(e) => handleInputChange('is_device_changed', e.target.checked ? 1 : 0)}
                    className="rounded border-navy-600 text-emerald-500 focus:ring-0"
                  />
                  <span className="text-xs text-slate-300">Device Changed</span>
                </label>

                <label className="flex items-center gap-2 p-2.5 bg-navy-900/60 rounded-xl border border-navy-700/60 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={formData.is_location_changed === 1}
                    onChange={(e) => handleInputChange('is_location_changed', e.target.checked ? 1 : 0)}
                    className="rounded border-navy-600 text-emerald-500 focus:ring-0"
                  />
                  <span className="text-xs text-slate-300">Location Shift</span>
                </label>

                <div>
                  <input
                    type="number"
                    min="0"
                    max="10"
                    placeholder="Failed PINs"
                    value={formData.failed_attempts}
                    onChange={(e) => handleInputChange('failed_attempts', parseInt(e.target.value) || 0)}
                    className="w-full bg-navy-900 border border-navy-700 rounded-xl p-2 text-xs text-white placeholder-slate-500 font-mono"
                  />
                </div>
              </div>
            </div>

            {/* Action Button */}
            <button
              type="submit"
              disabled={analyzing}
              className="w-full py-3 px-4 bg-emerald-500 hover:bg-emerald-400 text-navy-950 font-bold text-sm rounded-xl transition-all shadow-lg shadow-emerald-500/20 flex items-center justify-center gap-2"
            >
              {analyzing ? (
                <>
                  <div className="w-4 h-4 border-2 border-navy-950 border-t-transparent rounded-full animate-spin" />
                  <span>Computing Multi-Model Risk Score...</span>
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  <span>Analyse Transaction in Real Time</span>
                </>
              )}
            </button>
          </form>
        </div>

        {/* Prediction Results & XAI (Right) */}
        <div className="lg:col-span-5 space-y-6">
          {report ? (
            <div className="bg-navy-800/80 border border-navy-700 rounded-2xl p-6 space-y-6 animate-fadeIn">
              {/* Header */}
              <div className="flex items-center justify-between pb-4 border-b border-navy-700">
                <div>
                  <h3 className="text-sm font-bold text-white tracking-tight">Risk Assessment Verdict</h3>
                  <span className="text-[10px] text-slate-400 font-mono">Consensus Analysis</span>
                </div>
                <div className={`px-2.5 py-1 rounded-full text-xs font-bold border ${
                  report.risk_category === 'VERY HIGH RISK' || report.risk_category === 'CRITICAL RISK'
                    ? 'bg-red-500/10 border-red-500/30 text-red-400'
                    : report.risk_category === 'HIGH RISK'
                    ? 'bg-orange-500/10 border-orange-500/30 text-orange-400'
                    : report.risk_category === 'MEDIUM RISK'
                    ? 'bg-amber-500/10 border-amber-500/30 text-amber-400'
                    : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                }`}>
                  {report.risk_category}
                </div>
              </div>

              {/* Gauge and Suggested Action */}
              <div className="flex flex-col items-center justify-center py-2">
                <RiskScoreGauge 
                  score={report.risk_score} 
                  category={report.risk_category} 
                  status={report.prediction_status || 'SUSPECT'}
                />
                <div className="mt-4 text-center max-w-xs">
                  <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider block mb-1">
                    Operational Action Protocol
                  </span>
                  <div className={`text-xs font-semibold px-3 py-1.5 rounded-lg border ${
                    report.risk_score >= 80 
                      ? 'bg-red-500/10 border-red-500/30 text-red-300' 
                      : report.risk_score >= 50 
                      ? 'bg-amber-500/10 border-amber-500/30 text-amber-300' 
                      : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
                  }`}>
                    {report.suggested_action}
                  </div>
                </div>
              </div>

              {/* Multi-Model Probabilities */}
              <div className="bg-navy-900/60 p-4 rounded-xl border border-navy-700/60 space-y-3">
                <span className="text-[11px] font-bold text-slate-300 uppercase tracking-wider block">
                  Ensemble Confidence Breakdown
                </span>
                
                {report.model_breakdown && (
                  <div className="space-y-2 text-xs">
                    <div>
                      <div className="flex justify-between text-slate-300 mb-1">
                        <span>Supervised (Random Forest)</span>
                        <span className="font-mono text-white">{(report.model_breakdown.random_forest * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-navy-800 h-1.5 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-emerald-400 rounded-full" 
                          style={{ width: `${report.model_breakdown.random_forest * 100}%` }} 
                        />
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between text-slate-300 mb-1">
                        <span>Anomaly Detector (Isolation Forest)</span>
                        <span className="font-mono text-white">{(report.model_breakdown.isolation_forest * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-navy-800 h-1.5 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-cyan-400 rounded-full" 
                          style={{ width: `${report.model_breakdown.isolation_forest * 100}%` }} 
                        />
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between text-slate-300 mb-1">
                        <span>Semi-Supervised (Self-Training)</span>
                        <span className="font-mono text-white">{(report.model_breakdown.self_training * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-navy-800 h-1.5 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-amber-400 rounded-full" 
                          style={{ width: `${report.model_breakdown.self_training * 100}%` }} 
                        />
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Plain-Language XAI Explanations */}
              <div>
                <span className="text-[11px] font-bold text-slate-300 uppercase tracking-wider block mb-2.5">
                  Explainable AI (XAI) Attribution Factors
                </span>
                <div className="space-y-2">
                  {report.explanations?.map((exp: any, idx: number) => (
                    <div 
                      key={idx}
                      className="p-3 bg-navy-900/70 border border-navy-700/60 rounded-xl flex items-start gap-2.5 text-xs"
                    >
                      <span className={`w-2 h-2 rounded-full mt-1.5 shrink-0 ${
                        exp.impact === 'CRITICAL' ? 'bg-red-400' : exp.impact === 'HIGH' ? 'bg-amber-400' : 'bg-emerald-400'
                      }`} />
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <span className="font-semibold text-white">{exp.feature}</span>
                          <span className="text-[10px] font-mono text-slate-400">{exp.impact}</span>
                        </div>
                        <p className="text-slate-400 text-[11px] mt-0.5">{exp.explanation}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-navy-800/80 border border-navy-700 rounded-2xl p-8 text-center flex flex-col items-center justify-center min-h-[380px] space-y-3">
              <div className="w-12 h-12 rounded-2xl bg-navy-700 border border-navy-600 flex items-center justify-center text-slate-400">
                <SearchCheck className="w-6 h-6" />
              </div>
              <h4 className="text-sm font-bold text-white">Awaiting Transaction Analysis</h4>
              <p className="text-xs text-slate-400 max-w-xs">
                Select one of the 4 scenario presets above or customize the form values, then click "Analyse Transaction" to view the animated risk breakdown.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
