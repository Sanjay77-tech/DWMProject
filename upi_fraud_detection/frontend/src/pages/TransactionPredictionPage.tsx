import React, { useState, useEffect } from 'react';
import { 
  SearchCheck, 
  Sparkles, 
  Zap,
  AlertOctagon,
  Cpu,
  CheckCircle2,
  ShieldAlert,
  Info
} from 'lucide-react';
import { api } from '../api/client';
import { RiskScoreGauge } from '../components/RiskScoreGauge';

export const TransactionPredictionPage: React.FC = () => {
  const [presets, setPresets] = useState<any[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [predictionResult, setPredictionResult] = useState<any | null>(null);

  // Active Model Registry State
  const [availableModels, setAvailableModels] = useState<any[]>([]);
  const [activeModelId, setActiveModelId] = useState<string>('random_forest');
  const [activeModelMeta, setActiveModelMeta] = useState<any | null>(null);
  const [switchingModel, setSwitchingModel] = useState<boolean>(false);

  // Form State
  const [formData, setFormData] = useState({
    amount: 350.0,
    transaction_type: 'PAYMENT',
    hour_of_day: 14,
    sender_id: 'ananya.sharma@okaxis',
    receiver_id: 'fresh_groceries@paytm',
    sender_avg_amt: 420.0,
    tx_count_5m: 0,
    tx_count_30m: 0,
    tx_count_24h: 1,
    time_since_prev_min: 180.0,
    is_new_recipient: 0,
    is_device_changed: 0,
    is_location_changed: 0,
    failed_attempts: 0,
    old_balance: 18500.0,
    new_balance: 18150.0,
  });

  const loadModels = async () => {
    try {
      const data = await api.getAvailableModels();
      if (data && data.models) {
        setAvailableModels(data.models);
        setActiveModelId(data.active_model_id);
      }
      const act = await api.getActiveModel();
      if (act && act.active_model) {
        setActiveModelMeta(act.active_model);
      }
    } catch (e) {
      console.warn("Could not load model registry", e);
    }
  };

  useEffect(() => {
    loadModels();
  }, []);

  const handleModelChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newId = e.target.value;
    setSwitchingModel(true);
    try {
      await api.setActiveModel(newId);
      setActiveModelId(newId);
      await loadModels();
      // Re-run inference with newly active model
      if (predictionResult) {
        setAnalyzing(true);
        const resp = await api.analyzeTransaction({ ...formData, model_id: newId });
        setPredictionResult(resp.report);
      }
    } catch (err: any) {
      setError(err.message || "Failed to switch active model.");
    } finally {
      setSwitchingModel(false);
      setAnalyzing(false);
    }
  };

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
      const response = await api.analyzeTransaction({ ...updatedForm, model_id: activeModelId });
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
          api.analyzeTransaction({ ...initialForm, model_id: activeModelId }).then(res => {
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
      const response = await api.analyzeTransaction({ ...formData, model_id: activeModelId });
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
      {/* Header with Active Model Selector */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold text-emerald-400 uppercase tracking-wider mb-1">
            <SearchCheck className="w-3.5 h-3.5" />
            <span>Real-Time Inference Engine</span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Transaction Risk Inference & XAI</h1>
          <p className="text-slate-400 text-sm mt-1 max-w-2xl">
            Evaluate live single transactions against verified machine learning pipelines with genuine feature attributions and forensic rule auditing.
          </p>
        </div>

        {/* Active Model Indicator & Switcher */}
        <div className="bg-navy-800/90 border border-navy-700 p-3 rounded-2xl flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider block">
              Active Prediction Model
            </span>
            <div className="flex items-center gap-2 mt-0.5">
              <select
                value={activeModelId}
                onChange={handleModelChange}
                disabled={switchingModel}
                aria-label="Active Prediction Model Selector"
                className="bg-navy-900 border border-navy-600 rounded-lg text-xs font-bold text-emerald-300 py-1 px-2.5 focus:outline-none focus:border-emerald-500"
              >
                {availableModels.length > 0 ? (
                  availableModels.map(m => (
                    <option key={m.model_id} value={m.model_id}>
                      {m.algorithm} {m.is_active ? '★ (Active)' : ''}
                    </option>
                  ))
                ) : (
                  <option value="random_forest">Random Forest ★ (Active)</option>
                )}
              </select>
              {activeModelMeta?.version && (
                <span className="text-[10px] text-slate-500 font-mono hidden sm:inline">
                  {activeModelMeta.version}
                </span>
              )}
            </div>
          </div>
        </div>
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
              <p className="text-[11px] text-slate-400 line-clamp-2 leading-relaxed">
                {preset.description}
              </p>
            </button>
          ))}
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 flex items-start gap-3 text-red-300 text-xs animate-fadeIn">
          <AlertOctagon className="w-4 h-4 shrink-0 mt-0.5 text-red-400" />
          <div>{error}</div>
        </div>
      )}

      {/* Main Grid: Form (Left) & Results (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Form Inputs (Left) */}
        <div className="lg:col-span-7 bg-navy-800/80 border border-navy-700 rounded-2xl p-6">
          <form onSubmit={handleAnalyze} className="space-y-6">
            {/* Section 1: Core Transaction Values */}
            <div>
              <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-3 flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                1. Core Financial Parameters
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div>
                  <label className="block text-[10px] text-slate-400 mb-1">Amount (₹)</label>
                  <input
                    type="number"
                    step="any"
                    min="1"
                    value={formData.amount}
                    onChange={(e) => handleInputChange('amount', parseFloat(e.target.value) || 0)}
                    className="w-full bg-navy-900 border border-navy-700 rounded-lg p-2 text-xs text-white font-mono"
                  />
                </div>
                <div>
                  <label className="block text-[10px] text-slate-400 mb-1">Transaction Type</label>
                  <select
                    value={formData.transaction_type}
                    onChange={(e) => handleInputChange('transaction_type', e.target.value)}
                    className="w-full bg-navy-900 border border-navy-700 rounded-lg p-2 text-xs text-white"
                  >
                    <option value="PAYMENT">PAYMENT (P2M)</option>
                    <option value="TRANSFER">TRANSFER (P2P)</option>
                    <option value="CASH_OUT">CASH_OUT</option>
                    <option value="BILL_PAY">BILL_PAY</option>
                  </select>
                </div>
                <div>
                  <label className="block text-[10px] text-slate-400 mb-1">Hour of Day (0–23)</label>
                  <input
                    type="number"
                    min="0"
                    max="23"
                    value={formData.hour_of_day}
                    onChange={(e) => handleInputChange('hour_of_day', parseInt(e.target.value) || 0)}
                    className="w-full bg-navy-900 border border-navy-700 rounded-lg p-2 text-xs text-white font-mono"
                  />
                </div>
              </div>
            </div>

            {/* Section 2: Account & Sender Behavioral Profile */}
            <div>
              <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-3 flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                2. Counterparties & Account Balances
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-3">
                <div>
                  <label className="block text-[10px] text-slate-400 mb-1">Sender UPI ID</label>
                  <input
                    type="text"
                    value={formData.sender_id}
                    onChange={(e) => handleInputChange('sender_id', e.target.value)}
                    className="w-full bg-navy-900 border border-navy-700 rounded-lg p-2 text-xs text-white font-mono"
                  />
                </div>
                <div>
                  <label className="block text-[10px] text-slate-400 mb-1">Receiver UPI ID</label>
                  <input
                    type="text"
                    value={formData.receiver_id}
                    onChange={(e) => handleInputChange('receiver_id', e.target.value)}
                    className="w-full bg-navy-900 border border-navy-700 rounded-lg p-2 text-xs text-white font-mono"
                  />
                </div>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div>
                  <label className="block text-[10px] text-slate-400 mb-1">Old Balance (₹)</label>
                  <input
                    type="number"
                    step="any"
                    min="0"
                    value={formData.old_balance}
                    onChange={(e) => handleInputChange('old_balance', parseFloat(e.target.value) || 0)}
                    className="w-full bg-navy-900 border border-navy-700 rounded-lg p-2 text-xs text-white font-mono"
                  />
                </div>
                <div>
                  <label className="block text-[10px] text-slate-400 mb-1">New Balance (₹)</label>
                  <input
                    type="number"
                    step="any"
                    min="0"
                    value={formData.new_balance}
                    onChange={(e) => handleInputChange('new_balance', parseFloat(e.target.value) || 0)}
                    className="w-full bg-navy-900 border border-navy-700 rounded-lg p-2 text-xs text-white font-mono"
                  />
                </div>
                <div>
                  <label className="block text-[10px] text-slate-400 mb-1">Sender Avg Amount (₹)</label>
                  <input
                    type="number"
                    step="any"
                    min="1"
                    value={formData.sender_avg_amt}
                    onChange={(e) => handleInputChange('sender_avg_amt', parseFloat(e.target.value) || 0)}
                    className="w-full bg-navy-900 border border-navy-700 rounded-lg p-2 text-xs text-white font-mono"
                  />
                </div>
              </div>
            </div>

            {/* Section 3: Velocity & Contextual Behavioral Anomaly Flags */}
            <div>
              <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-3 flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                3. Velocity & Contextual Behavioral Flags
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
                  <span>Evaluating using {activeModelMeta?.algorithm || 'Active Model'}...</span>
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
              {/* Header with Active Algorithm and Version */}
              <div className="flex items-center justify-between pb-4 border-b border-navy-700">
                <div>
                  <h3 className="text-sm font-bold text-white tracking-tight flex items-center gap-2">
                    <span>Risk Verdict:</span>
                    <span className="text-emerald-400 font-mono">{report.active_algorithm || 'Random Forest'}</span>
                  </h3>
                  <span className="text-[10px] text-slate-400 font-mono">
                    Model Version: {report.model_version || 'v1.0'}
                  </span>
                </div>
                <div className={`px-2.5 py-1 rounded-full text-xs font-bold border ${
                  report.risk_category?.toUpperCase().includes('CRITICAL') || report.risk_category?.toUpperCase().includes('VERY HIGH')
                    ? 'bg-red-500/10 border-red-500/30 text-red-400'
                    : report.risk_category?.toUpperCase().includes('HIGH')
                    ? 'bg-orange-500/10 border-orange-500/30 text-orange-400'
                    : report.risk_category?.toUpperCase().includes('MEDIUM')
                    ? 'bg-amber-500/10 border-amber-500/30 text-amber-400'
                    : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                }`}>
                  {report.risk_category}
                </div>
              </div>

              {/* Behavioral Profile Indicator */}
              {report.derived_from_history ? (
                <div className="p-2.5 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 shrink-0 text-cyan-400" />
                  <span>Sender behavioral baseline retrieved from persistent history.</span>
                </div>
              ) : (
                <div className="p-2.5 rounded-xl bg-slate-700/30 border border-slate-600/40 text-slate-300 text-xs flex items-center gap-2">
                  <Info className="w-4 h-4 shrink-0 text-slate-400" />
                  <span>First-seen account: default cold-start behavioral baseline applied.</span>
                </div>
              )}

              {/* Gauge and Suggested Action */}
              <div className="flex flex-col items-center justify-center py-2">
                <RiskScoreGauge 
                  score={report.risk_score} 
                  category={report.risk_category} 
                  status={report.prediction_status || 'SUSPECT'}
                />
                <div className="mt-4 text-center max-w-xs">
                  <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider block mb-1">
                    Operational Protocol
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

              {/* Model Confidence & Calculation Formula */}
              <div className="bg-navy-900/60 p-4 rounded-xl border border-navy-700/60 space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-400">Predicted Fraud Probability:</span>
                  <span className="font-mono font-bold text-emerald-400">
                    {report.fraud_probability !== null && report.fraud_probability !== undefined 
                      ? `${report.fraud_probability}%` 
                      : 'N/A (Unsupervised Anomaly Mode)'}
                  </span>
                </div>
                {report.components?.anomaly_score !== null && report.components?.anomaly_score !== undefined && (
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-slate-400">Isolation Anomaly Index:</span>
                    <span className="font-mono text-cyan-400">{report.components.anomaly_score} / 100</span>
                  </div>
                )}
                <div className="text-[11px] text-slate-400 pt-1 border-t border-navy-800">
                  <span className="font-medium text-slate-300">Derivation: </span>
                  {report.formula_explanation}
                </div>
              </div>

              {/* Genuine Model Feature Contributions */}
              {report.model_attributions && report.model_attributions.length > 0 && (
                <div>
                  <span className="text-[11px] font-bold text-slate-300 uppercase tracking-wider block mb-2.5">
                    Model Feature Attributions ({report.active_algorithm})
                  </span>
                  <div className="space-y-2">
                    {report.model_attributions.map((att: any, idx: number) => (
                      <div 
                        key={idx}
                        className="p-2.5 bg-navy-900/70 border border-navy-700/60 rounded-xl text-xs"
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-medium text-white">{att.feature}</span>
                          <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded ${
                            att.direction === 'increases_risk'
                              ? 'bg-red-500/15 text-red-400'
                              : 'bg-emerald-500/15 text-emerald-400'
                          }`}>
                            {att.direction === 'increases_risk' ? '▲ Increases Risk' : '▼ Lowers Risk'} ({att.contribution_pct}%)
                          </span>
                        </div>
                        <div className="w-full bg-navy-800 h-1 rounded-full overflow-hidden">
                          <div 
                            className={`h-full ${att.direction === 'increases_risk' ? 'bg-red-400' : 'bg-emerald-400'}`}
                            style={{ width: `${Math.min(100, att.contribution_pct * 2.5)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Forensic Security Rule Warnings */}
              {report.rule_warnings && report.rule_warnings.length > 0 && (
                <div>
                  <span className="text-[11px] font-bold text-slate-300 uppercase tracking-wider block mb-2.5 flex items-center gap-1.5 text-amber-400">
                    <ShieldAlert className="w-3.5 h-3.5" />
                    <span>Forensic Security Warnings ({report.rule_warnings.length})</span>
                  </span>
                  <div className="space-y-2">
                    {report.rule_warnings.map((warn: any, idx: number) => (
                      <div 
                        key={idx}
                        className="p-3 bg-navy-900/70 border border-navy-700/60 rounded-xl flex items-start gap-2.5 text-xs"
                      >
                        <span className={`w-2 h-2 rounded-full mt-1.5 shrink-0 ${
                          warn.severity === 'CRITICAL' ? 'bg-red-400' : warn.severity === 'HIGH' ? 'bg-amber-400' : 'bg-emerald-400'
                        }`} />
                        <div className="flex-1">
                          <div className="flex items-center justify-between">
                            <span className="font-semibold text-white">{warn.factor}</span>
                            <span className="text-[10px] font-mono text-slate-400">{warn.severity}</span>
                          </div>
                          <p className="text-slate-400 text-[11px] mt-0.5">{warn.message}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Executive Narrative */}
              {report.narrative && (
                <div className="p-3 bg-navy-900/40 border border-navy-700/40 rounded-xl text-xs text-slate-300 leading-relaxed">
                  <span className="font-bold text-white block mb-0.5">Forensic Executive Summary:</span>
                  {report.narrative}
                </div>
              )}
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
