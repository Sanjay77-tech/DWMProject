import React, { useState, useEffect, useRef } from 'react';
import { 
  Cpu, 
  Play, 
  CheckCircle2, 
  Layers, 
  Terminal, 
  Sliders, 
  Trophy,
  AlertOctagon,
  Info
} from 'lucide-react';
import { api } from '../api/client';

export const ModelTrainingPage: React.FC = () => {
  const [paradigm, setParadigm] = useState<string>("Supervised Classification (Labelled)");
  const [algorithm, setAlgorithm] = useState<string>("Random Forest");
  
  // Hyperparameters
  const [rfEstimators, setRfEstimators] = useState<number>(100);
  const [rfDepth, setRfDepth] = useState<number>(10);
  const [gbEstimators, setGbEstimators] = useState<number>(80);
  const [gbRate, setGbRate] = useState<number>(0.1);
  const [lrC, setLrC] = useState<number>(1.0);
  const [isoContamination, setIsoContamination] = useState<number>(0.045);
  const [ocNu, setOcNu] = useState<number>(0.05);
  const [stRatio, setStRatio] = useState<number>(0.15);
  const [stThreshold, setStThreshold] = useState<number>(0.75);
  const [lsGamma, setLsGamma] = useState<number>(20);
  const [kClusters, setKClusters] = useState<number>(3);

  // Job execution state
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string>('idle');
  const [jobProgress, setJobProgress] = useState<number>(0);
  const [jobLogs, setJobLogs] = useState<string[]>([]);
  const [jobResult, setJobResult] = useState<any | null>(null);
  const [jobError, setJobError] = useState<string | null>(null);

  // Benchmark suite state
  const [benchmarkLoading, setBenchmarkLoading] = useState(false);
  const [benchmarkResults, setBenchmarkResults] = useState<any[] | null>(null);
  const [benchmarkNotice, setBenchmarkNotice] = useState<string | null>(null);

  const logsEndRef = useRef<HTMLDivElement>(null);

  // Algorithm options map
  const paradigmAlgorithms: Record<string, string[]> = {
    "Supervised Classification (Labelled)": [
      "Random Forest",
      "Gradient Boosting",
      "Logistic Regression"
    ],
    "Semi-Supervised Learning (Partial Labels + Pseudo-Labelling)": [
      "Self-Training Classifier",
      "Label Spreading",
      "Label Propagation"
    ],
    "Anomaly Detection (One-Class Semi-Supervised)": [
      "Isolation Forest",
      "One-Class SVM"
    ],
    "Clustering (Unsupervised Discovery)": [
      "K-Means Clustering"
    ]
  };

  const handleParadigmChange = (newP: string) => {
    setParadigm(newP);
    const algos = paradigmAlgorithms[newP];
    if (algos && algos.length > 0) {
      setAlgorithm(algos[0]);
    }
  };

  // Scroll terminal logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [jobLogs]);

  // Polling hook
  useEffect(() => {
    if (!jobId || jobStatus === 'completed' || jobStatus === 'failed') return;

    let failCount = 0;
    const pollStatus = async () => {
      try {
        const statusData = await api.getTrainingStatus(jobId);
        failCount = 0;
        setJobStatus(statusData.status);
        setJobProgress(statusData.progress || 0);
        setJobLogs(statusData.logs || []);
        if (statusData.status === 'completed') {
          setJobResult(statusData.result);
        } else if (statusData.status === 'failed') {
          setJobError(statusData.error || "Training failed.");
        }
      } catch (err: any) {
        failCount++;
        console.error("Polling error", err);
        if (failCount >= 3) {
          setJobStatus('failed');
          setJobError(err.message || "Failed to communicate with training executor.");
        }
      }
    };

    pollStatus();
    const interval = setInterval(pollStatus, 1000);

    return () => clearInterval(interval);
  }, [jobId, jobStatus]);

  const handleStartTraining = async () => {
    setJobStatus('queued');
    setJobProgress(5);
    setJobLogs([`Preparing training job for ${algorithm}...`]);
    setJobResult(null);
    setJobError(null);

    // Build hyperparameter payload
    const hyperparams: Record<string, any> = {};
    if (algorithm === "Random Forest") {
      hyperparams.n_estimators = rfEstimators;
      hyperparams.max_depth = rfDepth;
    } else if (algorithm === "Gradient Boosting") {
      hyperparams.n_estimators = gbEstimators;
      hyperparams.learning_rate = gbRate;
    } else if (algorithm === "Logistic Regression") {
      hyperparams.C = lrC;
    } else if (algorithm === "Isolation Forest") {
      hyperparams.contamination = isoContamination;
      hyperparams.n_estimators = 100;
    } else if (algorithm === "One-Class SVM") {
      hyperparams.nu = ocNu;
    } else if (algorithm === "Self-Training Classifier") {
      hyperparams.labeled_ratio = stRatio;
      hyperparams.threshold = stThreshold;
    } else if (algorithm === "Label Spreading" || algorithm === "Label Propagation") {
      hyperparams.labeled_ratio = stRatio;
      hyperparams.gamma = lsGamma;
    } else if (algorithm === "K-Means Clustering") {
      hyperparams.n_clusters = kClusters;
    }

    try {
      const resp = await api.startTraining(paradigm, algorithm, hyperparams);
      setJobId(resp.job_id);
    } catch (err: any) {
      setJobStatus('failed');
      setJobError(err.message || "Failed to start training.");
      setJobLogs(prev => [...prev, `ERROR: ${err.message}`]);
    }
  };

  const handleRunBenchmarkSuite = async () => {
    setBenchmarkLoading(true);
    setBenchmarkNotice(null);
    try {
      const res = await api.runBenchmarkSuite();
      setBenchmarkResults(res.benchmark_table || res.comparison || []);
      if (res.notice_message) {
        setBenchmarkNotice(res.notice_message);
      }
    } catch (err: any) {
      alert(`Benchmark Suite Error: ${err.message}`);
    } finally {
      setBenchmarkLoading(false);
    }
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-xs font-semibold text-emerald-400 uppercase tracking-wider mb-1">
          <Cpu className="w-3.5 h-3.5" />
          <span>Machine Learning Studio</span>
        </div>
        <h1 className="text-2xl font-bold text-white tracking-tight">Multi-Paradigm Model Training Studio</h1>
        <p className="text-slate-400 text-sm mt-1 max-w-3xl">
          Train and evaluate machine learning models across Supervised, Semi-Supervised, Anomaly Detection, and Clustering paradigms with strict chronological anti-leakage validation.
        </p>
      </div>

      {/* Configuration & Sliders */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Paradigm & Algorithm Selector */}
        <div className="lg:col-span-5 bg-navy-800/80 border border-navy-700 rounded-2xl p-6 space-y-5">
          <div className="flex items-center gap-2 text-xs font-bold text-slate-300 uppercase tracking-wider">
            <Layers className="w-4 h-4 text-emerald-400" />
            <span>1. Paradigm & Model Architecture</span>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-2">Learning Paradigm</label>
            <div className="space-y-2">
              {Object.keys(paradigmAlgorithms).map((pKey) => (
                <button
                  key={pKey}
                  type="button"
                  onClick={() => handleParadigmChange(pKey)}
                  className={`w-full text-left p-3 rounded-xl border text-xs font-medium transition-all ${
                    paradigm === pKey
                      ? 'bg-emerald-500/10 border-emerald-500 text-white font-semibold'
                      : 'bg-navy-900/60 border-navy-700/60 text-slate-400 hover:text-white hover:bg-navy-900'
                  }`}
                >
                  {pKey}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-2">Target Algorithm</label>
            <div className="grid grid-cols-1 gap-2">
              {paradigmAlgorithms[paradigm]?.map((algoName) => (
                <button
                  key={algoName}
                  type="button"
                  onClick={() => setAlgorithm(algoName)}
                  className={`p-2.5 rounded-lg border text-xs font-mono transition-all text-left flex items-center justify-between ${
                    algorithm === algoName
                      ? 'bg-navy-700 border-cyan-500 text-cyan-300 font-bold'
                      : 'bg-navy-900/80 border-navy-700/80 text-slate-300 hover:text-white'
                  }`}
                >
                  <span>{algoName}</span>
                  {algorithm === algoName && <CheckCircle2 className="w-3.5 h-3.5 text-cyan-400" />}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Hyperparameter Tuner & Action */}
        <div className="lg:col-span-7 bg-navy-800/80 border border-navy-700 rounded-2xl p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2 text-xs font-bold text-slate-300 uppercase tracking-wider">
                <Sliders className="w-4 h-4 text-emerald-400" />
                <span>2. Hyperparameter Configuration</span>
              </div>
              <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                {algorithm}
              </span>
            </div>

            {/* Dynamic Sliders depending on selected algorithm */}
            <div className="space-y-4 bg-navy-900/60 p-4 rounded-xl border border-navy-700/60">
              {algorithm === "Random Forest" && (
                <>
                  <div>
                    <div className="flex justify-between text-xs text-slate-300 mb-1">
                      <span>Number of Estimators (Trees):</span>
                      <span className="font-mono text-white font-bold">{rfEstimators}</span>
                    </div>
                    <input
                      type="range"
                      min="20"
                      max="300"
                      step="10"
                      value={rfEstimators}
                      onChange={(e) => setRfEstimators(parseInt(e.target.value))}
                      className="w-full accent-emerald-500"
                    />
                  </div>
                  <div>
                    <div className="flex justify-between text-xs text-slate-300 mb-1">
                      <span>Maximum Depth:</span>
                      <span className="font-mono text-white font-bold">{rfDepth}</span>
                    </div>
                    <input
                      type="range"
                      min="3"
                      max="25"
                      step="1"
                      value={rfDepth}
                      onChange={(e) => setRfDepth(parseInt(e.target.value))}
                      className="w-full accent-emerald-500"
                    />
                  </div>
                </>
              )}

              {algorithm === "Gradient Boosting" && (
                <>
                  <div>
                    <div className="flex justify-between text-xs text-slate-300 mb-1">
                      <span>Number of Boosting Stages:</span>
                      <span className="font-mono text-white font-bold">{gbEstimators}</span>
                    </div>
                    <input
                      type="range"
                      min="20"
                      max="200"
                      step="10"
                      value={gbEstimators}
                      onChange={(e) => setGbEstimators(parseInt(e.target.value))}
                      className="w-full accent-emerald-500"
                    />
                  </div>
                  <div>
                    <div className="flex justify-between text-xs text-slate-300 mb-1">
                      <span>Learning Rate (Shrinkage):</span>
                      <span className="font-mono text-white font-bold">{gbRate}</span>
                    </div>
                    <input
                      type="range"
                      min="0.01"
                      max="0.5"
                      step="0.01"
                      value={gbRate}
                      onChange={(e) => setGbRate(parseFloat(e.target.value))}
                      className="w-full accent-emerald-500"
                    />
                  </div>
                </>
              )}

              {algorithm === "Logistic Regression" && (
                <div>
                  <div className="flex justify-between text-xs text-slate-300 mb-1">
                    <span>Inverse Regularization Strength (C):</span>
                    <span className="font-mono text-white font-bold">{lrC}</span>
                  </div>
                  <input
                    type="range"
                    min="0.05"
                    max="10.0"
                    step="0.05"
                    value={lrC}
                    onChange={(e) => setLrC(parseFloat(e.target.value))}
                    className="w-full accent-emerald-500"
                  />
                </div>
              )}

              {algorithm === "Isolation Forest" && (
                <div>
                  <div className="flex justify-between text-xs text-slate-300 mb-1">
                    <span>Assumed Contamination Rate:</span>
                    <span className="font-mono text-white font-bold">{(isoContamination * 100).toFixed(1)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0.01"
                    max="0.15"
                    step="0.005"
                    value={isoContamination}
                    onChange={(e) => setIsoContamination(parseFloat(e.target.value))}
                    className="w-full accent-emerald-500"
                  />
                </div>
              )}

              {algorithm === "One-Class SVM" && (
                <div>
                  <div className="flex justify-between text-xs text-slate-300 mb-1">
                    <span>Outlier Fraction (nu):</span>
                    <span className="font-mono text-white font-bold">{ocNu}</span>
                  </div>
                  <input
                    type="range"
                    min="0.01"
                    max="0.15"
                    step="0.01"
                    value={ocNu}
                    onChange={(e) => setOcNu(parseFloat(e.target.value))}
                    className="w-full accent-emerald-500"
                  />
                </div>
              )}

              {algorithm === "Self-Training Classifier" && (
                <>
                  <div>
                    <div className="flex justify-between text-xs text-slate-300 mb-1">
                      <span>Ground-Truth Labelled Ratio:</span>
                      <span className="font-mono text-white font-bold">{(stRatio * 100).toFixed(0)}%</span>
                    </div>
                    <input
                      type="range"
                      min="0.05"
                      max="0.40"
                      step="0.05"
                      value={stRatio}
                      onChange={(e) => setStRatio(parseFloat(e.target.value))}
                      className="w-full accent-emerald-500"
                    />
                  </div>
                  <div>
                    <div className="flex justify-between text-xs text-slate-300 mb-1">
                      <span>Pseudo-Labelling Confidence Threshold:</span>
                      <span className="font-mono text-white font-bold">{(stThreshold * 100).toFixed(0)}%</span>
                    </div>
                    <input
                      type="range"
                      min="0.60"
                      max="0.95"
                      step="0.05"
                      value={stThreshold}
                      onChange={(e) => setStThreshold(parseFloat(e.target.value))}
                      className="w-full accent-emerald-500"
                    />
                  </div>
                </>
              )}

              {(algorithm === "Label Spreading" || algorithm === "Label Propagation") && (
                <>
                  <div>
                    <div className="flex justify-between text-xs text-slate-300 mb-1">
                      <span>Initial Labelled Ratio:</span>
                      <span className="font-mono text-white font-bold">{(stRatio * 100).toFixed(0)}%</span>
                    </div>
                    <input
                      type="range"
                      min="0.05"
                      max="0.40"
                      step="0.05"
                      value={stRatio}
                      onChange={(e) => setStRatio(parseFloat(e.target.value))}
                      className="w-full accent-emerald-500"
                    />
                  </div>
                  <div>
                    <div className="flex justify-between text-xs text-slate-300 mb-1">
                      <span>RBF Kernel Gamma:</span>
                      <span className="font-mono text-white font-bold">{lsGamma}</span>
                    </div>
                    <input
                      type="range"
                      min="5"
                      max="50"
                      step="5"
                      value={lsGamma}
                      onChange={(e) => setLsGamma(parseInt(e.target.value))}
                      className="w-full accent-emerald-500"
                    />
                  </div>
                </>
              )}

              {algorithm === "K-Means Clustering" && (
                <div>
                  <div className="flex justify-between text-xs text-slate-300 mb-1">
                    <span>Number of Clusters (k):</span>
                    <span className="font-mono text-white font-bold">{kClusters}</span>
                  </div>
                  <input
                    type="range"
                    min="2"
                    max="6"
                    step="1"
                    value={kClusters}
                    onChange={(e) => setKClusters(parseInt(e.target.value))}
                    className="w-full accent-emerald-500"
                  />
                </div>
              )}
            </div>
          </div>

          {/* Action Trigger Buttons */}
          <div className="mt-6 pt-4 border-t border-navy-700 flex flex-col sm:flex-row gap-3">
            <button
              onClick={handleStartTraining}
              disabled={jobStatus === 'running'}
              className="flex-1 py-3 px-4 bg-emerald-500 hover:bg-emerald-400 text-navy-950 font-bold text-xs rounded-xl transition-all shadow-lg shadow-emerald-500/20 flex items-center justify-center gap-2"
            >
              {jobStatus === 'running' ? (
                <>
                  <div className="w-4 h-4 border-2 border-navy-950 border-t-transparent rounded-full animate-spin" />
                  <span>Training {algorithm}... ({jobProgress}%)</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-navy-950" />
                  <span>Launch Training Job</span>
                </>
              )}
            </button>

            <button
              onClick={handleRunBenchmarkSuite}
              disabled={benchmarkLoading}
              className="py-3 px-4 bg-navy-700 hover:bg-navy-600 border border-navy-600 text-white font-semibold text-xs rounded-xl transition-all flex items-center justify-center gap-2"
            >
              {benchmarkLoading ? (
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : (
                <Trophy className="w-4 h-4 text-amber-400" />
              )}
              <span>Benchmark All Models</span>
            </button>
          </div>
        </div>
      </div>

      {/* Error Banner if any */}
      {jobError && (
        <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 flex items-start gap-3 text-red-300 text-xs animate-fadeIn">
          <AlertOctagon className="w-4 h-4 shrink-0 mt-0.5 text-red-400" />
          <div>
            <span className="font-semibold text-red-200">Execution Error: </span>
            {jobError}
          </div>
        </div>
      )}

      {/* Real-Time Terminal Execution Console & Progress */}
      {(jobStatus !== 'idle' || jobLogs.length > 0) && (
        <div className="bg-navy-900 border border-navy-700 rounded-2xl overflow-hidden shadow-2xl animate-fadeIn">
          {/* Terminal Title Bar */}
          <div className="px-4 py-3 bg-navy-950 border-b border-navy-800 flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-mono text-slate-300">
              <Terminal className="w-4 h-4 text-emerald-400" />
              <span>Training Job Executor {jobId ? `[${jobId}]` : ''}</span>
            </div>
            <div className="flex items-center gap-3">
              <span className={`text-[10px] font-mono px-2 py-0.5 rounded ${
                jobStatus === 'completed'
                  ? 'bg-emerald-500/20 text-emerald-400'
                  : jobStatus === 'failed'
                  ? 'bg-red-500/20 text-red-400'
                  : 'bg-amber-500/20 text-amber-400'
              }`}>
                STATUS: {jobStatus.toUpperCase()}
              </span>
            </div>
          </div>

          {/* Animated Progress Bar */}
          <div className="w-full bg-navy-800 h-2">
            <div 
              className={`h-full transition-all duration-500 ease-out ${
                jobStatus === 'failed' ? 'bg-red-500' : 'bg-emerald-400'
              }`}
              style={{ width: `${jobProgress}%` }}
            />
          </div>

          {/* Terminal Console Output */}
          <div className="p-4 font-mono text-xs text-slate-300 max-h-52 overflow-y-auto space-y-1 bg-navy-950/80">
            {jobLogs.map((log, idx) => (
              <div key={idx} className="flex items-start gap-2">
                <span className="text-slate-500 text-[10px] select-none">&gt;</span>
                <span className={log.startsWith('ERROR') ? 'text-red-400' : 'text-slate-300'}>{log}</span>
              </div>
            ))}
            <div ref={logsEndRef} />
          </div>
        </div>
      )}

      {/* Model Evaluation Results Section (Visible on Completion) */}
      {jobResult && (
        <div className="space-y-6 animate-fadeIn">
          {/* Single Class / Small Sample Notice Banner */}
          {(jobResult.single_class_notice || (jobResult.metrics && jobResult.metrics.true_positives === 0 && jobResult.metrics.false_negatives === 0)) && (
            <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-start gap-3 text-amber-200 text-xs">
              <Info className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <div className="font-semibold text-amber-300">Why are all metrics 100% or 0%? (Single-Class Holdout Notice)</div>
                <p className="text-slate-300 leading-relaxed">
                  In this dataset, the chronological holdout test window (the final 15% of transactions) contains 0 fraud instances because all fraud attempts occurred earlier in the timeline. All 5 test transactions are legitimate, so every model correctly cleared them (100% Accuracy, 5 True Negatives).
                </p>
                <p className="text-amber-400/90 font-medium pt-1">
                  💡 Recommendation: To evaluate multi-model trade-offs, false alarms, and ROC curves across both classes, load the 7,500-sample UPI Benchmark Dataset from the Dataset Upload page.
                </p>
              </div>
            </div>
          )}

          {/* Metrics Strip */}
          {jobResult.metrics && (
            <div>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-bold text-white tracking-tight flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  <span>Validation Metrics: {jobResult.algorithm} (Holdout Test Split)</span>
                </h3>
                <span className="text-xs text-slate-400 font-mono">
                  Finished in {jobResult.elapsed_seconds}s
                </span>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                <div className="bg-navy-800/80 border border-navy-700/80 p-4 rounded-xl">
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Accuracy</span>
                  <span className="text-xl font-bold text-white mt-1 block">
                    {(jobResult.metrics.accuracy * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="bg-navy-800/80 border border-navy-700/80 p-4 rounded-xl">
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Precision</span>
                  <span className="text-xl font-bold text-emerald-400 mt-1 block">
                    {(jobResult.metrics.precision * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="bg-navy-800/80 border border-navy-700/80 p-4 rounded-xl">
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Recall</span>
                  <span className="text-xl font-bold text-cyan-400 mt-1 block">
                    {(jobResult.metrics.recall * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="bg-navy-800/80 border border-navy-700/80 p-4 rounded-xl">
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">F1-Score</span>
                  <span className="text-xl font-bold text-amber-400 mt-1 block">
                    {(jobResult.metrics.f1_score * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="bg-navy-800/80 border border-navy-700/80 p-4 rounded-xl">
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">ROC-AUC</span>
                  <span className="text-xl font-bold text-white mt-1 block">
                    {jobResult.metrics.roc_auc !== null ? `${(jobResult.metrics.roc_auc * 100).toFixed(2)}%` : 'N/A'}
                  </span>
                </div>
              </div>

              {/* Real 2x2 Confusion Matrix Heatmap */}
              {jobResult.metrics.confusion_matrix && (
                <div className="mt-6 bg-navy-800/80 border border-navy-700 rounded-2xl p-6">
                  <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-1">
                    Holdout Test Confusion Matrix Heatmap
                  </h4>
                  <p className="text-xs text-slate-400 mb-4">
                    Strict binary classification matrix across true legitimate (0) and fraud (1) samples.
                  </p>

                  <div className="max-w-md mx-auto grid grid-cols-2 gap-3 text-center">
                    <div className="bg-navy-900/90 border border-emerald-500/30 p-4 rounded-xl">
                      <span className="text-[10px] uppercase font-bold text-slate-400 block mb-1">True Negatives (TN)</span>
                      <span className="text-2xl font-black text-emerald-400 font-mono">
                        {jobResult.metrics.confusion_matrix[0][0]?.toLocaleString()}
                      </span>
                      <span className="text-[10px] text-slate-500 block mt-1">Correctly Cleared Normal</span>
                    </div>

                    <div className="bg-navy-900/90 border border-amber-500/30 p-4 rounded-xl">
                      <span className="text-[10px] uppercase font-bold text-slate-400 block mb-1">False Positives (FP)</span>
                      <span className="text-2xl font-black text-amber-400 font-mono">
                        {jobResult.metrics.confusion_matrix[0][1]?.toLocaleString()}
                      </span>
                      <span className="text-[10px] text-slate-500 block mt-1">False Alarms</span>
                    </div>

                    <div className="bg-navy-900/90 border border-red-500/30 p-4 rounded-xl">
                      <span className="text-[10px] uppercase font-bold text-slate-400 block mb-1">False Negatives (FN)</span>
                      <span className="text-2xl font-black text-red-400 font-mono">
                        {jobResult.metrics.confusion_matrix[1][0]?.toLocaleString()}
                      </span>
                      <span className="text-[10px] text-slate-500 block mt-1">Missed Fraud Attacks</span>
                    </div>

                    <div className="bg-navy-900/90 border border-emerald-500/30 p-4 rounded-xl">
                      <span className="text-[10px] uppercase font-bold text-slate-400 block mb-1">True Positives (TP)</span>
                      <span className="text-2xl font-black text-emerald-400 font-mono">
                        {jobResult.metrics.confusion_matrix[1][1]?.toLocaleString()}
                      </span>
                      <span className="text-[10px] text-slate-500 block mt-1">Successfully Intercepted</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Semi-Supervised Stats Card (if applicable) */}
          {jobResult.semi_supervised_stats && (
            <div className="bg-navy-800/80 border border-navy-700 rounded-2xl p-6">
              <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">
                Semi-Supervised Pseudo-Labelling Propagation Metrics
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                <div className="bg-navy-900/60 p-3 rounded-xl border border-navy-700/60">
                  <span className="text-slate-400 text-[10px] block">Initial Labelled Ratio</span>
                  <span className="font-mono text-white font-bold text-sm">{(jobResult.semi_supervised_stats.labeled_ratio * 100).toFixed(0)}%</span>
                </div>
                <div className="bg-navy-900/60 p-3 rounded-xl border border-navy-700/60">
                  <span className="text-slate-400 text-[10px] block">Initially Labelled Samples</span>
                  <span className="font-mono text-white font-bold text-sm">{jobResult.semi_supervised_stats.n_labeled?.toLocaleString()}</span>
                </div>
                <div className="bg-navy-900/60 p-3 rounded-xl border border-navy-700/60">
                  <span className="text-slate-400 text-[10px] block">Unlabeled Pool</span>
                  <span className="font-mono text-white font-bold text-sm">{jobResult.semi_supervised_stats.n_unlabeled?.toLocaleString()}</span>
                </div>
                <div className="bg-navy-900/60 p-3 rounded-xl border border-navy-700/60">
                  <span className="text-slate-400 text-[10px] block">Pseudo-Labels Inferred</span>
                  <span className="font-mono text-emerald-400 font-bold text-sm">{jobResult.semi_supervised_stats.pseudo_labeled_count?.toLocaleString() || 'All Propagated'}</span>
                </div>
              </div>
            </div>
          )}

          {/* K-Means Cluster Breakdown */}
          {jobResult.cluster_summary && (
            <div className="bg-navy-800/80 border border-navy-700 rounded-2xl p-6">
              <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-3">
                K-Means Unsupervised Behavioral Clusters
              </h4>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                {jobResult.cluster_summary.map((cl: any) => (
                  <div key={cl.cluster_id} className="bg-navy-900/70 p-4 rounded-xl border border-navy-700/60 text-xs">
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-bold text-white">Cluster #{cl.cluster_id}</span>
                      <span className="text-slate-400 font-mono text-[11px]">{cl.count?.toLocaleString()} txns</span>
                    </div>
                    <div className="text-slate-400">Avg Amount: <span className="font-mono text-white">₹{cl.avg_amount?.toLocaleString()}</span></div>
                    <div className="text-slate-400 mt-1">Empirical Fraud Rate: <span className="font-mono text-amber-400">{cl.fraud_rate?.toFixed(2)}%</span></div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Multi-Model Benchmark Comparison Table */}
      {benchmarkResults && benchmarkResults.length > 0 && (
        <div className="bg-navy-800/80 border border-navy-700 rounded-2xl p-6 animate-fadeIn">
          <div className="flex items-center gap-2 mb-1">
            <Trophy className="w-4 h-4 text-amber-400" />
            <h3 className="text-sm font-bold text-white tracking-tight">
              Multi-Model Benchmark Suite Leaderboard
            </h3>
          </div>
          <p className="text-xs text-slate-400 mb-4">
            Side-by-side empirical performance across all algorithms on identical chronological holdout data.
          </p>

          {benchmarkNotice && (
            <div className="mb-4 p-3 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-start gap-2.5 text-amber-200 text-xs">
              <Info className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <span>{benchmarkNotice}</span>
            </div>
          )}

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-navy-900/80 text-slate-400 uppercase text-[10px] font-semibold border-b border-navy-700">
                <tr>
                  <th className="py-2.5 px-4">Algorithm / Paradigm</th>
                  <th className="py-2.5 px-4">F1-Score</th>
                  <th className="py-2.5 px-4">Recall</th>
                  <th className="py-2.5 px-4">Precision</th>
                  <th className="py-2.5 px-4">ROC-AUC</th>
                  <th className="py-2.5 px-4">Accuracy</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-navy-700/60 font-mono text-[11px]">
                {benchmarkResults.map((m: any, idx: number) => {
                  const modelName = m.model || m.Model || m['index'] || 'Algorithm';
                  const f1 = m.f1_score ?? m['F1-Score'] ?? 0;
                  const rec = m.recall ?? m['Recall'] ?? 0;
                  const prec = m.precision ?? m['Precision'] ?? 0;
                  const roc = m.roc_auc ?? m['ROC-AUC'];
                  const acc = m.accuracy ?? m['Accuracy'] ?? 0;
                  return (
                    <tr key={idx} className="hover:bg-navy-700/30 transition-colors">
                      <td className="py-3 px-4 font-sans font-medium text-white">{modelName}</td>
                      <td className="py-3 px-4 text-emerald-400 font-bold">{(f1 * 100).toFixed(2)}%</td>
                      <td className="py-3 px-4 text-cyan-400">{(rec * 100).toFixed(2)}%</td>
                      <td className="py-3 px-4 text-slate-300">{(prec * 100).toFixed(2)}%</td>
                      <td className="py-3 px-4 text-slate-300">{roc !== null && roc !== undefined ? `${(roc * 100).toFixed(2)}%` : 'N/A'}</td>
                      <td className="py-3 px-4 text-slate-400">{(acc * 100).toFixed(2)}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
