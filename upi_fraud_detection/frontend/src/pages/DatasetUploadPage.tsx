import React, { useState, useRef } from 'react';
import { 
  UploadCloud, 
  FileSpreadsheet, 
  CheckCircle2, 
  AlertTriangle, 
  ArrowRight, 
  Download, 
  Database,
  Layers,
  Sparkles,
  HelpCircle,
  Hash
} from 'lucide-react';
import { api, type DatasetStatus, type InspectionData } from '../api/client';

interface DatasetUploadPageProps {
  datasetStatus: DatasetStatus | null;
  onDatasetUpdated: (status: DatasetStatus) => void;
  onNavigateToDashboard: () => void;
}

export const DatasetUploadPage: React.FC<DatasetUploadPageProps> = ({
  datasetStatus,
  onDatasetUpdated,
  onNavigateToDashboard
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      await processFileUpload(files[0]);
    }
  };

  const handleFileInputChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      await processFileUpload(e.target.files[0]);
    }
  };

  const processFileUpload = async (file: File) => {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setErrorMessage("Please select a valid CSV (.csv) file format.");
      return;
    }

    setUploading(true);
    setErrorMessage(null);
    setSuccessMessage(null);

    try {
      const result = await api.uploadDataset(file);
      const updatedStatus = await api.getDatasetStatus();
      onDatasetUpdated(updatedStatus);
      setSuccessMessage(`Successfully ingested "${file.name}" (${result.inspection.total_rows.toLocaleString()} rows).`);
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to process the uploaded dataset.");
    } finally {
      setUploading(false);
    }
  };

  const handleLoadBenchmark = async () => {
    setUploading(true);
    setErrorMessage(null);
    setSuccessMessage(null);
    try {
      await api.loadBenchmark();
      const updatedStatus = await api.getDatasetStatus();
      onDatasetUpdated(updatedStatus);
      setSuccessMessage("Standard 5,000-transaction UPI benchmark dataset loaded successfully.");
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to load benchmark dataset.");
    } finally {
      setUploading(false);
    }
  };

  const inspection: InspectionData | undefined = datasetStatus?.inspection;

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Page Title & Intro */}
      <div>
        <div className="flex items-center gap-2 text-xs font-semibold text-emerald-400 uppercase tracking-wider mb-1">
          <Database className="w-3.5 h-3.5" />
          <span>Data Ingestion & Schema Adaptation</span>
        </div>
        <h1 className="text-2xl font-bold text-white tracking-tight">Dataset Ingestion & Inspection</h1>
        <p className="text-slate-400 text-sm mt-1 max-w-3xl">
          Upload your UPI or financial transaction CSV file to calibrate feature pipelines, train machine learning detectors, and inspect fraud vectors in real time.
        </p>
      </div>

      {/* Upload & Benchmark Zone */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Drag-and-Drop Area */}
        <div className="lg:col-span-2">
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-200 flex flex-col items-center justify-center min-h-[260px] ${
              isDragging 
                ? 'border-emerald-500 bg-emerald-500/10' 
                : 'border-navy-600 bg-navy-800/60 hover:border-navy-500 hover:bg-navy-800'
            }`}
          >
            <input 
              ref={fileInputRef} 
              type="file" 
              accept=".csv" 
              onChange={handleFileInputChange} 
              className="hidden" 
            />

            <div className="w-14 h-14 rounded-2xl bg-navy-700 border border-navy-600 flex items-center justify-center text-emerald-400 mb-4 shadow-lg">
              {uploading ? (
                <div className="w-7 h-7 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin" />
              ) : (
                <UploadCloud className="w-7 h-7" />
              )}
            </div>

            <h3 className="text-base font-semibold text-white mb-1">
              {uploading ? "Ingesting and Inspecting Dataset..." : "Drag & Drop CSV Dataset Here"}
            </h3>
            <p className="text-xs text-slate-400 max-w-md mb-4">
              Supports standard PaySim exports or native UPI Sentinel transaction logs up to 100MB.
            </p>

            <button
              type="button"
              disabled={uploading}
              className="px-4 py-2 bg-emerald-500 hover:bg-emerald-400 text-navy-950 font-semibold text-xs rounded-lg transition-all flex items-center gap-2 shadow-md shadow-emerald-500/20"
            >
              <FileSpreadsheet className="w-4 h-4" />
              <span>Browse CSV Files</span>
            </button>
          </div>

          {/* Feedback Banners */}
          {errorMessage && (
            <div className="mt-4 p-4 rounded-xl bg-red-500/10 border border-red-500/30 flex items-start gap-3 text-red-300 text-xs animate-fadeIn">
              <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5 text-red-400" />
              <div>
                <span className="font-semibold text-red-200">Upload Rejected: </span>
                {errorMessage}
              </div>
            </div>
          )}

          {successMessage && (
            <div className="mt-4 p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-start gap-3 text-emerald-300 text-xs animate-fadeIn">
              <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5 text-emerald-400" />
              <div>
                <span className="font-semibold text-emerald-200">Dataset Active: </span>
                {successMessage}
              </div>
            </div>
          )}
        </div>

        {/* Quick Launch & Sample Card */}
        <div className="bg-navy-800/80 border border-navy-700 rounded-2xl p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-300 mb-2">
              <Sparkles className="w-4 h-4 text-emerald-400" />
              <span>Quick Test & Demonstration</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed mb-4">
              Don't have your own CSV ready? You can instantly load the benchmark dataset with 5,000 realistic UPI transactions, pre-labeled fraud incidents, and time-stamped velocity indicators.
            </p>

            <button
              onClick={handleLoadBenchmark}
              disabled={uploading}
              className="w-full py-2.5 px-4 bg-navy-700 hover:bg-navy-600 border border-navy-600 rounded-xl text-xs font-semibold text-white transition-all flex items-center justify-center gap-2 mb-4"
            >
              <Database className="w-4 h-4 text-emerald-400" />
              <span>Load 5,000-Row Benchmark</span>
            </button>

            <div className="border-t border-navy-700/80 pt-4">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-2">
                Download Sample Templates
              </span>
              <div className="space-y-2">
                <a
                  href="/api/dataset/sample/benchmark"
                  download="upi_benchmark_sample.csv"
                  className="w-full flex items-center justify-between p-2.5 rounded-lg bg-navy-900/60 hover:bg-navy-900 border border-navy-700/60 text-xs text-slate-300 transition-colors"
                >
                  <span className="flex items-center gap-2">
                    <Download className="w-3.5 h-3.5 text-slate-400" />
                    <span>Benchmark CSV Template</span>
                  </span>
                  <span className="text-[10px] text-slate-500 font-mono">5k Rows</span>
                </a>
                <a
                  href="/api/dataset/sample/custom"
                  download="custom_sample_upload.csv"
                  className="w-full flex items-center justify-between p-2.5 rounded-lg bg-navy-900/60 hover:bg-navy-900 border border-navy-700/60 text-xs text-slate-300 transition-colors"
                >
                  <span className="flex items-center gap-2">
                    <Download className="w-3.5 h-3.5 text-slate-400" />
                    <span>Minimal Test CSV</span>
                  </span>
                  <span className="text-[10px] text-slate-500 font-mono">50 Rows</span>
                </a>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-navy-700/60 flex items-center gap-2 text-[11px] text-slate-400">
            <HelpCircle className="w-3.5 h-3.5 text-slate-400" />
            <span>Automatic header mapping handles varied column namings.</span>
          </div>
        </div>
      </div>

      {/* Dataset Inspection Results (Visible once dataset is active) */}
      {datasetStatus?.is_loaded && inspection && (
        <div className="bg-navy-800/80 border border-navy-700 rounded-2xl p-6 space-y-6 animate-fadeIn">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-navy-700">
            <div>
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
                <h2 className="text-base font-bold text-white tracking-tight">
                  Dataset Inspection Summary: {inspection.filename || datasetStatus.source_label}
                </h2>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Schema parsed and validated. Ready for real-time risk inference and multi-model machine learning.
              </p>
            </div>

            <button
              onClick={onNavigateToDashboard}
              className="px-5 py-2.5 bg-emerald-500 hover:bg-emerald-400 text-navy-950 font-bold text-xs rounded-xl transition-all flex items-center gap-2 shadow-lg shadow-emerald-500/25 shrink-0"
            >
              <span>Explore Dashboard & Analytics</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>

          {/* Key Inspection Metrics Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
            <div className="bg-navy-900/60 border border-navy-700/60 p-3 rounded-xl">
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Total Records</span>
              <span className="text-lg font-bold text-white mt-1 block">
                {inspection.total_rows?.toLocaleString() ?? 0}
              </span>
            </div>

            <div className="bg-navy-900/60 border border-navy-700/60 p-3 rounded-xl">
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Total Columns</span>
              <span className="text-lg font-bold text-white mt-1 block">
                {inspection.total_columns ?? 0}
              </span>
            </div>

            <div className="bg-navy-900/60 border border-navy-700/60 p-3 rounded-xl">
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Fraud Labels</span>
              <span className={`text-xs font-bold mt-1.5 inline-flex items-center gap-1 ${
                inspection.has_labels ? 'text-emerald-400' : 'text-amber-400'
              }`}>
                {inspection.has_labels ? 'Verified Available' : 'Unlabeled Data'}
              </span>
            </div>

            <div className="bg-navy-900/60 border border-navy-700/60 p-3 rounded-xl">
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Fraud Cases</span>
              <span className="text-lg font-bold text-red-400 mt-1 block">
                {inspection.fraud_count?.toLocaleString() ?? 'N/A'}
              </span>
            </div>

            <div className="bg-navy-900/60 border border-navy-700/60 p-3 rounded-xl">
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Fraud Rate</span>
              <span className="text-lg font-bold text-amber-400 mt-1 block">
                {inspection.fraud_rate !== undefined ? `${inspection.fraud_rate.toFixed(2)}%` : 'N/A'}
              </span>
            </div>

            <div className="bg-navy-900/60 border border-navy-700/60 p-3 rounded-xl">
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Missing Values</span>
              <span className="text-lg font-bold text-white mt-1 block">
                {inspection.missing_values_total?.toLocaleString() ?? 0}
              </span>
            </div>

            <div className="bg-navy-900/60 border border-navy-700/60 p-3 rounded-xl">
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Duplicates</span>
              <span className="text-lg font-bold text-white mt-1 block">
                {inspection.duplicate_rows?.toLocaleString() ?? 0}
              </span>
            </div>
          </div>

          {/* Available Columns & Datatypes */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 pt-2">
            <div>
              <span className="text-xs font-semibold text-slate-300 block mb-2.5 flex items-center gap-1.5">
                <Hash className="w-3.5 h-3.5 text-emerald-400" />
                <span>Detected Columns ({inspection.raw_columns?.length || 0})</span>
              </span>
              <div className="flex flex-wrap gap-1.5 max-h-40 overflow-y-auto p-3 bg-navy-900/70 border border-navy-700/60 rounded-xl">
                {inspection.raw_columns?.map((col, idx) => (
                  <span 
                    key={idx}
                    className="px-2 py-1 bg-navy-800 text-slate-300 border border-navy-700 text-[11px] font-mono rounded"
                  >
                    {col}
                  </span>
                ))}
              </div>
            </div>

            <div>
              <span className="text-xs font-semibold text-slate-300 block mb-2.5 flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-emerald-400" />
                <span>Column Data Types</span>
              </span>
              <div className="grid grid-cols-2 gap-2 max-h-40 overflow-y-auto p-3 bg-navy-900/70 border border-navy-700/60 rounded-xl">
                {inspection.dtypes && Object.entries(inspection.dtypes).slice(0, 16).map(([col, dtype], idx) => (
                  <div key={idx} className="flex justify-between items-center text-[11px] p-1 bg-navy-800/80 rounded px-2">
                    <span className="text-slate-300 font-mono truncate max-w-[120px]">{col}</span>
                    <span className="text-emerald-400 font-mono text-[10px]">{dtype}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
