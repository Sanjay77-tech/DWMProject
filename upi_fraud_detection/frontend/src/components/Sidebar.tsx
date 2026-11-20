import React from 'react';
import { 
  UploadCloud, 
  LayoutDashboard, 
  SearchCheck, 
  Activity, 
  Cpu, 
  FileText, 
  ShieldCheck, 
  Lock
} from 'lucide-react';

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  datasetLoaded: boolean;
  datasetInfo?: {
    filename?: string;
    total_records?: number;
    source_label?: string;
  };
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  setActiveTab,
  datasetLoaded,
  datasetInfo
}) => {
  const navItems = [
    { id: 'upload', label: 'Dataset Upload', icon: UploadCloud, requiresData: false },
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard, requiresData: true },
    { id: 'prediction', label: 'Transaction Prediction', icon: SearchCheck, requiresData: false },
    { id: 'analytics', label: 'Fraud Analytics', icon: Activity, requiresData: true },
    { id: 'training', label: 'Model Training', icon: Cpu, requiresData: true },
    { id: 'history', label: 'Transaction History', icon: FileText, requiresData: false },
  ];

  return (
    <aside className="w-64 bg-navy-850 border-r border-navy-700 flex flex-col justify-between shrink-0 h-screen sticky top-0">
      <div>
        {/* Branding */}
        <div className="p-5 border-b border-navy-700">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-700 flex items-center justify-center shadow-lg shadow-emerald-500/20 text-white font-bold">
              <ShieldCheck className="w-6 h-6" />
            </div>
            <div>
              <h1 className="font-bold text-white tracking-tight text-base">UPI Sentinel AI</h1>
              <p className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider">Fintech Fraud Defense</p>
            </div>
          </div>
        </div>

        {/* Navigation Links */}
        <nav className="p-3 space-y-1 mt-2">
          <p className="px-3 py-1.5 text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
            Navigation Menu
          </p>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            const isLocked = item.requiresData && !datasetLoaded;

            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-lg text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 shadow-sm'
                    : 'text-slate-300 hover:text-white hover:bg-navy-800'
                }`}
              >
                <div className="flex items-center gap-3">
                  <Icon className={`w-4 h-4 ${isActive ? 'text-emerald-400' : 'text-slate-400'}`} />
                  <span>{item.label}</span>
                </div>
                {isLocked && (
                  <span className="flex items-center gap-1 text-[10px] text-amber-400/80 bg-amber-400/10 px-1.5 py-0.5 rounded border border-amber-400/20">
                    <Lock className="w-2.5 h-2.5" /> Required
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* System Monitor Panel */}
      <div className="p-4 border-t border-navy-700/80 bg-navy-900/60 m-3 rounded-xl border">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${datasetLoaded ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
            System Monitor
          </span>
          <span className="text-[10px] font-mono text-slate-400">v2.0 (FastAPI)</span>
        </div>

        <div className="space-y-1.5 text-xs text-slate-300">
          <div className="flex justify-between items-center text-[11px]">
            <span className="text-slate-400">Data Status:</span>
            <span className={datasetLoaded ? "text-emerald-400 font-medium" : "text-amber-400 font-medium"}>
              {datasetLoaded ? "Active Ingested" : "Awaiting CSV"}
            </span>
          </div>
          {datasetLoaded && datasetInfo && (
            <div className="text-[11px] text-slate-400 truncate">
              {datasetInfo.filename || datasetInfo.source_label} ({datasetInfo.total_records?.toLocaleString()} rows)
            </div>
          )}
          <div className="flex justify-between items-center text-[11px]">
            <span className="text-slate-400">Detector Suite:</span>
            <span className="font-mono text-slate-300 text-[10px]">RF + IF + STC</span>
          </div>
        </div>

        <div className="mt-3 pt-2.5 border-t border-navy-700/60 text-[10px] text-slate-400 text-center">
          Academic College Final-Year Project
        </div>
      </div>
    </aside>
  );
};
