import React from 'react';
import { Database, Shield, RefreshCw } from 'lucide-react';

interface HeaderProps {
  activeTabTitle: string;
  datasetLoaded: boolean;
  datasetLabel?: string;
  onResetDataset?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeTabTitle,
  datasetLoaded,
  datasetLabel,
  onResetDataset
}) => {
  return (
    <header className="h-16 border-b border-navy-700 bg-navy-900/80 backdrop-blur sticky top-0 z-20 px-8 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <h2 className="text-lg font-bold text-white tracking-tight">{activeTabTitle}</h2>
      </div>

      <div className="flex items-center gap-4">
        {datasetLoaded ? (
          <div className="flex items-center gap-3 bg-emerald-500/10 border border-emerald-500/20 px-3 py-1.5 rounded-lg text-xs font-medium text-emerald-400">
            <Database className="w-3.5 h-3.5" />
            <span className="truncate max-w-xs">{datasetLabel || "Active Dataset Loaded"}</span>
            {onResetDataset && (
              <button 
                onClick={onResetDataset}
                title="Unload active dataset" 
                className="hover:text-white transition-colors ml-1 text-slate-400"
              >
                <RefreshCw className="w-3 h-3" />
              </button>
            )}
          </div>
        ) : (
          <div className="flex items-center gap-2 bg-navy-800 border border-navy-700 px-3 py-1.5 rounded-lg text-xs text-amber-400 font-medium">
            <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
            <span>Awaiting CSV Ingestion</span>
          </div>
        )}

        <div className="flex items-center gap-2 text-xs font-medium text-slate-400 bg-navy-800 px-3 py-1.5 rounded-lg border border-navy-700">
          <Shield className="w-3.5 h-3.5 text-emerald-400" />
          <span>Real-Time Engine</span>
        </div>
      </div>
    </header>
  );
};
