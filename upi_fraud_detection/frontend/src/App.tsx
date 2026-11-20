import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { DatasetUploadPage } from './pages/DatasetUploadPage';
import { DashboardPage } from './pages/DashboardPage';
import { TransactionPredictionPage } from './pages/TransactionPredictionPage';
import { FraudAnalyticsPage } from './pages/FraudAnalyticsPage';
import { ModelTrainingPage } from './pages/ModelTrainingPage';
import { TransactionHistoryPage } from './pages/TransactionHistoryPage';
import { api, type DatasetStatus } from './api/client';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('upload');
  const [datasetStatus, setDatasetStatus] = useState<DatasetStatus | null>(null);

  // Load initial dataset status
  const checkDatasetStatus = async () => {
    try {
      const status = await api.getDatasetStatus();
      setDatasetStatus(status);
      if (!status.is_loaded) {
        setActiveTab('upload');
      }
    } catch (err) {
      console.warn("FastAPI backend not ready or unreachable yet", err);
    }
  };

  useEffect(() => {
    checkDatasetStatus();
  }, []);

  const handleResetDataset = async () => {
    if (window.confirm("Are you sure you want to unload the current active dataset?")) {
      try {
        await api.resetDataset();
        await checkDatasetStatus();
        setActiveTab('upload');
      } catch (err) {
        console.error("Failed to reset dataset", err);
      }
    }
  };

  const pageTitles: Record<string, string> = {
    upload: 'Dataset Ingestion & Inspection',
    dashboard: 'Executive Threat Dashboard',
    prediction: 'Transaction Risk Inference & XAI',
    analytics: 'Forensic Analytics & Mule Graph',
    training: 'Multi-Paradigm Model Training Studio',
    history: 'Evaluated Transaction Ledger'
  };

  return (
    <div className="flex min-h-screen bg-navy-950 text-slate-100 font-sans selection:bg-emerald-500/30 selection:text-emerald-300">
      {/* Sidebar Navigation */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        datasetLoaded={datasetStatus?.is_loaded ?? false}
        datasetInfo={{
          filename: datasetStatus?.inspection?.filename,
          total_records: datasetStatus?.total_records,
          source_label: datasetStatus?.source_label
        }}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        <Header
          activeTabTitle={pageTitles[activeTab] || 'UPI Fraud Detection'}
          datasetLoaded={datasetStatus?.is_loaded ?? false}
          datasetLabel={datasetStatus?.inspection?.filename || datasetStatus?.source_label}
          onResetDataset={datasetStatus?.is_loaded ? handleResetDataset : undefined}
        />

        <main className="flex-1 p-8 overflow-y-auto max-w-7xl w-full mx-auto">
          {activeTab === 'upload' && (
            <DatasetUploadPage
              datasetStatus={datasetStatus}
              onDatasetUpdated={(status) => {
                setDatasetStatus(status);
              }}
              onNavigateToDashboard={() => setActiveTab('dashboard')}
            />
          )}

          {activeTab === 'dashboard' && (
            <DashboardPage onNavigateToTab={(tab) => setActiveTab(tab)} />
          )}

          {activeTab === 'prediction' && (
            <TransactionPredictionPage />
          )}

          {activeTab === 'analytics' && (
            <FraudAnalyticsPage />
          )}

          {activeTab === 'training' && (
            <ModelTrainingPage />
          )}

          {activeTab === 'history' && (
            <TransactionHistoryPage />
          )}
        </main>
      </div>
    </div>
  );
};

export default App;
