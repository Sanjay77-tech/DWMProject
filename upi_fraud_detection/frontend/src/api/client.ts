/**
 * REST API Client for UPI Sentinel AI FastAPI Backend
 */

const BASE_URL = '/api';

export interface InspectionData {
  filename?: string;
  total_rows?: number;
  total_columns?: number;
  raw_columns?: string[];
  mapped_columns?: Record<string, string>;
  dtypes?: Record<string, string>;
  missing_values_total?: number;
  missing_per_column?: Record<string, number>;
  duplicate_rows?: number;
  has_labels?: boolean;
  fraud_count?: number;
  fraud_rate?: number;
  has_timestamp?: boolean;
  has_device?: boolean;
  has_location?: boolean;
  has_balance?: boolean;
}

export interface DatasetStatus {
  is_loaded: boolean;
  source_label: string;
  filename?: string;
  total_records: number;
  total_columns?: number;
  inspection?: InspectionData;
}

export const api = {
  // Dataset Endpoints
  async getDatasetStatus(): Promise<DatasetStatus> {
    const res = await fetch(`${BASE_URL}/dataset/status`);
    if (!res.ok) throw new Error("Failed to fetch dataset status");
    return res.json();
  },

  async uploadDataset(file: File) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${BASE_URL}/dataset/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
      throw new Error(err.detail || 'Upload failed');
    }
    return res.json();
  },

  async loadBenchmark() {
    const res = await fetch(`${BASE_URL}/dataset/load-benchmark`, { method: 'POST' });
    if (!res.ok) throw new Error("Failed to load benchmark dataset");
    return res.json();
  },

  async resetDataset() {
    const res = await fetch(`${BASE_URL}/dataset/reset`, { method: 'POST' });
    return res.json();
  },

  // Dashboard Endpoints
  async getDashboardStats() {
    const res = await fetch(`${BASE_URL}/dashboard/stats`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Failed to fetch dashboard stats' }));
      throw new Error(err.detail);
    }
    return res.json();
  },

  // Prediction Endpoints
  async analyzeTransaction(payload: any) {
    const res = await fetch(`${BASE_URL}/prediction/single`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Analysis failed' }));
      throw new Error(err.detail);
    }
    return res.json();
  },

  async getActiveModel() {
    const res = await fetch(`${BASE_URL}/prediction/active-model`);
    return res.json();
  },

  async getPresets() {
    const res = await fetch(`${BASE_URL}/prediction/presets`);
    return res.json();
  },

  // Analytics Endpoints
  async getAnalytics() {
    const res = await fetch(`${BASE_URL}/analytics/all`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Failed to fetch analytics' }));
      throw new Error(err.detail);
    }
    return res.json();
  },

  // Training Endpoints
  async startTraining(paradigm: string, algorithm: string, hyperparameters: Record<string, any>) {
    const res = await fetch(`${BASE_URL}/training/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ paradigm, algorithm, hyperparameters }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Training launch failed' }));
      throw new Error(err.detail);
    }
    return res.json();
  },

  async getTrainingStatus(jobId: string) {
    const res = await fetch(`${BASE_URL}/training/status/${jobId}`);
    if (!res.ok) throw new Error("Failed to fetch training status");
    return res.json();
  },

  async runBenchmarkSuite() {
    const res = await fetch(`${BASE_URL}/training/benchmark-suite`, { method: 'POST' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Benchmark suite failed' }));
      throw new Error(err.detail);
    }
    return res.json();
  },

  async getAvailableModels() {
    const res = await fetch(`${BASE_URL}/training/models`);
    return res.json();
  },

  async setActiveModel(modelId: string) {
    const res = await fetch(`${BASE_URL}/training/set-active`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: modelId }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Failed to set active model' }));
      throw new Error(err.detail);
    }
    return res.json();
  },

  // History Endpoints
  async getHistory(params: { risk_category?: string; min_amount?: number; search_query?: string; record_source?: string; page?: number; page_size?: number }) {
    const query = new URLSearchParams();
    if (params.risk_category) query.append('risk_category', params.risk_category);
    if (params.min_amount !== undefined) query.append('min_amount', params.min_amount.toString());
    if (params.search_query) query.append('search_query', params.search_query);
    if (params.record_source) query.append('record_source', params.record_source);
    if (params.page) query.append('page', params.page.toString());
    if (params.page_size) query.append('page_size', params.page_size.toString());

    const res = await fetch(`${BASE_URL}/history/records?${query.toString()}`);
    if (!res.ok) throw new Error("Failed to fetch history");
    return res.json();
  }
};
