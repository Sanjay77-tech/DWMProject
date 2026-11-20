import React, { useState, useEffect } from 'react';
import { 
  FileText, 
  Search, 
  Filter, 
  Download, 
  ChevronLeft, 
  ChevronRight, 
  RefreshCw
} from 'lucide-react';
import { api } from '../api/client';

export const TransactionHistoryPage: React.FC = () => {
  const [records, setRecords] = useState<any[]>([]);
  const [totalRecords, setTotalRecords] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize] = useState(15);
  const [loading, setLoading] = useState(false);

  // Filters
  const [riskCategory, setRiskCategory] = useState<string>('');
  const [minAmount, setMinAmount] = useState<number>(0);
  const [searchQuery, setSearchQuery] = useState<string>('');

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const data = await api.getHistory({
        risk_category: riskCategory || undefined,
        min_amount: minAmount > 0 ? minAmount : undefined,
        search_query: searchQuery || undefined,
        page: currentPage,
        page_size: pageSize
      });
      setRecords(data.records || []);
      setTotalRecords(data.total_matched || 0);
    } catch (err) {
      console.error("Failed to fetch transaction history", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [currentPage, riskCategory, minAmount]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setCurrentPage(1);
    fetchHistory();
  };

  const totalPages = Math.max(1, Math.ceil(totalRecords / pageSize));

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header & Export */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold text-emerald-400 uppercase tracking-wider mb-1">
            <FileText className="w-3.5 h-3.5" />
            <span>Forensic Audit & Ledger</span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Evaluated Transaction Audit Ledger</h1>
          <p className="text-slate-400 text-sm mt-1">
            Historical transaction ledger scored and tagged by the multi-model fraud detection suite.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <a
            href="/api/history/export"
            download="upi_fraud_scored_audit.csv"
            className="px-4 py-2 bg-emerald-500 hover:bg-emerald-400 text-navy-950 font-bold text-xs rounded-xl transition-all shadow-md shadow-emerald-500/20 flex items-center gap-2"
          >
            <Download className="w-4 h-4" />
            <span>Export Scored Audit CSV</span>
          </a>
        </div>
      </div>

      {/* Filter & Search Bar */}
      <div className="bg-navy-800/80 border border-navy-700 rounded-2xl p-5">
        <form onSubmit={handleSearchSubmit} className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Search Query */}
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">
              Search Accounts / Transaction ID
            </label>
            <div className="relative">
              <input
                type="text"
                placeholder="e.g. user_0042, TXN..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full bg-navy-900 border border-navy-700 rounded-lg pl-8 pr-3 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500"
              />
              <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2.5 top-2.5" />
            </div>
          </div>

          {/* Risk Tier Filter */}
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">
              Risk Tier Filter
            </label>
            <select
              value={riskCategory}
              onChange={(e) => {
                setRiskCategory(e.target.value);
                setCurrentPage(1);
              }}
              className="w-full bg-navy-900 border border-navy-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-emerald-500"
            >
              <option value="">All Risk Tiers</option>
              <option value="LOW">Low Risk (&le; 30)</option>
              <option value="MEDIUM">Medium Risk (31–60)</option>
              <option value="HIGH">High Risk (61–80)</option>
              <option value="VERY HIGH">Very High Risk (&gt; 80)</option>
            </select>
          </div>

          {/* Minimum Amount */}
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">
              Minimum Amount (₹)
            </label>
            <input
              type="number"
              min="0"
              placeholder="0"
              value={minAmount || ''}
              onChange={(e) => {
                setMinAmount(parseFloat(e.target.value) || 0);
                setCurrentPage(1);
              }}
              className="w-full bg-navy-900 border border-navy-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-emerald-500 font-mono"
            />
          </div>

          {/* Apply Button */}
          <div className="flex items-end">
            <button
              type="submit"
              className="w-full py-2 px-4 bg-navy-700 hover:bg-navy-600 border border-navy-600 rounded-lg text-xs font-semibold text-white transition-colors flex items-center justify-center gap-2"
            >
              <Filter className="w-3.5 h-3.5 text-emerald-400" />
              <span>Filter Ledger</span>
            </button>
          </div>
        </form>
      </div>

      {/* Table & Pagination */}
      <div className="bg-navy-800/80 border border-navy-700 rounded-2xl overflow-hidden">
        <div className="p-4 border-b border-navy-700 flex items-center justify-between text-xs text-slate-400">
          <div>
            Showing <span className="font-semibold text-white">{records.length}</span> of{' '}
            <span className="font-semibold text-white">{totalRecords.toLocaleString()}</span> matched records
          </div>
          {loading && (
            <div className="flex items-center gap-2 text-emerald-400 font-mono text-[11px]">
              <RefreshCw className="w-3 h-3 animate-spin" />
              <span>Refreshing...</span>
            </div>
          )}
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-navy-900/80 text-slate-400 uppercase text-[10px] font-semibold border-b border-navy-700">
              <tr>
                <th className="py-3 px-4">Timestamp</th>
                <th className="py-3 px-4">Transaction ID</th>
                <th className="py-3 px-4">Counterparties (Sender → Receiver)</th>
                <th className="py-3 px-4">Amount (₹)</th>
                <th className="py-3 px-4">Risk Score</th>
                <th className="py-3 px-4">Risk Category</th>
                <th className="py-3 px-4">Primary Attribution Factor</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-navy-700/60 font-mono text-[11px]">
              {records.length > 0 ? (
                records.map((rec: any, idx: number) => {
                  const score = rec.risk_score || 0;
                  return (
                    <tr key={idx} className="hover:bg-navy-700/30 transition-colors">
                      <td className="py-3 px-4 text-slate-400 text-[10px] whitespace-nowrap">
                        {rec.timestamp || '2026-11-20 09:30:00'}
                      </td>
                      <td className="py-3 px-4 text-white font-medium whitespace-nowrap">
                        {rec.transaction_id || `TXN${idx + 1000}`}
                      </td>
                      <td className="py-3 px-4 text-slate-300 font-sans">
                        <span className="text-white font-mono text-[11px]">{rec.sender_id || 'sender'}</span>
                        <span className="text-slate-500 mx-1.5">→</span>
                        <span className="text-slate-300 font-mono text-[11px]">{rec.receiver_id || 'receiver'}</span>
                      </td>
                      <td className="py-3 px-4 text-white font-bold whitespace-nowrap">
                        ₹{Number(rec.amount || 0).toLocaleString()}
                      </td>
                      <td className="py-3 px-4 whitespace-nowrap">
                        <div className="flex items-center gap-2">
                          <div className="w-16 bg-navy-900 h-1.5 rounded-full overflow-hidden">
                            <div 
                              className={`h-full rounded-full ${
                                score >= 80 ? 'bg-red-400' : score >= 50 ? 'bg-amber-400' : 'bg-emerald-400'
                              }`}
                              style={{ width: `${Math.min(100, score)}%` }}
                            />
                          </div>
                          <span className={`font-bold ${
                            score >= 80 ? 'text-red-400' : score >= 50 ? 'text-amber-400' : 'text-emerald-400'
                          }`}>
                            {score}
                          </span>
                        </div>
                      </td>
                      <td className="py-3 px-4 font-sans whitespace-nowrap">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          rec.risk_category?.includes('HIGH') || score >= 60
                            ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                            : rec.risk_category?.includes('MEDIUM') || score >= 30
                            ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                            : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        }`}>
                          {rec.risk_category || (score >= 60 ? 'HIGH RISK' : 'LOW RISK')}
                        </span>
                      </td>
                      <td className="py-3 px-4 font-sans text-slate-400 text-xs truncate max-w-xs">
                        {rec.top_reason || 'Normal baseline velocity and transaction amounts.'}
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={7} className="py-8 text-center text-slate-400 font-sans text-xs">
                    No transactions matched your current filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Bar */}
        <div className="p-4 border-t border-navy-700 flex items-center justify-between text-xs text-slate-400">
          <div>
            Page <span className="font-semibold text-white">{currentPage}</span> of{' '}
            <span className="font-semibold text-white">{totalPages}</span>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
              disabled={currentPage <= 1 || loading}
              className="p-1.5 rounded-lg bg-navy-900 border border-navy-700 hover:border-navy-600 disabled:opacity-40 disabled:cursor-not-allowed transition-all text-white"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
              disabled={currentPage >= totalPages || loading}
              className="p-1.5 rounded-lg bg-navy-900 border border-navy-700 hover:border-navy-600 disabled:opacity-40 disabled:cursor-not-allowed transition-all text-white"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
