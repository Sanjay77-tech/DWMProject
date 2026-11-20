import React from 'react';
import { motion } from 'framer-motion';

interface RiskScoreGaugeProps {
  score: number; // 0 - 100
  category: string;
  status: string;
}

export const RiskScoreGauge: React.FC<RiskScoreGaugeProps> = ({ score, category, status }) => {
  const radius = 70;
  const stroke = 12;
  const normalizedRadius = radius - stroke * 2;
  const circumference = normalizedRadius * 2 * Math.PI;
  const strokeDashoffset = circumference - (score / 100) * circumference;

  let color = '#10b981';
  let badgeBg = 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
  if (score > 80) {
    color = '#ef4444';
    badgeBg = 'bg-rose-500/10 text-rose-400 border-rose-500/30';
  } else if (score > 60) {
    color = '#f97316';
    badgeBg = 'bg-orange-500/10 text-orange-400 border-orange-500/30';
  } else if (score > 30) {
    color = '#f59e0b';
    badgeBg = 'bg-amber-500/10 text-amber-400 border-amber-500/30';
  }

  return (
    <div className="flex flex-col items-center justify-center p-6 bg-navy-850 rounded-2xl border border-navy-700">
      <div className="relative flex items-center justify-center">
        <svg height={radius * 2} width={radius * 2} className="transform -rotate-90">
          <circle
            stroke="#1e293b"
            fill="transparent"
            strokeWidth={stroke}
            r={normalizedRadius}
            cx={radius}
            cy={radius}
          />
          <motion.circle
            stroke={color}
            fill="transparent"
            strokeWidth={stroke}
            strokeDasharray={`${circumference} ${circumference}`}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset }}
            transition={{ duration: 1.5, ease: 'easeOut' }}
            strokeLinecap="round"
            r={normalizedRadius}
            cx={radius}
            cy={radius}
          />
        </svg>

        <div className="absolute text-center flex flex-col items-center justify-center">
          <motion.span
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="text-3xl font-extrabold text-white tracking-tight"
          >
            {score}
          </motion.span>
          <span className="text-[10px] text-slate-400 uppercase tracking-widest font-semibold">/ 100</span>
        </div>
      </div>

      <div className="mt-4 text-center">
        <span className={`inline-block px-3 py-1 rounded-full text-xs font-semibold border ${badgeBg}`}>
          {category} • {status}
        </span>
      </div>
    </div>
  );
};
