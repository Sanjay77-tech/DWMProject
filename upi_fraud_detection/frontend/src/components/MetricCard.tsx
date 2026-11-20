import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

interface MetricCardProps {
  label: string;
  value: number | string;
  isDecimal?: boolean;
  delta?: string;
  deltaColor?: 'normal' | 'inverse' | 'off';
  icon?: React.ReactNode;
  prefix?: string;
  suffix?: string;
  color?: string;
  delay?: number;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  isDecimal = false,
  delta,
  deltaColor = 'normal',
  icon,
  prefix = '',
  suffix = '',
  delay = 0
}) => {
  const isNumber = typeof value === 'number';
  const [displayValue, setDisplayValue] = useState<number>(0);

  useEffect(() => {
    if (!isNumber) return;
    const target = Number(value);
    const duration = 1200; // ms
    const startTime = performance.now();

    const animate = (currentTime: number) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // easeOutQuart
      const ease = 1 - Math.pow(1 - progress, 4);
      setDisplayValue(target * ease);

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    requestAnimationFrame(animate);
  }, [value, isNumber]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay }}
      className="bg-navy-800/80 border border-navy-700/80 rounded-2xl p-4 flex flex-col justify-between hover:border-navy-600 transition-colors shadow-sm"
    >
      <div className="flex items-center justify-between gap-2 mb-2">
        <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider truncate">
          {label}
        </span>
        {icon && (
          <div className="p-1.5 rounded-lg bg-navy-700/60 border border-navy-600/40 shrink-0">
            {icon}
          </div>
        )}
      </div>

      <div>
        <div className="text-xl font-black text-white tracking-tight font-mono">
          {prefix}
          {isNumber
            ? isDecimal
              ? displayValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
              : Math.round(displayValue).toLocaleString()
            : value}
          {suffix}
        </div>

        {delta && (
          <div
            className={`text-[10px] font-medium mt-1 ${
              deltaColor === 'inverse'
                ? 'text-red-400'
                : deltaColor === 'off'
                ? 'text-slate-400'
                : 'text-emerald-400'
            }`}
          >
            {delta}
          </div>
        )}
      </div>
    </motion.div>
  );
};
