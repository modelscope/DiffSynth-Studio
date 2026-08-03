"use client";

import React from "react";

type Point = { step: number; loss: number };

export function LossChart({
  data,
  height = 220,
}: {
  data: Point[];
  height?: number;
}) {
  const validData = (data || []).filter(
    (point) => Number.isFinite(point.step) && Number.isFinite(point.loss),
  );
  if (validData.length === 0) {
    return (
      <div
        className="flex items-center justify-center text-slate-400 text-sm border border-slate-800 rounded-lg"
        style={{ height }}
      >
        暂无 loss 数据
      </div>
    );
  }
  const points = validData.slice(-500);
  const w = 800;
  const h = height;
  const padL = 40;
  const padR = 12;
  const padT = 12;
  const padB = 24;

  const xs = points.map((p) => p.step);
  const ys = points.map((p) => p.loss);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const xRange = Math.max(1, xMax - xMin);
  const yRange = Math.max(1e-6, yMax - yMin);

  const toX = (step: number) => padL + ((step - xMin) / xRange) * (w - padL - padR);
  const toY = (loss: number) => padT + (1 - (loss - yMin) / yRange) * (h - padT - padB);

  const path = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${toX(p.step).toFixed(1)} ${toY(p.loss).toFixed(1)}`)
    .join(" ");

  const yTicks = 4;
  const ticks = Array.from({ length: yTicks + 1 }, (_, i) => yMin + (yRange / yTicks) * i);

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-2">
      <svg viewBox={`0 0 ${w} ${h}`} width="100%" height={h} preserveAspectRatio="none">
        {ticks.map((t, i) => {
          const y = toY(t);
          return (
            <g key={i}>
              <line
                x1={padL}
                x2={w - padR}
                y1={y}
                y2={y}
                stroke="rgba(174,189,202,0.18)"
                strokeWidth={1}
              />
              <text x={4} y={y + 3} fontSize={10} fill="#aebdca">
                {t.toFixed(4)}
              </text>
            </g>
          );
        })}
        <line
          x1={padL}
          x2={padL}
          y1={padT}
          y2={h - padB}
          stroke="rgba(174,189,202,0.38)"
          strokeWidth={1}
        />
        <line
          x1={padL}
          x2={w - padR}
          y1={h - padB}
          y2={h - padB}
          stroke="rgba(174,189,202,0.38)"
          strokeWidth={1}
        />
        <text x={padL} y={h - 6} fontSize={10} fill="#91a2b3">
          step {xMin}
        </text>
        <text x={w - padR - 40} y={h - 6} fontSize={10} fill="#91a2b3">
          step {xMax}
        </text>
        <path d={path} fill="none" stroke="#60a5fa" strokeWidth={1.5} />
      </svg>
    </div>
  );
}
