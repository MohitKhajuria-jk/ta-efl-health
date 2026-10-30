import { useState } from "react";
import { motion, useReducedMotion } from "motion/react";
import type { Job, Round } from "../types";
import { Badge } from "./ui";
import { fmt, percent, label } from "../api";

export function Network({
  count = 5,
  job,
  round,
}: {
  count?: number;
  job?: Job | null;
  round?: Round;
}) {
  const [selected, setSelected] = useState<string | null>(null);
  const reduced = useReducedMotion();
  const shown = Math.min(count, 20);
  const nodes = Array.from({ length: shown }, (_, i) => {
    const angle = (i / shown) * Math.PI * 2 - Math.PI / 2;
    return {
      id: `client_${i + 1}`,
      x: 250 + Math.cos(angle) * 180,
      y: 174 + Math.sin(angle) * 125,
    };
  });
  const client = round?.clients.find((c) => c.cid === selected);
  const latest = selected ? job?.clients[selected] : null;
  return (
    <div className="network">
      <svg
        viewBox="0 0 500 350"
        aria-label="In-process federated simulation network"
      >
        <defs>
          <pattern
            id="network-grid"
            width="18"
            height="18"
            patternUnits="userSpaceOnUse"
          >
            <circle cx="1" cy="1" r=".8" fill="var(--border)" />
          </pattern>
        </defs>
        <rect width="500" height="350" fill="url(#network-grid)" />
        {nodes.map((n) => {
          const event = job?.clients[n.id];
          const moving =
            job?.status === "running" &&
            event &&
            ["client_training", "client_submitted"].includes(event.type);
          return (
            <g key={n.id}>
              <line
                x1={250}
                y1={174}
                x2={n.x}
                y2={n.y}
                stroke="var(--network-line)"
                strokeWidth="1.5"
              />
              {moving && !reduced && (
                <motion.circle
                  key={`${job.id}-${event.sequence}`}
                  r={4}
                  fill="var(--brand)"
                  initial={{
                    cx: event.type === "client_training" ? 250 : n.x,
                    cy: event.type === "client_training" ? 174 : n.y,
                    opacity: 1,
                  }}
                  animate={{
                    cx: event.type === "client_training" ? n.x : 250,
                    cy: event.type === "client_training" ? n.y : 174,
                    opacity: 0,
                  }}
                  transition={{ duration: 0.8 }}
                />
              )}
              <g
                role="button"
                tabIndex={0}
                aria-label={`Inspect simulated client ${n.id.split("_")[1]}`}
                onClick={() => setSelected(n.id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    setSelected(n.id);
                  }
                }}
                className="network-node"
              >
                <rect
                  x={n.x - 26}
                  y={n.y - 24}
                  width={52}
                  height={48}
                  rx={13}
                  fill={
                    selected === n.id ? "var(--brand-tint)" : "var(--surface)"
                  }
                  stroke={selected === n.id ? "var(--brand)" : "var(--border)"}
                />
                <path
                  d={`M ${n.x - 9} ${n.y + 9} v-17 h18 v17 M ${n.x - 14} ${n.y + 9} h28 M ${n.x - 3} ${n.y - 3} h6 M ${n.x} ${n.y - 6} v6`}
                  stroke="var(--brand)"
                  strokeWidth="1.8"
                  fill="none"
                />
                <text
                  x={n.x}
                  y={n.y + 40}
                  textAnchor="middle"
                  fill="var(--text-muted)"
                  fontSize="11"
                >
                  Client {n.id.split("_")[1]}
                </text>
              </g>
            </g>
          );
        })}
        <circle cx={250} cy={174} r={47} fill="var(--brand-tint)" />
        <rect
          x={213}
          y={137}
          width={74}
          height={74}
          rx={23}
          fill="var(--navy)"
        />
        <path
          d="M 231 174 h10 l6 -12 8 24 6 -12 h9"
          stroke="var(--accent)"
          fill="none"
          strokeWidth="2.5"
        />
        <text
          x={250}
          y={232}
          textAnchor="middle"
          fill="var(--text)"
          fontWeight={600}
          fontSize={11}
        >
          Coordinator
        </text>
      </svg>
      <div className="network-caption">
        <Badge tone={job?.status === "running" ? "info" : "neutral"}>
          {job?.status === "running"
            ? label(job.stage)
            : round
              ? "Saved round"
              : "Ready when you are"}
        </Badge>
        <span>In-process simulation</span>
      </div>
      {selected && (
        <div className="client-inspect">
          <strong>Client {selected.split("_")[1]}</strong>
          {client ? (
            <span>
              Trust {fmt(client.trust)} · weight{" "}
              {client.weight == null ? "N/A (median)" : percent(client.weight)}{" "}
              · {client.num_examples} training rows
            </span>
          ) : (
            <span>
              {latest
                ? label(latest.type.replace("client_", ""))
                : "No update recorded for this client yet."}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
