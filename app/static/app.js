let selectedPlanId = null;

function getRequiredElement(id) {
  const el = document.getElementById(id);
  if (!el) {
    throw new Error(`Missing required UI element: #${id}. Please refresh the page and try again.`);
  }
  return el;
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed: ${res.status}`);
  }
  if (res.status === 204) return null;
  return res.json();
}

function setStatus(msg) {
  document.getElementById("source-status").textContent = msg;
}

function formatNum(value) {
  if (value === null || value === undefined) return "-";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function formatBudgetValue(value) {
  if (value === null || value === undefined) return "-";
  const n = Number(value);
  const abs = Math.abs(n);
  if (abs >= 1000) {
    return `${formatNum(n / 1000)}B`;
  }
  return `${formatNum(n)}M`;
}

function formatPct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${formatNum(value)}%`;
}

function titleCase(text) {
  if (!text) return "Unknown";
  return String(text)
    .replaceAll("_", " ")
    .split(" ")
    .map((w) => (w ? w[0].toUpperCase() + w.slice(1) : w))
    .join(" ");
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderInlineMarkdown(text) {
  return escapeHtml(text)
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>')
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}

function renderAnswerMarkdown(markdown) {
  const lines = String(markdown || "").split(/\r?\n/);
  const blocks = [];
  let paragraph = [];
  let list = null;

  const flushParagraph = () => {
    if (!paragraph.length) return;
    blocks.push(`<p>${renderInlineMarkdown(paragraph.join(" "))}</p>`);
    paragraph = [];
  };

  const flushList = () => {
    if (!list) return;
    blocks.push(`<${list.type}>${list.items.join("")}</${list.type}>`);
    list = null;
  };

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }

    const heading = trimmed.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      flushList();
      const level = heading[1].length + 2;
      blocks.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }

    const bullet = trimmed.match(/^[-*]\s+(.+)$/);
    const numbered = trimmed.match(/^\d+[.)]\s+(.+)$/);
    if (bullet || numbered) {
      flushParagraph();
      const type = bullet ? "ul" : "ol";
      if (!list || list.type !== type) {
        flushList();
        list = { type, items: [] };
      }
      list.items.push(`<li>${renderInlineMarkdown((bullet || numbered)[1])}</li>`);
      continue;
    }

    flushList();
    paragraph.push(trimmed);
  }

  flushParagraph();
  flushList();
  return blocks.join("");
}

function renderDebug(debug) {
  const summaryEl = document.getElementById("debug-summary");
  const outputEl = document.getElementById("debug-output");
  const panelEl = document.getElementById("debug-panel");
  if (!summaryEl || !outputEl || !panelEl) return;

  if (!debug) {
    summaryEl.textContent = "No debug details for this response.";
    outputEl.textContent = "";
    return;
  }

  const reasoning = Array.isArray(debug.reasoning_summaries) ? debug.reasoning_summaries : [];
  const toolCalls = Array.isArray(debug.tool_calls) ? debug.tool_calls : [];
  const blocks = Array.isArray(debug.response_blocks) ? debug.response_blocks : [];
  const retrieval = debug.retrieval_policy || {};

  const lines = [];
  if (reasoning.length) {
    lines.push(
      "Reasoning Summaries:\n" +
        reasoning.map((r, i) => `${i + 1}. ${String(r).replace(/\s+/g, " ").trim()}`).join("\n")
    );
  }

  if (toolCalls.length) {
    lines.push(`Tool Calls:\n${JSON.stringify(toolCalls, null, 2)}`);
  }

  if (Object.keys(retrieval).length) {
    lines.push(`Retrieval Policy:\n${JSON.stringify(retrieval, null, 2)}`);
  }

  lines.push(`Response Blocks:\n${JSON.stringify(blocks, null, 2)}`);

  summaryEl.textContent = `Blocks: ${blocks.length} | Reasoning: ${reasoning.length} | Tool calls: ${toolCalls.length}`;
  outputEl.textContent = lines.join("\n\n");
}

function renderComparisonSummary(summary, rows, plan) {
  const increaseCount = summary.increase_count ?? rows.filter((r) => r.delta_directional === "increase").length;
  const decreaseCount = summary.decrease_count ?? rows.filter((r) => r.delta_directional === "decrease").length;
  const flatCount = rows.filter((r) => r.delta_directional === "flat").length;

  const mayorSubtotal = rows.reduce(
    (acc, row) => acc + (row.mayor_fy_2026_27 === null || row.mayor_fy_2026_27 === undefined ? 0 : Number(row.mayor_fy_2026_27)),
    0
  );
  const proposedSubtotal = rows.reduce(
    (acc, row) => acc + (row.proposed_fy_2026_27 === null || row.proposed_fy_2026_27 === undefined ? 0 : Number(row.proposed_fy_2026_27)),
    0
  );
  const subtotalDelta = proposedSubtotal - mayorSubtotal;
  const subtotalDeltaClass = subtotalDelta > 0 ? "delta-pos" : subtotalDelta < 0 ? "delta-neg" : "";
  const subtotalDeltaSign = subtotalDelta > 0 ? "+" : "";

  const fullMayorTotal = plan?.mayor_total_budget === null || plan?.mayor_total_budget === undefined
    ? null
    : Number(plan.mayor_total_budget);
  const fullProposedTotal = plan?.generated_total_budget === null || plan?.generated_total_budget === undefined
    ? null
    : Number(plan.generated_total_budget);
  const fullDelta =
    fullMayorTotal === null || fullProposedTotal === null ? null : fullProposedTotal - fullMayorTotal;
  const fullDeltaClass = fullDelta > 0 ? "delta-pos" : fullDelta < 0 ? "delta-neg" : "";
  const fullDeltaSign = fullDelta > 0 ? "+" : "";
  const coveragePct =
    fullMayorTotal && fullMayorTotal !== 0 ? (mayorSubtotal / fullMayorTotal) * 100 : null;

  const summaryEl = document.getElementById("comparison-summary");
  summaryEl.innerHTML = `
    <div class="summary-card"><div class="label">Departments Up</div><div class="value">${increaseCount}</div></div>
    <div class="summary-card"><div class="label">Departments Down</div><div class="value">${decreaseCount}</div></div>
    <div class="summary-card"><div class="label">Departments Flat</div><div class="value">${flatCount}</div></div>
    <div class="summary-card"><div class="label">Full Plan Mayor Total (USD, auto M/B)</div><div class="value">${formatBudgetValue(fullMayorTotal)}</div></div>
    <div class="summary-card"><div class="label">Full Plan Proposed Total (USD, auto M/B)</div><div class="value">${formatBudgetValue(fullProposedTotal)}</div></div>
    <div class="summary-card"><div class="label">Full Plan Net Change (USD, auto M/B)</div><div class="value ${fullDeltaClass}">${fullDelta === null ? "-" : `${fullDeltaSign}${formatBudgetValue(fullDelta)}`}</div></div>
    <div class="summary-card"><div class="label">Displayed Dept Subtotal (Mayor, USD, auto M/B)</div><div class="value">${formatBudgetValue(mayorSubtotal)}</div></div>
    <div class="summary-card"><div class="label">Displayed Dept Subtotal (Proposed, USD, auto M/B)</div><div class="value">${formatBudgetValue(proposedSubtotal)}</div></div>
    <div class="summary-card"><div class="label">Displayed Subtotal Net Change (USD, auto M/B)</div><div class="value ${subtotalDeltaClass}">${subtotalDeltaSign}${formatBudgetValue(subtotalDelta)}</div></div>
    <div class="summary-card"><div class="label">Displayed Coverage of Mayor Total</div><div class="value">${formatPct(coveragePct)}</div></div>
  `;

  const largestIncreases = summary.largest_increases || [];
  const largestDecreases = summary.largest_decreases || [];
  const deficitHint = summary.deficit_actions_hint || "See deficit handling section in plan memo.";

  const contextEl = document.getElementById("comparison-context");
  contextEl.innerHTML = `
    <div><strong>How to read these numbers:</strong> the <em>Full Plan</em> totals come from the generated plan constraints, while the <em>Displayed Dept Subtotal</em> only sums departments currently extracted into the table.</div>
    <div style="margin-top:8px;">Values are shown in USD with auto-scaling: <strong>M</strong> for millions and <strong>B</strong> for billions (at 1,000M+).</div>
    <div class="context-lists">
      <div>
        <strong>Largest Increases</strong>
        <ul>${largestIncreases.map((x) => `<li>${x}</li>`).join("") || "<li>None identified</li>"}</ul>
      </div>
      <div>
        <strong>Largest Decreases</strong>
        <ul>${largestDecreases.map((x) => `<li>${x}</li>`).join("") || "<li>None identified</li>"}</ul>
      </div>
    </div>
    <div style="margin-top:8px;"><strong>Deficit Context:</strong> ${deficitHint}</div>
  `;
}

async function loadSources() {
  const data = await api("/sources");
  const freshness = data.last_refreshed_at
    ? new Date(data.last_refreshed_at).toLocaleString()
    : "never";
  const staleLabel = data.stale ? "STALE" : "FRESH";
  setStatus(
    `Sources: ${data.sources.length} | Last refreshed: ${freshness} | ${staleLabel}`
  );
}

async function loadPlans() {
  const data = await api("/plans");
  const el = document.getElementById("plan-list");
  el.innerHTML = "";

  if (!data.plans.length) {
    el.textContent = "No plan versions yet. Generate one.";
    return;
  }

  for (const plan of data.plans) {
    const row = document.createElement("div");
    row.className = "plan-row";
    row.innerHTML = `
      <button class="plan-select">Plan #${plan.id}</button>
      <span>${new Date(plan.created_at).toLocaleString()}</span>
      <span>Spend cap: ${plan.spending_rule_passed ? "pass" : "unknown/fail"}</span>
      <span>Total: ${formatBudgetValue(plan.generated_total_budget)}</span>
    `;
    row.querySelector(".plan-select").onclick = () => selectPlan(plan.id);
    el.appendChild(row);
  }

  if (!selectedPlanId) {
    await selectPlan(data.plans[0].id);
  }
}

async function selectPlan(planId) {
  selectedPlanId = planId;
  const plan = await api(`/plans/${planId}`);
  document.getElementById("plan-meta").textContent =
    `Plan #${plan.id} | Scope: ${plan.scope_label} | Model: ${plan.model_used} | ` +
    `Mayor total: ${formatBudgetValue(plan.mayor_total_budget)} | Proposed total: ${formatBudgetValue(plan.generated_total_budget)}`;
  document.getElementById("plan-memo").textContent = plan.memo_markdown;

  const comp = await api(`/plans/${planId}/comparison`);
  renderComparisonSummary(comp.summary || {}, comp.rows || [], plan);

  const tbody = document.querySelector("#comparison-table tbody");
  tbody.innerHTML = "";
  for (const row of comp.rows) {
    const mayor = row.mayor_fy_2026_27;
    const proposed = row.proposed_fy_2026_27;
    const amountDelta =
      mayor === null || mayor === undefined || proposed === null || proposed === undefined
        ? null
        : Number(proposed) - Number(mayor);
    const pctDelta =
      amountDelta === null || Number(mayor) === 0
        ? null
        : (amountDelta / Number(mayor)) * 100;
    const deltaClass = amountDelta > 0 ? "delta-pos" : amountDelta < 0 ? "delta-neg" : "";
    const deltaSign = amountDelta > 0 ? "+" : "";

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.department}</td>
      <td>${titleCase(row.mayor_directional)}</td>
      <td>${titleCase(row.proposed_directional)}</td>
      <td>${titleCase(row.delta_directional)}</td>
      <td>${formatBudgetValue(row.mayor_fy_2026_27)}</td>
      <td>${formatBudgetValue(row.proposed_fy_2026_27)}</td>
      <td class="${deltaClass}">${amountDelta === null ? "-" : `${deltaSign}${formatBudgetValue(amountDelta)}`}</td>
      <td class="${deltaClass}">${pctDelta === null ? "-" : `${deltaSign}${formatNum(pctDelta)}%`}</td>
    `;
    tbody.appendChild(tr);
  }

  document.getElementById("download-pdf").disabled = false;
  document.getElementById("download-csv").disabled = false;
  document.getElementById("ask-btn").disabled = false;
}

async function refreshSources() {
  setStatus("Refreshing sources... this may take a few minutes.");
  await api("/ingest/run", { method: "POST" });
  await loadSources();
}

async function generatePlan() {
  if (!confirm("Generate a new immutable plan version now?")) return;
  await api("/plans/generate", {
    method: "POST",
    body: JSON.stringify({ force_refresh_hint: false }),
  });
  await loadPlans();
}

async function askQuestion() {
  if (!selectedPlanId) return;
  const query = getRequiredElement("question").value.trim();
  if (!query) return;

  const threadId = "default";
  const escalate = getRequiredElement("escalate").checked;

  getRequiredElement("answer").textContent = "Thinking...";
  getRequiredElement("answer-meta").textContent = "";
  renderDebug(null);
  const result = await api(`/plans/${selectedPlanId}/chat`, {
    method: "POST",
    body: JSON.stringify({ query, thread_id: threadId, escalate }),
  });

  const includePlan = !!result?.debug?.retrieval_policy?.include_plan_sources;
  const modeLabel = includePlan ? "official + plan" : "official only";
  getRequiredElement("answer-meta").textContent = `Model: ${result.model_used} | Retrieval: ${modeLabel}`;
  getRequiredElement("answer").innerHTML = renderAnswerMarkdown(result.answer);
  renderDebug(result.debug || null);
  const citations = getRequiredElement("citations");
  citations.innerHTML =
    "<h4>Citations</h4>" +
    result.citations
      .map(
        (c) => {
          const kind = (c.source_kind || "official").toLowerCase();
          const badge = `<span class="source-badge ${kind}">${titleCase(kind)}</span>`;
          const planSuffix = c.plan_id ? ` (Plan #${c.plan_id})` : "";
          const label = `${c.label || c.url}${planSuffix}`;
          if (typeof c.url === "string" && c.url.startsWith("http")) {
            return `<div class="citation-item">${badge}<a href="${c.url}" target="_blank" rel="noreferrer">${label}</a></div>`;
          }
          return `<div class="citation-item">${badge}${label}</div>`;
        }
      )
      .join("");
}

document.getElementById("refresh-btn").onclick = () => refreshSources().catch((err) => alert(err.message));
document.getElementById("generate-btn").onclick = () => generatePlan().catch((err) => alert(err.message));
document.getElementById("ask-btn").onclick = () => askQuestion().catch((err) => alert(err.message));
document.getElementById("download-pdf").onclick = () => {
  if (!selectedPlanId) return;
  window.open(`/plans/${selectedPlanId}/export/pdf`, "_blank");
};
document.getElementById("download-csv").onclick = () => {
  if (!selectedPlanId) return;
  window.open(`/plans/${selectedPlanId}/comparison/export/csv`, "_blank");
};

(async function init() {
  try {
    await loadSources();
    await loadPlans();
  } catch (err) {
    alert(err.message);
  }
})();
