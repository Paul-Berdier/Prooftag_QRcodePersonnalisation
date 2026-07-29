const state = {
  schema: null,
  methods: [],
  campaigns: [],
  selectedCampaign: null,
  selectedTrial: null,
  poller: null,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_) {}
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

function switchTab(name) {
  $$(".tab").forEach((button) => button.classList.toggle("active", button.dataset.tab === name));
  $$(".panel").forEach((panel) => panel.classList.toggle("active", panel.id === `${name}-panel`));
}

function deepCopy(value) {
  return JSON.parse(JSON.stringify(value));
}

function slug(value) {
  return value.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "").slice(0, 90) || "method";
}

function renderMethods() {
  const container = $("#methods");
  container.innerHTML = "";
  state.methods.forEach((method, index) => {
    const node = $("#method-template").content.firstElementChild.cloneNode(true);
    node.dataset.index = index;
    node.classList.toggle("disabled", !method.enabled);
    $(".method-enabled", node).checked = method.enabled;
    $(".method-name-label", node).textContent = method.name;
    $(".method-description", node).textContent = method.description || "";
    $(".method-name", node).value = method.name;
    $(".method-id", node).value = method.id;
    $(".method-backend", node).value = method.backend;
    $(".method-output", node).value = method.output_variant || "auto";
    $(".reuse-stage1", node).checked = method.reuse_stage1 !== false;
    $(".gen-steps", node).value = method.generation?.steps ?? 40;
    $(".gen-cfg", node).value = method.generation?.guidance_scale ?? 7.5;
    $(".gen-control", node).value = method.generation?.controlnet_scale ?? 1.35;
    $(".gen-strength", node).value = method.generation?.strength ?? 1;
    $(".tool-srpg", node).checked = !!method.tools?.srpg_enabled;
    $(".tool-guided", node).checked = !!method.tools?.guided_rediffusion_enabled;
    $(".tool-latent", node).checked = !!method.tools?.latent_refinement_enabled;
    $(".model-json", node).value = JSON.stringify(method.model || {}, null, 2);
    $(".tools-json", node).value = JSON.stringify(method.tools?.settings || {}, null, 2);

    const sync = () => {
      const item = state.methods[index];
      item.enabled = $(".method-enabled", node).checked;
      item.name = $(".method-name", node).value.trim();
      item.id = $(".method-id", node).value.trim();
      item.backend = $(".method-backend", node).value;
      item.output_variant = $(".method-output", node).value;
      item.reuse_stage1 = $(".reuse-stage1", node).checked;
      item.generation = {
        steps: Number($(".gen-steps", node).value),
        guidance_scale: Number($(".gen-cfg", node).value),
        controlnet_scale: Number($(".gen-control", node).value),
        strength: Number($(".gen-strength", node).value),
      };
      item.tools = item.tools || {};
      item.tools.srpg_enabled = $(".tool-srpg", node).checked;
      item.tools.guided_rediffusion_enabled = $(".tool-guided", node).checked;
      item.tools.latent_refinement_enabled = $(".tool-latent", node).checked;
      $(".method-name-label", node).textContent = item.name || "Sans nom";
      node.classList.toggle("disabled", !item.enabled);
      updateTrialCount();
    };
    $$("input, select", node).forEach((input) => input.addEventListener("change", sync));
    $(".method-name", node).addEventListener("input", sync);
    $(".model-json", node).addEventListener("change", () => {
      try {
        state.methods[index].model = JSON.parse($(".model-json", node).value || "{}");
        clearComposeError();
      } catch (error) { showComposeError(`Modèle JSON — ${method.name}: ${error.message}`); }
    });
    $(".tools-json", node).addEventListener("change", () => {
      try {
        state.methods[index].tools.settings = JSON.parse($(".tools-json", node).value || "{}");
        clearComposeError();
      } catch (error) { showComposeError(`Outils JSON — ${method.name}: ${error.message}`); }
    });
    $(".duplicate-method", node).addEventListener("click", () => {
      sync();
      const clone = deepCopy(state.methods[index]);
      clone.name += " — copie";
      clone.id = `${slug(clone.id)}_copy_${Date.now().toString().slice(-4)}`;
      state.methods.splice(index + 1, 0, clone);
      renderMethods();
    });
    $(".remove-method", node).addEventListener("click", () => {
      state.methods.splice(index, 1);
      renderMethods();
    });
    container.append(node);
  });
  updateTrialCount();
}

function parsePrompts() {
  return $("#prompts").value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean).map((line, i) => {
    const [rawId, text, negative = ""] = line.split("|").map((part) => part.trim());
    if (!text) throw new Error(`Prompt ligne ${i + 1}: utiliser "id | prompt".`);
    return { id: slug(rawId), text, negative_prompt: negative };
  });
}

function parseSeeds() {
  const seeds = $("#seeds").value.split(/[\s,;]+/).filter(Boolean).map(Number);
  if (!seeds.length || seeds.some((seed) => !Number.isInteger(seed) || seed < 0)) {
    throw new Error("Les seeds doivent être des entiers positifs séparés par des virgules.");
  }
  return seeds;
}

function updateTrialCount() {
  try {
    const prompts = parsePrompts().length;
    const seeds = parseSeeds().length;
    const methods = state.methods.filter((method) => method.enabled).length;
    const total = prompts * seeds * methods;
    $("#trial-count").textContent = `${total} essai${total > 1 ? "s" : ""}`;
    $("#launch").disabled = total === 0 || total > 500;
    $("#launch-hint").textContent = total > 500
      ? "Maximum 500 essais par campagne."
      : "Les calculs seront exécutés un par un sur le GPU.";
  } catch (_) {
    $("#trial-count").textContent = "Configuration incomplète";
  }
}

function showComposeError(message) { $("#compose-error").textContent = message; }
function clearComposeError() { $("#compose-error").textContent = ""; }

async function launchCampaign() {
  clearComposeError();
  const button = $("#launch");
  button.disabled = true;
  button.textContent = "Mise en file…";
  try {
    const methods = state.methods.filter((method) => method.enabled).map((method) => ({
      id: method.id,
      name: method.name,
      backend: method.backend,
      enabled: true,
      output_variant: method.output_variant || "auto",
      reuse_stage1: method.reuse_stage1 !== false,
      generation: method.generation,
      model: method.model || {},
      tools: {
        srpg_enabled: !!method.tools?.srpg_enabled,
        guided_rediffusion_enabled: !!method.tools?.guided_rediffusion_enabled,
        latent_refinement_enabled: !!method.tools?.latent_refinement_enabled,
        settings: method.tools?.settings || {},
      },
    }));
    const body = {
      name: $("#campaign-name").value.trim(),
      payload: $("#payload").value.trim(),
      error_correction: $("#ecc").value,
      prompts: parsePrompts(),
      seeds: parseSeeds(),
      methods,
      max_attempts: Number($("#max-attempts").value),
    };
    const campaign = await api("/v1/lab/campaigns", { method: "POST", body: JSON.stringify(body) });
    await loadCampaigns();
    selectCampaign(campaign.id);
    switchTab("results");
  } catch (error) {
    showComposeError(error.message);
  } finally {
    button.textContent = "Lancer la campagne";
    updateTrialCount();
  }
}

function statusLabel(status) {
  const labels = {
    queued: "En attente", running: "En cours", completed: "Terminée",
    completed_with_errors: "Terminée avec erreurs", interrupted: "Interrompue",
    cancelled: "Arrêtée", accepted: "Accepté", rejected: "Rejeté", error: "Erreur",
  };
  return labels[status] || status;
}

async function loadCampaigns() {
  state.campaigns = await api("/v1/lab/campaigns");
  $("#campaign-count").textContent = state.campaigns.length;
  const list = $("#campaign-list");
  list.innerHTML = "";
  state.campaigns.forEach((campaign) => {
    const button = document.createElement("button");
    button.className = `campaign-item${state.selectedCampaign?.id === campaign.id ? " active" : ""}`;
    const percent = campaign.total_trials ? Math.round(100 * campaign.completed_trials / campaign.total_trials) : 0;
    button.innerHTML = `<strong></strong><span></span>`;
    $("strong", button).textContent = campaign.name;
    $("span", button).textContent = `${statusLabel(campaign.status)} · ${percent}% · ${campaign.accepted_trials}/${campaign.total_trials} acceptés`;
    button.addEventListener("click", () => selectCampaign(campaign.id));
    list.append(button);
  });
}

async function selectCampaign(id) {
  state.selectedCampaign = await api(`/v1/lab/campaigns/${id}`);
  renderCampaign();
  await loadCampaigns();
  clearInterval(state.poller);
  if (["queued", "running"].includes(state.selectedCampaign.status)) {
    state.poller = setInterval(async () => {
      state.selectedCampaign = await api(`/v1/lab/campaigns/${id}`);
      renderCampaign();
      if (!["queued", "running"].includes(state.selectedCampaign.status)) {
        clearInterval(state.poller);
        loadCampaigns();
      }
    }, 3000);
  }
}

function mean(values) {
  const clean = values.filter((value) => value !== null && value !== undefined && Number.isFinite(Number(value))).map(Number);
  return clean.length ? clean.reduce((a, b) => a + b, 0) / clean.length : null;
}

function formatPercent(value) { return value == null ? "—" : `${(100 * value).toFixed(1)}%`; }
function formatTime(ms) { return ms == null ? "—" : `${(ms / 1000).toFixed(1)} s`; }
function formatScore(value) { return value == null ? "—" : Number(value).toFixed(2); }
function isTechnicalControl(trial) {
  return trial.configuration?.method?.backend === "qr";
}

function renderCampaign() {
  const campaign = state.selectedCampaign;
  $("#empty-results").hidden = true;
  $("#campaign-content").hidden = false;
  $("#campaign-title").textContent = campaign.name;
  $("#campaign-status").textContent = statusLabel(campaign.status).toUpperCase();
  $("#campaign-meta").textContent = `${campaign.completed_trials}/${campaign.total_trials} essais · ${campaign.accepted_trials} acceptés · payload ${campaign.payload_hash.slice(0, 10)}…`;
  $("#campaign-progress").style.width = `${campaign.total_trials ? 100 * campaign.completed_trials / campaign.total_trials : 0}%`;
  $("#export-csv").href = `/v1/lab/campaigns/${campaign.id}/results.csv`;
  $("#cancel-campaign").hidden = !["queued", "running"].includes(campaign.status);
  renderMethodSummary(campaign.trials);
  renderCharts(campaign.trials);
  renderTrials();
}

function renderMethodSummary(trials) {
  const groups = Object.groupBy
    ? Object.groupBy(trials, (trial) => trial.method_id)
    : trials.reduce((acc, trial) => ((acc[trial.method_id] ||= []).push(trial), acc), {});
  const container = $("#method-summary");
  container.innerHTML = "";
  Object.entries(groups).forEach(([method, rows]) => {
    const finished = rows.filter((row) => ["accepted", "rejected", "error"].includes(row.status));
    const accepted = rows.filter((row) => row.status === "accepted").length;
    const scan = mean(finished.map((row) => row.generation?.scan_pass_rate));
    const time = mean(finished.map((row) => row.generation?.total_ms));
    const card = document.createElement("article");
    card.className = "summary-card";
    const technicalControl = rows.every(isTechnicalControl);
    card.innerHTML = `
      <strong></strong>
      ${technicalControl ? '<div class="muted">TÉMOIN TECHNIQUE — HORS CLASSEMENT</div>' : ""}
      <div class="summary-line"><span>Stricts</span><b>${accepted}/${rows.length}</b></div>
      <div class="summary-line"><span>SSR moyen</span><b>${formatPercent(scan)}</b></div>
      <div class="summary-line"><span>Temps moyen</span><b>${formatTime(time)}</b></div>
      <div class="mini-bar"><span style="width:${scan == null ? 0 : scan * 100}%"></span></div>`;
    $("strong", card).textContent = method;
    container.append(card);
  });
}

const SVG_NS = "http://www.w3.org/2000/svg";
const CHART_COLORS = ["#65e6ba", "#38bdf8", "#fbbf24", "#fb7185", "#c084fc", "#fb923c"];

function svgElement(name, attributes = {}, text = null) {
  const node = document.createElementNS(SVG_NS, name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
  if (text !== null) node.textContent = text;
  return node;
}

function emptyChart(svg, message) {
  svg.replaceChildren();
  svg.setAttribute("viewBox", "0 0 620 220");
  svg.append(svgElement("text", {
    x: 310, y: 110, "text-anchor": "middle", class: "chart-axis-label",
  }, message));
}

function renderMethodChart(trials) {
  const svg = $("#method-chart");
  const complete = trials.filter((trial) => trial.generation && !isTechnicalControl(trial));
  if (!complete.length) {
    emptyChart(svg, "Les points apparaîtront après les premières validations.");
    return;
  }
  const groups = complete.reduce((result, trial) => {
    (result[trial.method_id] ||= []).push(trial);
    return result;
  }, {});
  const rows = Object.entries(groups).map(([method, items]) => ({
    method,
    ssr: mean(items.map((item) => item.generation?.scan_pass_rate)) ?? 0,
    strict: items.filter((item) => item.status === "accepted").length / items.length,
  }));
  const width = 620;
  const height = 240;
  const left = 42;
  const right = 14;
  const top = 16;
  const bottom = 54;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  [0, 0.25, 0.5, 0.75, 1].forEach((tick) => {
    const y = top + plotHeight * (1 - tick);
    svg.append(svgElement("line", {
      x1: left, y1: y, x2: width - right, y2: y, class: "chart-grid-line",
    }));
    svg.append(svgElement("text", {
      x: left - 7, y: y + 3, "text-anchor": "end", class: "chart-tick",
    }, `${Math.round(tick * 100)}%`));
  });
  const groupWidth = plotWidth / rows.length;
  const barWidth = Math.min(32, groupWidth * 0.28);
  rows.forEach((row, index) => {
    const center = left + groupWidth * (index + 0.5);
    [
      { value: row.ssr, x: center - barWidth - 2, color: "#38bdf8", label: "SSR moyen" },
      { value: row.strict, x: center + 2, color: "#65e6ba", label: "Acceptés stricts" },
    ].forEach((bar) => {
      const barHeight = plotHeight * bar.value;
      const rect = svgElement("rect", {
        x: bar.x, y: top + plotHeight - barHeight, width: barWidth,
        height: barHeight, rx: 3, fill: bar.color,
      });
      rect.append(svgElement("title", {}, `${row.method} · ${bar.label}: ${formatPercent(bar.value)}`));
      svg.append(rect);
    });
    svg.append(svgElement("text", {
      x: center, y: height - 31, "text-anchor": "middle", class: "chart-method-label",
    }, row.method.length > 18 ? `${row.method.slice(0, 17)}…` : row.method));
  });
  svg.append(svgElement("text", {
    x: left, y: height - 8, class: "chart-tick", fill: "#38bdf8",
  }, "■ SSR moyen"));
  svg.append(svgElement("text", {
    x: left + 100, y: height - 8, class: "chart-tick", fill: "#65e6ba",
  }, "■ Acceptés stricts"));
}

function renderTradeoffChart(trials) {
  const svg = $("#tradeoff-chart");
  const complete = trials.filter((trial) =>
    !isTechnicalControl(trial)
      && trial.generation?.scan_pass_rate != null
      && trial.generation?.module_error_rate != null
  );
  if (!complete.length) {
    emptyChart(svg, "Les points apparaîtront après les premières validations.");
    return;
  }
  const width = 620;
  const height = 240;
  const left = 48;
  const right = 20;
  const top = 16;
  const bottom = 42;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const methods = [...new Set(complete.map((trial) => trial.method_id))];
  const maxError = Math.max(0.05, ...complete.map((trial) => trial.generation.module_error_rate));
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  [0, 0.25, 0.5, 0.75, 1].forEach((tick) => {
    const y = top + plotHeight * (1 - tick);
    svg.append(svgElement("line", {
      x1: left, y1: y, x2: width - right, y2: y, class: "chart-grid-line",
    }));
    svg.append(svgElement("text", {
      x: left - 7, y: y + 3, "text-anchor": "end", class: "chart-tick",
    }, `${Math.round(tick * 100)}%`));
  });
  complete.forEach((trial) => {
    const x = left + plotWidth * trial.generation.module_error_rate / maxError;
    const y = top + plotHeight * (1 - trial.generation.scan_pass_rate);
    const color = CHART_COLORS[methods.indexOf(trial.method_id) % CHART_COLORS.length];
    const radius = trial.rating?.overall_score ? 4 + trial.rating.overall_score * 0.35 : 5;
    const circle = svgElement("circle", {
      cx: x, cy: y, r: radius, fill: color, opacity: 0.82,
      stroke: trial.status === "accepted" ? "#f3f6fa" : "none",
      "stroke-width": 1.2,
    });
    circle.append(svgElement(
      "title",
      {},
      `${trial.prompt_id} · ${trial.method_id} · seed ${trial.seed}\n`
        + `SSR ${formatPercent(trial.generation.scan_pass_rate)} · `
        + `MER ${formatPercent(trial.generation.module_error_rate)}`
        + (trial.rating?.overall_score ? ` · note ${trial.rating.overall_score}/10` : ""),
    ));
    svg.append(circle);
  });
  svg.append(svgElement("text", {
    x: width / 2, y: height - 7, "text-anchor": "middle", class: "chart-axis-label",
  }, `Erreur modules (0 à ${formatPercent(maxError)}) →`));
  svg.append(svgElement("text", {
    x: 11, y: height / 2, transform: `rotate(-90 11 ${height / 2})`,
    "text-anchor": "middle", class: "chart-axis-label",
  }, "SSR robuste"));
}

function renderCharts(trials) {
  renderMethodChart(trials);
  renderTradeoffChart(trials);
}

function visibleTrials() {
  const filter = $("#status-filter").value;
  let rows = [...(state.selectedCampaign?.trials || [])];
  rows = rows.filter((trial) => {
    if (filter === "all") return true;
    if (filter === "rated") return !!trial.rating;
    if (filter === "favorite") return !!trial.rating?.favorite;
    return trial.status === filter;
  });
  const sort = $("#sort-results").value;
  if (sort === "scan") rows.sort((a, b) => (b.generation?.scan_pass_rate ?? -1) - (a.generation?.scan_pass_rate ?? -1));
  if (sort === "rating") rows.sort((a, b) => (b.rating?.overall_score ?? -1) - (a.rating?.overall_score ?? -1));
  if (sort === "time") rows.sort((a, b) => (a.generation?.total_ms ?? Infinity) - (b.generation?.total_ms ?? Infinity));
  return rows;
}

function renderTrials() {
  const grid = $("#trial-grid");
  grid.innerHTML = "";
  visibleTrials().forEach((trial) => {
    const card = document.createElement("article");
    card.className = "trial-card";
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.setAttribute(
      "aria-label",
      `Ouvrir ${trial.prompt_id}, ${trial.method_id}, seed ${trial.seed}`,
    );
    const image = trial.generation?.image_url
      ? `<img loading="lazy" src="${trial.generation.image_url}" alt="Résultat ${trial.prompt_id} ${trial.method_id}">`
      : `<span class="trial-placeholder">${statusLabel(trial.status)}</span>`;
    card.innerHTML = `
      <div class="trial-image">${image}</div>
      <div class="trial-body">
        <div class="trial-title"><strong></strong><span class="status ${trial.status}">${statusLabel(trial.status)}</span></div>
        <div class="muted">${trial.prompt_id} · seed ${trial.seed} · sortie ${trial.generation?.selected_variant || "—"}${trial.generation?.quality_metrics?.stage1_mean_absolute_change == null ? "" : ` · Δ Stage 1 ${formatPercent(trial.generation.quality_metrics.stage1_mean_absolute_change)}`}${trial.rating?.favorite ? " · ★" : ""}</div>
        <div class="trial-metrics">
          <span>SSR<b>${formatPercent(trial.generation?.scan_pass_rate)}</b></span>
          <span>MER<b>${formatPercent(trial.generation?.module_error_rate)}</b></span>
          <span>Note<b>${trial.rating?.overall_score ?? "—"}</b></span>
        </div>
      </div>`;
    $(".trial-title strong", card).textContent = trial.method_id;
    card.addEventListener("click", () => openTrial(trial.id));
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openTrial(trial.id);
      }
    });
    grid.append(card);
  });
}

async function openTrial(id) {
  const trial = await api(`/v1/lab/trials/${id}`);
  state.selectedTrial = trial;
  $("#dialog-kicker").textContent = `${statusLabel(trial.status)} · seed ${trial.seed}`;
  $("#dialog-title").textContent = `${trial.prompt_id} / ${trial.method_id}`;
  const run = trial.generation;
  const metrics = [
    ["Sortie réellement notée", run?.selected_variant || "—"],
    ["Sélection", run?.selection_mode === "forced" ? "Candidat forcé" : "Livraison automatique"],
    ["Stage 1", run?.stage1_reused ? "Réutilisé — aucune régénération" : "Généré pour cet essai"],
    ["SSR robuste", formatPercent(run?.scan_pass_rate)],
    ["Payload original", run?.exact_payload_match == null ? "—" : run.exact_payload_match ? "Exact" : "Échec"],
    ["MER", formatPercent(run?.module_error_rate)],
    ["Génération", formatTime(run?.generation_ms)],
    ["Validation", formatTime(run?.validation_ms)],
    ["Total", formatTime(run?.total_ms)],
  ];
  Object.entries(run?.quality_metrics || {}).sort(([left], [right]) =>
    left.localeCompare(right)
  ).forEach(([name, value]) => {
    metrics.push([name.replaceAll("_", " "), formatScore(value)]);
  });
  $("#dialog-metrics").innerHTML = metrics.map(([label, value]) =>
    `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`).join("");
  const gallery = $("#artifact-gallery");
  gallery.innerHTML = "";
  if (run) {
    const artifacts = await api(`/v1/generations/${run.id}/artifacts`);
    artifacts.forEach((artifact) => {
      const figure = document.createElement("figure");
      figure.className = "artifact";
      figure.innerHTML = `<img loading="lazy" src="${artifact.url}" alt=""><figcaption></figcaption>`;
      $("img", figure).alt = artifact.name;
      $("figcaption", figure).textContent = artifact.name === "final"
        ? `RÉSULTAT ÉVALUÉ — ${run.selected_variant || "inconnu"}`
        : artifact.name === "stage1_raw"
          ? "STAGE 1 PARTAGÉ — référence artistique"
          : artifact.name;
      gallery.append(figure);
    });
  }
  const rating = trial.rating || {};
  $("#rating-aesthetic").value = rating.aesthetic_score ?? "";
  $("#rating-fidelity").value = rating.prompt_fidelity_score ?? "";
  $("#rating-discretion").value = rating.qr_discretion_score ?? "";
  $("#rating-overall").value = rating.overall_score ?? "";
  $("#rating-favorite").checked = !!rating.favorite;
  $("#rating-notes").value = rating.notes ?? "";
  $("#physical-payload").value = "";
  $("#physical-device").value = "";
  $("#physical-notes").value = "";
  $("#dialog-message").textContent = "";
  $("#save-physical").disabled = !run;
  $("#trial-dialog").showModal();
}

function nullableNumber(selector) {
  const value = $(selector).value;
  return value === "" ? null : Number(value);
}

async function saveRating() {
  const trial = state.selectedTrial;
  const rating = await api(`/v1/lab/trials/${trial.id}/rating`, {
    method: "PUT",
    body: JSON.stringify({
      aesthetic_score: nullableNumber("#rating-aesthetic"),
      prompt_fidelity_score: nullableNumber("#rating-fidelity"),
      qr_discretion_score: nullableNumber("#rating-discretion"),
      overall_score: nullableNumber("#rating-overall"),
      favorite: $("#rating-favorite").checked,
      notes: $("#rating-notes").value,
    }),
  });
  $("#dialog-message").textContent = "Note enregistrée.";
  state.selectedTrial.rating = rating;
  const row = state.selectedCampaign.trials.find((item) => item.id === trial.id);
  if (row) row.rating = rating;
  renderCharts(state.selectedCampaign.trials);
  renderTrials();
}

async function savePhysical() {
  const run = state.selectedTrial?.generation;
  if (!run) return;
  const decoded = $("#physical-payload").value.trim();
  const result = await api(`/v1/generations/${run.id}/physical-validations`, {
    method: "POST",
    body: JSON.stringify({
      decoded_payload: decoded || null,
      device: $("#physical-device").value.trim() || "unknown",
      material: $("#physical-material").value,
      lighting: $("#physical-lighting").value.trim() || "unknown",
      notes: $("#physical-notes").value,
    }),
  });
  $("#dialog-message").textContent = `Scan enregistré : ${result.outcome}.`;
}

async function init() {
  $$(".tab").forEach((button) => button.addEventListener("click", () => switchTab(button.dataset.tab)));
  ["prompts", "seeds"].forEach((id) => $(`#${id}`).addEventListener("input", updateTrialCount));
  $("#launch").addEventListener("click", launchCampaign);
  $("#add-method").addEventListener("click", () => {
    state.methods.push({
      id: `method_${Date.now().toString().slice(-6)}`, name: "Nouvelle méthode",
      backend: "controlnet", enabled: true,
      output_variant: "raw", reuse_stage1: true,
      generation: { steps: 40, guidance_scale: 7.5, controlnet_scale: 1.35, strength: 1 },
      model: {}, tools: { settings: {} }, description: "Configuration personnalisée.",
    });
    renderMethods();
  });
  $("#refresh-campaigns").addEventListener("click", loadCampaigns);
  $("#status-filter").addEventListener("change", renderTrials);
  $("#sort-results").addEventListener("change", renderTrials);
  $("#cancel-campaign").addEventListener("click", async () => {
    await api(`/v1/lab/campaigns/${state.selectedCampaign.id}/cancel`, { method: "POST" });
    selectCampaign(state.selectedCampaign.id);
  });
  $("#save-rating").addEventListener("click", () => saveRating().catch((error) => $("#dialog-message").textContent = error.message));
  $("#save-physical").addEventListener("click", () => savePhysical().catch((error) => $("#dialog-message").textContent = error.message));

  try {
    const [schema, runtime] = await Promise.all([api("/v1/lab/schema"), api("/v1/runtime")]);
    state.schema = schema;
    state.methods = schema.profiles.map(deepCopy);
    renderMethods();
    const gpu = runtime.cuda_available ? runtime.cuda_device || "CUDA" : "CPU";
    const quality = schema.quality_scoring?.clip_enabled ? "CLIP CPU" : "CLIP désactivé";
    $("#runtime-status").textContent = `${runtime.environment || "API"} · ${gpu} · ${quality}`;
    $("#runtime-status").classList.add("ok");
    await loadCampaigns();
  } catch (error) {
    $("#runtime-status").textContent = `API indisponible · ${error.message}`;
    showComposeError(error.message);
  }
}

document.addEventListener("DOMContentLoaded", init);
