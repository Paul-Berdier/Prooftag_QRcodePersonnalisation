const state = {
  schema: null,
  methods: [],
  campaigns: [],
  campaign: null,
  visibleTrials: [],
  selectedIndex: -1,
  poller: null,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const copy = (value) => JSON.parse(JSON.stringify(value));
const OUTPUT_VARIANTS = new Set(["raw", "srpg", "srmpgd", "auto"]);

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {"Content-Type": "application/json", ...(options.headers || {})},
    ...options,
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_) {}
    throw new Error(message);
  }
  return response.status === 204 ? null : response.json();
}

function slug(text) {
  return text.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "").slice(0, 90) || "prompt";
}

function fmt(value, digits = 1) {
  return value == null || Number.isNaN(Number(value)) ? "—" : Number(value).toFixed(digits);
}

function pct(value) {
  return value == null ? "—" : `${(Number(value) * 100).toFixed(1)}%`;
}

function parsePrompts() {
  return $("#prompts").value.split(/\r?\n/).map(v => v.trim()).filter(Boolean).map((line, index) => {
    const [id, text, negative = ""] = line.split("|").map(v => v.trim());
    if (!text) throw new Error(`Ligne ${index + 1} : utiliser "id | prompt | négatif".`);
    return {id: slug(id), text, negative_prompt: negative};
  });
}

function parseSeeds() {
  const values = $("#seeds").value.split(",").map(v => Number(v.trim())).filter(Number.isFinite);
  if (!values.length) throw new Error("Ajoute au moins une seed.");
  return [...new Set(values)];
}

function updateCount() {
  try {
    const count = parsePrompts().length * parseSeeds().length * state.methods.filter(m => m.enabled).length;
    $("#trial-count").textContent = `${count} résultat${count > 1 ? "s" : ""}`;
    $("#launch-summary").textContent = `${count} génération${count > 1 ? "s" : ""} planifiée${count > 1 ? "s" : ""}`;
  } catch (_) {
    $("#trial-count").textContent = "Configuration incomplète";
  }
}

function normalizeMethod(method) {
  method.tools ||= {settings: {}};
  method.tools.settings ||= {};
  method.generation ||= {};
  method.model ||= {};
  if (!OUTPUT_VARIANTS.has(method.output_variant)) {
    method.output_variant = method.tools.srmpgd_enabled
      ? "srmpgd"
      : method.tools.srpg_enabled ? "srpg" : "raw";
  }
  return method;
}

function ensureOutputOption(select, value) {
  if (![...select.options].some(option => option.value === value)) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value === "auto" ? "Meilleur Stage 1 / Stage 2" : value;
    select.append(option);
  }
  select.value = value;
}

function renderMethods() {
  const host = $("#methods");
  host.innerHTML = "";
  state.methods.forEach((method, index) => {
    normalizeMethod(method);
    const node = $("#method-template").content.firstElementChild.cloneNode(true);
    const settings = method.tools.settings;
    $(".enabled", node).checked = method.enabled;
    $(".method-label", node).textContent = method.name;
    $(".description", node).textContent = method.description || "";
    $(".name", node).value = method.name;
    $(".id", node).value = method.id;
    ensureOutputOption($(".output", node), method.output_variant);
    $(".steps", node).value = method.generation.steps ?? 40;
    $(".cfg", node).value = method.generation.guidance_scale ?? 7.5;
    $(".control", node).value = method.generation.controlnet_scale ?? 1.35;
    $(".stage2-init", node).value = settings.diffqrcoder_stage2_initialization ?? "paper_stage1_noise";
    $(".stage2-target", node).value = settings.diffqrcoder_stage2_target_mode ?? "binary_exact";
    $(".stage2-strength", node).value = settings.diffqrcoder_stage2_strength ?? 0.65;
    $(".srpg-steps", node).value = settings.srpg_steps ?? 40;
    $(".srpg-control", node).value = settings.srpg_controlnet_scale ?? 1.35;
    $(".srg", node).value = settings.srpg_qr_weight ?? 500;
    $(".pg", node).value = settings.srpg_perceptual_weight ?? 2;
    $(".eta", node).value = settings.srpg_eta ?? 0;
    $(".seed-offset", node).value = settings.srpg_seed_offset ?? 2000003;
    $(".preview", node).value = settings.srpg_preview_interval ?? 5;
    $(".mpgd-iterations", node).value = settings.srmpgd_max_iterations ?? 4;
    $(".mpgd-lr", node).value = settings.srmpgd_step_size ?? 100;
    $(".mpgd-lpips", node).value = settings.srmpgd_lpips_weight ?? 0.10;
    $(".mpgd-max-mer", node).value = settings.srmpgd_max_initial_module_error_rate ?? 0.12;
    $(".mpgd-max-step-rms", node).value = settings.srmpgd_max_step_rms ?? 0.02;
    $(".mpgd-max-total-rms", node).value = settings.srmpgd_max_total_delta_rms ?? 0.06;
    $(".mpgd-min-improvement", node).value = settings.srmpgd_min_relative_module_improvement ?? 0.01;
    $(".mpgd-max-lpips", node).value = settings.srmpgd_max_lpips_loss ?? 0.15;
    $(".mpgd-max-change", node).value = settings.srmpgd_max_mean_absolute_change ?? 0.06;
    $(".mpgd-max-saturation", node).value = settings.srmpgd_max_saturation_mean_increase ?? 0.04;
    $(".mpgd-max-high-saturation", node).value = settings.srmpgd_max_high_saturation_ratio_increase ?? 0.05;
    $(".mpgd-max-rgb-clipping", node).value = settings.srmpgd_max_rgb_clipped_channel_ratio_increase ?? 0.01;
    $(".mpgd-robust-blur-weight", node).value = settings.srmpgd_robust_blur_weight ?? 0;
    $(".mpgd-robust-blur-kernel", node).value = settings.srmpgd_robust_blur_kernel ?? 3;
    $(".mpgd-robust-downscale-weight", node).value = settings.srmpgd_robust_downscale_weight ?? 0;
    $(".mpgd-robust-downscale-factor", node).value = settings.srmpgd_robust_downscale_factor ?? 0.75;
    $(".mpgd-robust-brightness-weight", node).value = settings.srmpgd_robust_brightness_weight ?? 0;
    $(".mpgd-robust-brightness-low", node).value = settings.srmpgd_robust_brightness_low ?? 0.80;
    $(".mpgd-robust-brightness-high", node).value = settings.srmpgd_robust_brightness_high ?? 1.20;
    $(".mpgd-robust-contrast-weight", node).value = settings.srmpgd_robust_contrast_weight ?? 0;
    $(".mpgd-robust-contrast-factor", node).value = settings.srmpgd_robust_contrast_factor ?? 0.75;
    $(".warn-saturation", node).value = settings.diffqrcoder_guard_max_saturation_mean_increase ?? 0.08;
    $(".hard-saturation", node).value = settings.diffqrcoder_guard_hard_max_saturation_mean_increase ?? 0.20;
    $(".warn-rgb-clipping", node).value = settings.diffqrcoder_guard_max_rgb_clipped_channel_ratio_increase ?? 0.02;
    $(".hard-rgb-clipping", node).value = settings.diffqrcoder_guard_hard_max_rgb_clipped_channel_ratio_increase ?? 0.25;
    $(".model-json", node).value = JSON.stringify(method.model, null, 2);
    node.classList.toggle("disabled", !method.enabled);
    node.classList.toggle("has-stage2", method.output_variant !== "raw");
    node.classList.toggle("has-srmpgd", method.output_variant === "srmpgd");

    const sync = () => {
      const item = state.methods[index];
      item.enabled = $(".enabled", node).checked;
      item.name = $(".name", node).value.trim();
      item.id = slug($(".id", node).value);
      const selectedOutput = $(".output", node).value;
      item.output_variant = OUTPUT_VARIANTS.has(selectedOutput)
        ? selectedOutput
        : OUTPUT_VARIANTS.has(item.output_variant) ? item.output_variant : "raw";
      item.reuse_stage1 = true;
      item.generation = {
        steps: Number($(".steps", node).value),
        guidance_scale: Number($(".cfg", node).value),
        controlnet_scale: Number($(".control", node).value),
        strength: 1,
      };
      item.tools.srpg_enabled = item.output_variant !== "raw";
      item.tools.srmpgd_enabled = item.output_variant === "srmpgd";
      item.tools.guided_rediffusion_enabled = false;
      item.tools.latent_refinement_enabled = false;
      item.tools.settings = item.output_variant === "raw" ? {} : {
        diffqrcoder_stage2_initialization: $(".stage2-init", node).value,
        diffqrcoder_stage2_target_mode: $(".stage2-target", node).value,
        diffqrcoder_qart_thresholds: [96, 112, 128, 144, 160],
        diffqrcoder_stage2_strength: Number($(".stage2-strength", node).value),
        srpg_steps: Number($(".srpg-steps", node).value),
        srpg_controlnet_scale: Number($(".srpg-control", node).value),
        srpg_qr_weight: Number($(".srg", node).value),
        srpg_perceptual_weight: Number($(".pg", node).value),
        diffqrcoder_guard_max_changed_pixel_ratio: 0.995,
        diffqrcoder_guard_max_mean_absolute_change: 0.35,
        diffqrcoder_guard_max_clipped_pixel_ratio_increase: 0.05,
        diffqrcoder_guard_max_rgb_clipped_channel_ratio_increase: Number($(".warn-rgb-clipping", node).value),
        diffqrcoder_guard_max_saturation_mean_increase: Number($(".warn-saturation", node).value),
        diffqrcoder_guard_max_high_saturation_ratio_increase: 0.05,
        diffqrcoder_guard_hard_max_mean_absolute_change: 0.40,
        diffqrcoder_guard_hard_max_clipped_pixel_ratio_increase: 0.20,
        diffqrcoder_guard_hard_max_rgb_clipped_channel_ratio_increase: Number($(".hard-rgb-clipping", node).value),
        diffqrcoder_guard_hard_max_saturation_mean_increase: Number($(".hard-saturation", node).value),
        diffqrcoder_guard_hard_max_high_saturation_ratio_increase: 0.30,
        srpg_eta: Number($(".eta", node).value),
        srpg_seed_offset: Number($(".seed-offset", node).value),
        srpg_save_step_previews: true,
        srpg_preview_interval: Number($(".preview", node).value),
        diffqrcoder_control_guidance_start: 0,
        diffqrcoder_control_guidance_end: 1,
        ...(item.output_variant === "srmpgd" ? {
          srmpgd_max_iterations: Number($(".mpgd-iterations", node).value),
          srmpgd_step_size: Number($(".mpgd-lr", node).value),
          srmpgd_lpips_weight: Number($(".mpgd-lpips", node).value),
          srmpgd_lpips_net: "vgg",
          srmpgd_crop_padding_px: 78,
          srmpgd_dark_threshold: 0.45,
          srmpgd_light_threshold: 0.65,
          srmpgd_center_fraction: 1 / 3,
          srmpgd_max_initial_module_error_rate: Number($(".mpgd-max-mer", node).value),
          srmpgd_max_step_rms: Number($(".mpgd-max-step-rms", node).value),
          srmpgd_max_total_delta_rms: Number($(".mpgd-max-total-rms", node).value),
          srmpgd_min_relative_module_improvement: Number($(".mpgd-min-improvement", node).value),
          srmpgd_max_lpips_loss: Number($(".mpgd-max-lpips", node).value),
          srmpgd_max_mean_absolute_change: Number($(".mpgd-max-change", node).value),
          srmpgd_max_saturation_mean_increase: Number($(".mpgd-max-saturation", node).value),
          srmpgd_max_high_saturation_ratio_increase: Number($(".mpgd-max-high-saturation", node).value),
          srmpgd_max_rgb_clipped_channel_ratio_increase: Number($(".mpgd-max-rgb-clipping", node).value),
          srmpgd_robust_blur_weight: Number($(".mpgd-robust-blur-weight", node).value),
          srmpgd_robust_blur_kernel: Number($(".mpgd-robust-blur-kernel", node).value),
          srmpgd_robust_downscale_weight: Number($(".mpgd-robust-downscale-weight", node).value),
          srmpgd_robust_downscale_factor: Number($(".mpgd-robust-downscale-factor", node).value),
          srmpgd_robust_brightness_weight: Number($(".mpgd-robust-brightness-weight", node).value),
          srmpgd_robust_brightness_low: Number($(".mpgd-robust-brightness-low", node).value),
          srmpgd_robust_brightness_high: Number($(".mpgd-robust-brightness-high", node).value),
          srmpgd_robust_contrast_weight: Number($(".mpgd-robust-contrast-weight", node).value),
          srmpgd_robust_contrast_factor: Number($(".mpgd-robust-contrast-factor", node).value),
        } : {}),
      };
      $(".method-label", node).textContent = item.name || "Sans nom";
      node.classList.toggle("disabled", !item.enabled);
      node.classList.toggle("has-stage2", item.output_variant !== "raw");
      node.classList.toggle("has-srmpgd", item.output_variant === "srmpgd");
      updateCount();
    };
    $$("input, select", node).forEach(input => input.addEventListener("input", sync));
    $(".model-json", node).addEventListener("change", () => {
      try {
        state.methods[index].model = JSON.parse($(".model-json", node).value);
        $("#compose-error").textContent = "";
      } catch (error) {
        $("#compose-error").textContent = `JSON modèle invalide : ${error.message}`;
      }
    });
    $(".remove", node).addEventListener("click", () => {
      state.methods.splice(index, 1);
      renderMethods();
    });
    host.append(node);
  });
  updateCount();
}

function campaignPayload() {
  state.methods.forEach((method, index) => {
    if (!OUTPUT_VARIANTS.has(method.output_variant)) {
      throw new Error(
        `Méthode ${index + 1} (${method.name || method.id}) : sortie invalide. ` +
        "Recharge la page avec Ctrl+F5."
      );
    }
  });
  return {
    name: $("#campaign-name").value.trim(),
    payload: $("#payload").value.trim(),
    error_correction: $("#ecc").value,
    prompts: parsePrompts(),
    seeds: parseSeeds(),
    methods: state.methods,
    max_attempts: Number($("#max-attempts").value),
  };
}

async function launch() {
  const button = $("#launch");
  $("#compose-error").textContent = "";
  button.disabled = true;
  try {
    const campaign = await api("/v1/lab/campaigns", {
      method: "POST",
      body: JSON.stringify(campaignPayload()),
    });
    await loadCampaigns();
    showPanel("results");
    await selectCampaign(campaign.id);
  } catch (error) {
    $("#compose-error").textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

function showPanel(id) {
  $$(".panel").forEach(panel => panel.classList.toggle("active", panel.id === id));
  $$(".tab").forEach(tab => tab.classList.toggle("active", tab.dataset.panel === id));
}

async function loadCampaigns() {
  state.campaigns = await api("/v1/lab/campaigns?limit=100");
  $("#campaign-count").textContent = state.campaigns.length;
  const host = $("#campaigns");
  host.innerHTML = state.campaigns.map(campaign => `
    <button class="campaign-item ${state.campaign?.id === campaign.id ? "active" : ""}" data-id="${campaign.id}">
      <b>${campaign.name}</b>
      <span>${campaign.completed_trials}/${campaign.total_trials} · ${campaign.status}</span>
    </button>`).join("");
  $$(".campaign-item", host).forEach(button => button.addEventListener("click", () => selectCampaign(button.dataset.id)));
}

function aggregate(trials) {
  const allGenerated = trials.filter(t => t.generation);
  const generated = allGenerated.filter(t => t.method_id !== "qr_reference");
  const qrVerified = generated.filter(t => Number(t.generation.quality_metrics?.qr_verify_available || 0) === 1);
  const rated = generated.filter(t => t.rating && (
    t.rating.aesthetic_ok != null || t.rating.human_scan_result !== "not_tested"
  ));
  const strict = qrVerified.filter(t => t.generation.exact_payload_match);
  const humanScan = generated.filter(t => t.rating?.human_scan_result === "scannable");
  const phoneAttempts = generated.reduce((sum, t) => sum + Number(t.rating?.human_scan_attempts || 0), 0);
  const phoneSuccesses = generated.reduce((sum, t) => sum + Number(t.rating?.human_scan_successes || 0), 0);
  const aesthetic = generated.filter(t => t.rating?.aesthetic_ok === true);
  const scored = (name) => generated.filter(t => {
    const value = t.generation.quality_metrics?.[name];
    return value != null && Number.isFinite(Number(value));
  });
  const mean = (items, getter) => items.length ? items.reduce((sum, item) => sum + Number(getter(item)), 0) / items.length : null;
  const clipAesthetic = scored("clip_aesthetic");
  const clipSimilarity = scored("clip_similarity");
  const clipScore = scored("clip_score");
  const hps = scored("hpsv2_1");
  return {
    generated: generated.length,
    qrVerified: qrVerified.length,
    rated: rated.length,
    strict: strict.length,
    humanScan: humanScan.length,
    phoneAttempts,
    phoneSuccesses,
    aesthetic: aesthetic.length,
    meanQrVerify: mean(qrVerified, t => t.generation.scan_pass_rate),
    directQrVerify: qrVerified.filter(t => Number(t.generation.quality_metrics?.qr_verify_direct_exact || 0) === 1).length,
    meanClipAesthetic: mean(clipAesthetic, t => t.generation.quality_metrics.clip_aesthetic),
    meanClipSimilarity: mean(clipSimilarity, t => t.generation.quality_metrics.clip_similarity),
    meanClipScore: mean(clipScore, t => t.generation.quality_metrics.clip_score),
    meanHps: mean(hps, t => t.generation.quality_metrics.hpsv2_1),
    qualityScored: clipAesthetic.length,
  };
}

function renderScores() {
  const a = aggregate(state.campaign.trials);
  $("#final-scores").innerHTML = [
    ["QR-Verify valides", `${a.strict}/${a.qrVerified}`],
    ["Score QR-Verify moyen", pct(a.meanQrVerify)],
    ["Lecture sans filtre", `${a.directQrVerify}/${a.qrVerified}`],
    ["CLIP-Aesthetic moyen", a.meanClipAesthetic == null ? "N/A" : fmt(a.meanClipAesthetic, 3)],
    ["CLIP sim. moyenne", a.meanClipSimilarity == null ? "N/A" : fmt(a.meanClipSimilarity, 4)],
    ["CLIPScore moyen", a.meanClipScore == null ? "N/A" : fmt(a.meanClipScore, 4)],
    ["HPS v2.1 moyen (indicatif)", a.meanHps == null ? "N/A" : fmt(a.meanHps, 4)],
    ["Tes scans positifs", `${a.humanScan}/${a.rated}`],
    ["Lectures téléphone", `${a.phoneSuccesses}/${a.phoneAttempts}`],
    ["Tes esthétiques positives", `${a.aesthetic}/${a.rated}`],
    ["Évalués", `${a.rated}/${a.generated}`],
  ].map(([label, value]) => `<article><span>${label}</span><b>${value}</b></article>`).join("");
}

function orderedTrials() {
  let trials = state.campaign.trials.filter(t => t.generation);
  const filter = $("#filter").value;
  if (filter === "unrated") trials = trials.filter(t => !t.rating || (t.rating.aesthetic_ok == null && t.rating.human_scan_result === "not_tested"));
  if (filter === "scannable" || filter === "not_scannable") trials = trials.filter(t => t.rating?.human_scan_result === filter);
  if (filter === "auto_pass") trials = trials.filter(t => t.generation.exact_payload_match);
  const sort = $("#sort").value;
  if (sort === "qrverify") trials.sort((a, b) => (b.generation.scan_pass_rate ?? -1) - (a.generation.scan_pass_rate ?? -1));
  if (sort === "aesthetic") trials.sort((a, b) => (b.generation.quality_metrics?.clip_aesthetic ?? -Infinity) - (a.generation.quality_metrics?.clip_aesthetic ?? -Infinity));
  if (sort === "clip") trials.sort((a, b) => (b.generation.quality_metrics?.clip_similarity ?? -Infinity) - (a.generation.quality_metrics?.clip_similarity ?? -Infinity));
  if (sort === "hps") trials.sort((a, b) => (b.generation.quality_metrics?.hpsv2_1 ?? -Infinity) - (a.generation.quality_metrics?.hpsv2_1 ?? -Infinity));
  if (sort === "time") trials.sort((a, b) => (a.generation.total_ms ?? Infinity) - (b.generation.total_ms ?? Infinity));
  return trials;
}

function renderTrials() {
  state.visibleTrials = orderedTrials();
  $("#trials").innerHTML = state.visibleTrials.map((trial, index) => {
    const run = trial.generation;
    const rating = trial.rating;
    const qrVerified = Number(run.quality_metrics?.qr_verify_available || 0) === 1;
    const diverged = Number(run.quality_metrics?.diffqrcoder_guard_diverged || 0) === 1;
    const warning = Number(run.quality_metrics?.diffqrcoder_guard_warning || 0) === 1;
    const fallback = Number(run.quality_metrics?.selection_preserved_stage1 || 0) === 1;
    const clipAesthetic = run.quality_metrics?.clip_aesthetic;
    const clipSimilarity = run.quality_metrics?.clip_similarity;
    const hps = run.quality_metrics?.hpsv2_1;
    return `<button class="trial" data-index="${index}">
      <div class="image">${run.image_url
        ? `<img src="${run.image_url}" loading="lazy" alt="${trial.prompt_id}">`
        : "<span class='missing-image'>Génération en erreur</span>"}</div>
      <div class="trial-body">
        <div class="trial-title"><b>${trial.prompt_id} / ${trial.method_id}</b><span class="status ${trial.status}">${trial.status}</span></div>
        <p class="${run.error ? "run-error" : ""}">${run.error || `seed ${trial.seed} · ${run.selected_variant}`}</p>
        <div class="trial-stats">
          <span>QR-Verify<b>${qrVerified ? pct(run.scan_pass_rate) : "N/A historique"}</b></span>
          <span>Au moins un exact<b>${qrVerified ? (run.exact_payload_match ? "OUI" : "NON") : "N/A"}</b></span>
          <span>Sans filtre<b>${qrVerified ? (Number(run.quality_metrics?.qr_verify_direct_exact || 0) === 1 ? "OUI" : "NON") : "N/A"}</b></span>
          <span>Presets exacts<b>${qrVerified ? `${fmt(run.quality_metrics?.qr_verify_exact_presets, 0)}/${fmt(run.quality_metrics?.qr_verify_supported_presets, 0)}` : "N/A"}</b></span>
          <span>MER<b>${pct(run.module_error_rate)}</b></span>
          <span>CLIP-AES<b>${clipAesthetic == null ? "N/A" : fmt(clipAesthetic, 3)}</b></span>
          <span>CLIP sim.<b>${clipSimilarity == null ? "N/A" : fmt(clipSimilarity, 4)}</b></span>
          <span>HPS v2.1<b>${hps == null ? "N/A" : fmt(hps, 4)}</b></span>
        </div>
        <div class="badges">
          ${diverged ? "<i class='bad'>Divergence Stage 2</i>" : ""}
          ${!diverged && warning ? "<i>Alerte couleur</i>" : ""}
          ${fallback ? "<i class='good'>Stage 1 préservé</i>" : ""}
          ${rating?.aesthetic_ok === true ? "<i class='good'>Esthétique ✓</i>" : rating?.aesthetic_ok === false ? "<i class='bad'>Esthétique ✕</i>" : "<i>Esthétique ?</i>"}
          ${rating?.human_scan_result === "scannable" ? `<i class='good'>Scan ✓ ${rating.human_scan_successes || 0}/${rating.human_scan_attempts || 0}</i>` : rating?.human_scan_result === "not_scannable" ? `<i class='bad'>Scan ✕ ${rating.human_scan_successes || 0}/${rating.human_scan_attempts || 0}</i>` : "<i>Scan ?</i>"}
        </div>
      </div>
    </button>`;
  }).join("");
  $$(".trial", $("#trials")).forEach(button => button.addEventListener("click", () => openTrial(Number(button.dataset.index))));
}

async function selectCampaign(id) {
  state.campaign = await api(`/v1/lab/campaigns/${id}`);
  $("#empty").hidden = true;
  $("#campaign-view").hidden = false;
  $("#campaign-title").textContent = state.campaign.name;
  $("#campaign-status").textContent = state.campaign.status;
  $("#campaign-meta").textContent = `${state.campaign.completed_trials}/${state.campaign.total_trials} terminés · ${state.campaign.accepted_trials} strictement acceptés`;
  $("#progress").style.width = `${state.campaign.total_trials ? 100 * state.campaign.completed_trials / state.campaign.total_trials : 0}%`;
  $("#export").href = `/v1/lab/campaigns/${id}/results.csv`;
  $("#cancel").disabled = !["queued", "running"].includes(state.campaign.status);
  renderScores();
  renderTrials();
  await loadCampaigns();
  clearInterval(state.poller);
  if (["queued", "running"].includes(state.campaign.status)) {
    state.poller = setInterval(() => selectCampaign(id).catch(console.error), 2500);
  }
}

function metric(label, value) {
  return `<article><span>${label}</span><b>${value}</b></article>`;
}

async function renderSrmpgdTrace(run) {
  const panel = $("#srmpgd-trace-panel");
  const host = $("#srmpgd-trace");
  panel.hidden = true;
  host.innerHTML = "";
  if (run.selected_variant !== "srmpgd") return;
  try {
    const trace = await api(`/v1/generations/${run.id}/metadata/srmpgd_trace`);
    panel.hidden = false;
    host.innerHTML = `<p class="muted">Loss robuste : ${trace.robust_loss_enabled ? "active" : "témoin officiel"} · état retenu : ${trace.selected_iteration} · arrêt : ${trace.stop_reason}</p>
      <div class="table-scroll"><table>
        <thead><tr><th>It.</th><th>Retenu</th><th>Score QR-Verify</th><th>MER</th><th>SRL</th><th>Base</th><th>Flou</th><th>Réduction</th><th>Lum.</th><th>Contraste</th><th>LPIPS</th><th>Δ latent</th><th>Pas</th><th>Garde</th><th>Gain QR</th><th>Éligible</th></tr></thead>
        <tbody>${trace.steps.map(step => `<tr class="${step.iteration === trace.selected_iteration ? "selected" : ""}">
          <td>${step.iteration}</td><td>${step.iteration === trace.selected_iteration ? "Oui" : ""}</td>
          <td>${pct(step.pass_rate)}</td><td>${pct(step.actual_module_error_rate)}</td>
          <td>${fmt(step.scanning_robust_loss, 6)}</td><td>${fmt(step.base_scanning_loss, 6)}</td>
          <td>${fmt(step.blur_scanning_loss, 6)}</td><td>${fmt(step.downscale_scanning_loss, 6)}</td>
          <td>${fmt(step.brightness_scanning_loss, 6)}</td><td>${fmt(step.contrast_scanning_loss, 6)}</td>
          <td>${fmt(step.lpips_loss, 5)}</td><td>${fmt(step.latent_delta_rms, 5)}</td>
          <td>${fmt(step.applied_step_rms, 5)}</td>
          <td>${step.aesthetic_guard_passed ? "OK" : "NON"}</td>
          <td>${step.qr_gain_sufficient ? "Oui" : "Non"}</td>
          <td>${step.eligible_for_selection ? "Oui" : "Non"}</td>
        </tr>`).join("")}</tbody>
      </table></div>`;
  } catch (error) {
    if (!String(error.message).startsWith("404")) {
      panel.hidden = false;
      host.textContent = `Trace indisponible : ${error.message}`;
    }
  }
}

async function openTrial(index) {
  state.selectedIndex = index;
  const trial = state.visibleTrials[index];
  const run = trial.generation;
  const quality = run.quality_metrics || {};
  const qrVerified = Number(quality.qr_verify_available || 0) === 1;
  const qartContract = Number(quality.diffqrcoder_stage2_control_target_qart || 0) === 1;
  $("#dialog-kicker").textContent = `${trial.status} · seed ${trial.seed}`;
  $("#dialog-title").textContent = `${trial.prompt_id} / ${trial.method_id}`;
  $("#dialog-metrics").innerHTML = [
    metric("Sortie", run.selected_variant),
    metric("QR-Verify — au moins un exact", qrVerified ? (run.exact_payload_match ? "OUI" : "NON") : "N/A historique"),
    metric("Score QR-Verify", qrVerified ? pct(run.scan_pass_rate) : "N/A historique"),
    metric("Presets exacts", qrVerified ? `${fmt(quality.qr_verify_exact_presets, 0)}/${fmt(quality.qr_verify_supported_presets, 0)}` : "N/A"),
    metric("Lecture sans filtre", qrVerified ? (Number(quality.qr_verify_direct_exact || 0) === 1 ? "Payload exact" : "Échec") : "N/A"),
    metric("Moteur", qrVerified ? "antfu/qr-verify 0.2.0 · WeChat WASM" : "Résultat historique — relancer E024"),
    metric("Nature de la mesure", "Test logiciel — pas une probabilité téléphone"),
    metric("CLIP-Aesthetic LAION", quality.clip_aesthetic == null ? "N/A" : fmt(quality.clip_aesthetic, 4)),
    metric("CLIP similarité (échelle papier)", quality.clip_similarity == null ? "N/A" : fmt(quality.clip_similarity, 5)),
    metric("CLIPScore (×2,5)", quality.clip_score == null ? "N/A" : fmt(quality.clip_score, 5)),
    metric("HPS v2.1", quality.hpsv2_1 == null ? "N/A" : fmt(quality.hpsv2_1, 5)),
    metric("Effet des scores esthétiques", "Aucun sur le verdict QR-Verify"),
    metric("Contrat payload", qartContract ? "URL canonique sans fragment" : "Byte-à-byte exact"),
    metric("MER", pct(run.module_error_rate)),
    metric("Erreur centres", pct(quality.structure_module_center_error_rate)),
    metric("Erreur centres fonctionnels", pct(quality.structure_functional_center_error_rate)),
    metric("Centres ambigus", pct(quality.structure_center_ambiguous_ratio)),
    metric("Confiance centres P10", fmt(quality.structure_center_confidence_p10, 3)),
    metric("Texture intra-module P95", fmt(quality.structure_intra_module_std_p95, 3)),
    metric("Quiet zone sombre", pct(quality.structure_quiet_zone_dark_pixel_ratio)),
    metric("Init. papier", Number(quality.diffqrcoder_stage2_paper_initialization || 0) === 1 ? "Oui" : "Non"),
    metric("Pas Stage 2 effectifs", fmt(quality.diffqrcoder_stage2_effective_steps, 0)),
    metric("Stage 2 réutilisé", Number(quality.diffqrcoder_stage2_reused || 0) === 1 ? "Oui — aucun recalcul" : "Non"),
    metric("Appariement Stage 2", run.provenance?.stage2_pairing_status === "exact_reuse" ? "Exact — SHA identique" : (run.provenance?.stage2_pairing_status === "generated_source" ? "Source SRPG" : "—")),
    metric("Source Stage 2", run.provenance?.stage2_source_method_id || "—"),
    metric("Cible Stage 2", Number(quality.diffqrcoder_stage2_control_target_qart || 0) === 1 ? "QArt réel — URL canonique" : "QR binaire exact"),
    metric("Contrat payload", Number(quality.diffqrcoder_stage2_control_target_qart || 0) === 1 ? "URL identique avant #" : "Byte-à-byte exact"),
    metric("Erreur centres cible", pct(quality.diffqrcoder_stage2_control_target_center_error_rate)),
    metric("QArt cible — ancien diagnostic", pct(quality.diffqrcoder_qart_target_scan_pass_rate)),
    metric("QArt seuil", fmt(quality.diffqrcoder_qart_threshold, 0)),
    metric("Changement Stage 1", pct(quality.diffqrcoder_stage2_changed_pixel_ratio)),
    metric("Saturation moyenne Δ", pct(quality.diffqrcoder_stage2_saturation_mean_increase)),
    metric("Pixels très saturés Δ", pct(quality.diffqrcoder_stage2_high_saturation_ratio_increase)),
    metric("Canaux RGB écrêtés", pct(quality.diffqrcoder_stage2_rgb_clipped_channel_ratio)),
    metric("Canaux RGB écrêtés Δ", pct(quality.diffqrcoder_stage2_rgb_clipped_channel_ratio_increase)),
    metric("Divergence", Number(quality.diffqrcoder_guard_diverged || 0) === 1 ? "OUI — résultat à écarter" : "Non"),
    metric("Alerte couleur", Number(quality.diffqrcoder_guard_warning || 0) === 1 ? "Oui — inspection humaine" : "Non"),
    metric("Sélection automatique", Number(quality.selection_auto_mode || 0) === 1 ? "Oui" : "Non"),
    metric("Repli Stage 1", Number(quality.selection_preserved_stage1 || 0) === 1 ? "Oui — Stage 2 écarté" : "Non"),
    metric("SR-MPGD gamma", fmt(quality.diffqrcoder_srmpgd_gamma, 3)),
    metric("Loss SR-MPGD", Number(quality.diffqrcoder_srmpgd_robust_loss_enabled || 0) === 1 ? "Robuste E020" : "Officielle"),
    metric("SR-MPGD LPIPS lambda", fmt(quality.diffqrcoder_srmpgd_lpips_weight, 3)),
    metric("Itération retenue", fmt(quality.diffqrcoder_srmpgd_selected_iteration, 0)),
    metric("Pas latent RMS max", fmt(quality.diffqrcoder_srmpgd_max_applied_step_rms, 4)),
    metric("Déplacement latent retenu", fmt(quality.diffqrcoder_srmpgd_selected_latent_delta_rms, 4)),
    metric("LPIPS retenu", fmt(quality.diffqrcoder_srmpgd_selected_lpips, 4)),
    metric("Changement retenu", pct(quality.diffqrcoder_srmpgd_selected_mean_absolute_change)),
    metric("Garde esthétique", Number(quality.diffqrcoder_srmpgd_selected_aesthetic_guard || 0) === 1 ? "Respectée" : "Échec / non exécutée"),
    metric("Gain QR suffisant", Number(quality.diffqrcoder_srmpgd_selected_qr_gain_sufficient || 0) === 1 ? "Oui" : "Non"),
    metric("Meilleur essai — itération", fmt(quality.diffqrcoder_srmpgd_attempted_best_iteration, 0)),
    metric("Meilleur essai — score QR", pct(quality.diffqrcoder_srmpgd_attempted_best_pass_rate)),
    metric("Meilleur essai — MER", pct(quality.diffqrcoder_srmpgd_attempted_best_mer)),
    metric("Meilleur essai — SRL", fmt(quality.diffqrcoder_srmpgd_attempted_best_srl, 6)),
    metric("Meilleur essai — éligible", Number(quality.diffqrcoder_srmpgd_attempted_best_eligible || 0) === 1 ? "Oui" : "Non"),
    metric("Arrêt SR-MPGD", run.provenance?.srmpgd_stop_reason || "—"),
    metric("Génération", `${fmt((run.generation_ms || 0) / 1000, 1)} s`),
    metric("Validation", `${fmt((run.validation_ms || 0) / 1000, 1)} s`),
    metric("SHA image finale", run.provenance?.final_image_sha256?.slice(0, 16) || "—"),
    metric("SHA Stage 1", run.provenance?.stage1_image_sha256?.slice(0, 16) || "—"),
    metric("SHA latent Stage 2", run.provenance?.stage2_latent_sha256?.slice(0, 16) || "—"),
    metric("SHA latent source", run.provenance?.stage2_source_latent_sha256?.slice(0, 16) || "—"),
  ].join("");
  const artifacts = await api(`/v1/generations/${run.id}/artifacts`);
  $("#artifacts").innerHTML = artifacts.map(item => `
    <figure><img src="${item.url}" loading="lazy"><figcaption>${item.name}</figcaption></figure>`).join("");
  await renderSrmpgdTrace(run);
  $("#validations").innerHTML = `<table>
    <thead><tr><th>Moteur</th><th>Test</th><th>Preset</th><th>${qartContract ? "URL canonique" : "Payload exact"}</th><th>Latence</th></tr></thead>
    <tbody>${(run.validations || []).map(item => `<tr>
      <td>${item.decoder}</td><td>${item.scenario}</td><td>${item.parameters?.preset || "—"}</td>
      <td class="${item.exact_payload_match ? "pass" : "fail"}">${item.exact_payload_match ? "Oui" : "Non"}</td>
      <td>${fmt(item.latency_ms, 1)} ms</td>
    </tr>`).join("")}</tbody>
  </table>`;
  const rating = trial.rating || {};
  $$("input[name='aesthetic-ok']").forEach(input => input.checked = String(rating.aesthetic_ok) === input.value);
  $$("input[name='human-scan']").forEach(input => input.checked = (rating.human_scan_result || "not_tested") === input.value);
  $("#aesthetic-score").value = rating.aesthetic_score ?? "";
  $("#human-scan-attempts").value = rating.human_scan_attempts ?? 0;
  $("#human-scan-successes").value = rating.human_scan_successes ?? 0;
  $("#human-scan-device").value = rating.human_scan_device ?? "";
  $("#rating-notes").value = rating.notes ?? "";
  $("#dialog-message").textContent = run.error || "";
  $("#dialog-message").classList.toggle("error", Boolean(run.error));
  $("#trial-dialog").showModal();
}

async function saveAndMove(direction = 1) {
  const trial = state.visibleTrials[state.selectedIndex];
  const aesthetic = $("input[name='aesthetic-ok']:checked")?.value;
  const scan = $("input[name='human-scan']:checked")?.value || "not_tested";
  await api(`/v1/lab/trials/${trial.id}/rating`, {
    method: "PUT",
    body: JSON.stringify({
      aesthetic_score: $("#aesthetic-score").value ? Number($("#aesthetic-score").value) : null,
      aesthetic_ok: aesthetic == null ? null : aesthetic === "true",
      human_scan_result: scan,
      human_scan_attempts: Number($("#human-scan-attempts").value || 0),
      human_scan_successes: Number($("#human-scan-successes").value || 0),
      human_scan_device: $("#human-scan-device").value,
      prompt_fidelity_score: null,
      qr_discretion_score: null,
      overall_score: null,
      favorite: false,
      notes: $("#rating-notes").value,
    }),
  });
  $("#dialog-message").textContent = "Verdict enregistré.";
  await selectCampaign(state.campaign.id);
  const next = Math.max(0, Math.min(state.visibleTrials.length - 1, state.selectedIndex + direction));
  if (next !== state.selectedIndex || direction > 0) await openTrial(next);
}

async function init() {
  try {
    const health = await api("/healthz");
    $("#runtime-status").textContent = `API ${health.version} · prête`;
    $("#runtime-status").classList.add("ok");
    state.schema = await api("/v1/lab/schema");
    state.methods = state.schema.profiles.map(normalizeMethod);
    renderMethods();
    await loadCampaigns();
  } catch (error) {
    $("#runtime-status").textContent = `API indisponible : ${error.message}`;
  }
}

$$(".tab").forEach(tab => tab.addEventListener("click", () => showPanel(tab.dataset.panel)));
$("#launch").addEventListener("click", launch);
$("#refresh").addEventListener("click", loadCampaigns);
$("#cancel").addEventListener("click", async () => {
  await api(`/v1/lab/campaigns/${state.campaign.id}/cancel`, {method: "POST"});
  await selectCampaign(state.campaign.id);
});
$("#filter").addEventListener("change", renderTrials);
$("#sort").addEventListener("change", renderTrials);
$("#prompts").addEventListener("input", updateCount);
$("#seeds").addEventListener("input", updateCount);
$("#add-method").addEventListener("click", () => {
  const source = state.methods.find(m => m.backend === "controlnet") || state.methods[0];
  const clone = copy(source);
  clone.id = `${slug(clone.id)}_${Date.now().toString().slice(-5)}`;
  clone.name = `${clone.name} — variante`;
  clone.enabled = true;
  state.methods.push(clone);
  renderMethods();
});
$("#close-dialog").addEventListener("click", () => $("#trial-dialog").close());
$("#save-next").addEventListener("click", () => saveAndMove(1).catch(error => $("#dialog-message").textContent = error.message));
$("#previous-trial").addEventListener("click", () => {
  const previous = Math.max(0, state.selectedIndex - 1);
  if (previous !== state.selectedIndex) openTrial(previous);
});

init();
