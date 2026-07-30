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
  return method;
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
    $(".output", node).value = method.output_variant;
    $(".steps", node).value = method.generation.steps ?? 40;
    $(".cfg", node).value = method.generation.guidance_scale ?? 7.5;
    $(".control", node).value = method.generation.controlnet_scale ?? 1.35;
    $(".stage2-init", node).value = settings.diffqrcoder_stage2_initialization ?? "paper_stage1_noise";
    $(".stage2-target", node).value = settings.diffqrcoder_stage2_target_mode ?? "binary_exact";
    $(".stage2-strength", node).value = settings.diffqrcoder_stage2_strength ?? 1;
    $(".srpg-steps", node).value = settings.srpg_steps ?? 40;
    $(".srpg-control", node).value = settings.srpg_controlnet_scale ?? 1.35;
    $(".srg", node).value = settings.srpg_qr_weight ?? 500;
    $(".pg", node).value = settings.srpg_perceptual_weight ?? 2;
    $(".eta", node).value = settings.srpg_eta ?? 0;
    $(".seed-offset", node).value = settings.srpg_seed_offset ?? 2000003;
    $(".preview", node).value = settings.srpg_preview_interval ?? 5;
    $(".mpgd-iterations", node).value = settings.srmpgd_max_iterations ?? 20;
    $(".mpgd-lr", node).value = settings.srmpgd_step_size ?? 1000;
    $(".mpgd-lpips", node).value = settings.srmpgd_lpips_weight ?? 0.01;
    $(".mpgd-max-mer", node).value = settings.srmpgd_max_initial_module_error_rate ?? 0.35;
    $(".model-json", node).value = JSON.stringify(method.model, null, 2);
    node.classList.toggle("disabled", !method.enabled);
    node.classList.toggle("has-stage2", method.output_variant !== "raw");
    node.classList.toggle("has-srmpgd", method.output_variant === "srmpgd");

    const sync = () => {
      const item = state.methods[index];
      item.enabled = $(".enabled", node).checked;
      item.name = $(".name", node).value.trim();
      item.id = slug($(".id", node).value);
      item.output_variant = $(".output", node).value;
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
        diffqrcoder_guard_max_rgb_clipped_channel_ratio_increase: 0.02,
        diffqrcoder_guard_max_saturation_mean_increase: 0.08,
        diffqrcoder_guard_max_high_saturation_ratio_increase: 0.05,
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

function autoOriginal(trial) {
  return (trial.generation?.validations || []).filter(v => v.scenario === "original");
}

function aggregate(trials) {
  const allGenerated = trials.filter(t => t.generation);
  const generated = allGenerated.filter(t => t.method_id !== "qr_reference");
  const rated = generated.filter(t => t.rating && (
    t.rating.aesthetic_ok != null || t.rating.human_scan_result !== "not_tested"
  ));
  const strict = generated.filter(t => t.generation.exact_payload_match && t.generation.scan_pass_rate === 1);
  const humanScan = generated.filter(t => t.rating?.human_scan_result === "scannable");
  const aesthetic = generated.filter(t => t.rating?.aesthetic_ok === true);
  const mean = (items, getter) => items.length ? items.reduce((sum, item) => sum + Number(getter(item) || 0), 0) / items.length : null;
  return {
    generated: generated.length,
    rated: rated.length,
    strict: strict.length,
    humanScan: humanScan.length,
    aesthetic: aesthetic.length,
    meanSsr: mean(generated, t => t.generation.scan_pass_rate),
    meanAes: mean(generated.filter(t => t.generation.quality_metrics?.clip_aesthetic != null), t => t.generation.quality_metrics.clip_aesthetic),
    meanClip: mean(generated.filter(t => t.generation.quality_metrics?.clip_score != null), t => t.generation.quality_metrics.clip_score),
  };
}

function renderScores() {
  const a = aggregate(state.campaign.trials);
  $("#final-scores").innerHTML = [
    ["Auto strict", `${a.strict}/${a.generated}`],
    ["SSR robuste moyen", pct(a.meanSsr)],
    ["CLIP-aesthetic moyen", fmt(a.meanAes, 2)],
    ["CLIPScore moyen", fmt(a.meanClip, 2)],
    ["Tes scans positifs", `${a.humanScan}/${a.rated}`],
    ["Tes esthétiques positives", `${a.aesthetic}/${a.rated}`],
    ["Évalués", `${a.rated}/${a.generated}`],
  ].map(([label, value]) => `<article><span>${label}</span><b>${value}</b></article>`).join("");
}

function orderedTrials() {
  let trials = state.campaign.trials.filter(t => t.generation);
  const filter = $("#filter").value;
  if (filter === "unrated") trials = trials.filter(t => !t.rating || (t.rating.aesthetic_ok == null && t.rating.human_scan_result === "not_tested"));
  if (filter === "scannable" || filter === "not_scannable") trials = trials.filter(t => t.rating?.human_scan_result === filter);
  if (filter === "auto_pass") trials = trials.filter(t => t.generation.exact_payload_match && t.generation.scan_pass_rate === 1);
  const sort = $("#sort").value;
  if (sort === "ssr") trials.sort((a, b) => (b.generation.scan_pass_rate ?? -1) - (a.generation.scan_pass_rate ?? -1));
  if (sort === "aesthetic") trials.sort((a, b) => (b.generation.quality_metrics?.clip_aesthetic ?? -1) - (a.generation.quality_metrics?.clip_aesthetic ?? -1));
  if (sort === "time") trials.sort((a, b) => (a.generation.total_ms ?? Infinity) - (b.generation.total_ms ?? Infinity));
  return trials;
}

function renderTrials() {
  state.visibleTrials = orderedTrials();
  $("#trials").innerHTML = state.visibleTrials.map((trial, index) => {
    const run = trial.generation;
    const rating = trial.rating;
    const original = autoOriginal(trial);
    const originalPass = original.filter(v => v.exact_payload_match).length;
    const qartContract = Number(run.quality_metrics?.diffqrcoder_stage2_control_target_qart || 0) === 1;
    const diverged = Number(run.quality_metrics?.diffqrcoder_guard_diverged || 0) === 1;
    return `<button class="trial" data-index="${index}">
      <div class="image">${run.image_url
        ? `<img src="${run.image_url}" loading="lazy" alt="${trial.prompt_id}">`
        : "<span class='missing-image'>Génération en erreur</span>"}</div>
      <div class="trial-body">
        <div class="trial-title"><b>${trial.prompt_id} / ${trial.method_id}</b><span class="status ${trial.status}">${trial.status}</span></div>
        <p class="${run.error ? "run-error" : ""}">${run.error || `seed ${trial.seed} · ${run.selected_variant}`}</p>
        <div class="trial-stats">
          <span>SSR<b>${pct(run.scan_pass_rate)}</b></span>
          <span>${qartContract ? "URL canon." : "Original"}<b>${originalPass}/${original.length}</b></span>
          <span>MER<b>${pct(run.module_error_rate)}</b></span>
          <span>CLIP-AES<b>${fmt(run.quality_metrics?.clip_aesthetic, 2)}</b></span>
        </div>
        <div class="badges">
          ${diverged ? "<i class='bad'>Divergence Stage 2</i>" : ""}
          ${rating?.aesthetic_ok === true ? "<i class='good'>Esthétique ✓</i>" : rating?.aesthetic_ok === false ? "<i class='bad'>Esthétique ✕</i>" : "<i>Esthétique ?</i>"}
          ${rating?.human_scan_result === "scannable" ? "<i class='good'>Scan ✓</i>" : rating?.human_scan_result === "not_scannable" ? "<i class='bad'>Scan ✕</i>" : "<i>Scan ?</i>"}
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

async function openTrial(index) {
  state.selectedIndex = index;
  const trial = state.visibleTrials[index];
  const run = trial.generation;
  const quality = run.quality_metrics || {};
  const qartContract = Number(quality.diffqrcoder_stage2_control_target_qart || 0) === 1;
  $("#dialog-kicker").textContent = `${trial.status} · seed ${trial.seed}`;
  $("#dialog-title").textContent = `${trial.prompt_id} / ${trial.method_id}`;
  const original = autoOriginal(trial);
  $("#dialog-metrics").innerHTML = [
    metric("Sortie", run.selected_variant),
    metric("SSR robuste", pct(run.scan_pass_rate)),
    metric(qartContract ? "URL canonique" : "Payload original", `${original.filter(v => v.exact_payload_match).length}/${original.length}`),
    metric("MER", pct(run.module_error_rate)),
    metric("CLIP-aesthetic", fmt(run.quality_metrics?.clip_aesthetic, 3)),
    metric("CLIPScore", fmt(run.quality_metrics?.clip_score, 3)),
    metric("Init. papier", Number(quality.diffqrcoder_stage2_paper_initialization || 0) === 1 ? "Oui" : "Non"),
    metric("Pas Stage 2 effectifs", fmt(quality.diffqrcoder_stage2_effective_steps, 0)),
    metric("Stage 2 réutilisé", Number(quality.diffqrcoder_stage2_reused || 0) === 1 ? "Oui — aucun recalcul" : "Non"),
    metric("Cible Stage 2", Number(quality.diffqrcoder_stage2_control_target_qart || 0) === 1 ? "QArt réel — URL canonique" : "QR binaire exact"),
    metric("Contrat payload", Number(quality.diffqrcoder_stage2_control_target_qart || 0) === 1 ? "URL identique avant #" : "Byte-à-byte exact"),
    metric("Erreur centres cible", pct(quality.diffqrcoder_stage2_control_target_center_error_rate)),
    metric("QArt cible SSR", pct(quality.diffqrcoder_qart_target_scan_pass_rate)),
    metric("QArt seuil", fmt(quality.diffqrcoder_qart_threshold, 0)),
    metric("Changement Stage 1", pct(quality.diffqrcoder_stage2_changed_pixel_ratio)),
    metric("Saturation moyenne Δ", pct(quality.diffqrcoder_stage2_saturation_mean_increase)),
    metric("Pixels très saturés Δ", pct(quality.diffqrcoder_stage2_high_saturation_ratio_increase)),
    metric("Canaux RGB écrêtés", pct(quality.diffqrcoder_stage2_rgb_clipped_channel_ratio)),
    metric("Canaux RGB écrêtés Δ", pct(quality.diffqrcoder_stage2_rgb_clipped_channel_ratio_increase)),
    metric("Divergence", Number(quality.diffqrcoder_guard_diverged || 0) === 1 ? "OUI — résultat à écarter" : "Non"),
    metric("SR-MPGD gamma", fmt(quality.diffqrcoder_srmpgd_gamma, 3)),
    metric("SR-MPGD LPIPS lambda", fmt(quality.diffqrcoder_srmpgd_lpips_weight, 3)),
    metric("Itération retenue", fmt(quality.diffqrcoder_srmpgd_selected_iteration, 0)),
    metric("Génération", `${fmt((run.generation_ms || 0) / 1000, 1)} s`),
    metric("Validation", `${fmt((run.validation_ms || 0) / 1000, 1)} s`),
  ].join("");
  const artifacts = await api(`/v1/generations/${run.id}/artifacts`);
  $("#artifacts").innerHTML = artifacts.map(item => `
    <figure><img src="${item.url}" loading="lazy"><figcaption>${item.name}</figcaption></figure>`).join("");
  $("#validations").innerHTML = `<table>
    <thead><tr><th>Décodeur</th><th>Scénario</th><th>${qartContract ? "URL canonique" : "Payload exact"}</th><th>Latence</th></tr></thead>
    <tbody>${(run.validations || []).map(item => `<tr>
      <td>${item.decoder}</td><td>${item.scenario}</td>
      <td class="${item.exact_payload_match ? "pass" : "fail"}">${item.exact_payload_match ? "Oui" : "Non"}</td>
      <td>${fmt(item.latency_ms, 1)} ms</td>
    </tr>`).join("")}</tbody>
  </table>`;
  const rating = trial.rating || {};
  $$("input[name='aesthetic-ok']").forEach(input => input.checked = String(rating.aesthetic_ok) === input.value);
  $$("input[name='human-scan']").forEach(input => input.checked = (rating.human_scan_result || "not_tested") === input.value);
  $("#aesthetic-score").value = rating.aesthetic_score ?? "";
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
