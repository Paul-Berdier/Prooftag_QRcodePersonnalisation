import fs from "node:fs/promises";
import { Workbook } from "@oai/artifact-tool";

const csvText = await fs.readFile(process.argv[2], "utf8");
const workbook = await Workbook.fromCSV(csvText, { sheetName: "E022" });
const values = workbook.worksheets.getItem("E022").getUsedRange(true).values;
const headers = values[0].map((value) => String(value ?? ""));
const rows = values.slice(1).map((valuesRow) =>
  Object.fromEntries(headers.map((header, index) => [header, valuesRow[index]])),
);

const number = (value) => value === "" || value == null ? null : Number(value);
const bool = (value) => String(value).toLowerCase() === "true";
const mean = (items) => {
  const present = items.map(number).filter((value) => value != null && Number.isFinite(value));
  return present.length ? present.reduce((sum, value) => sum + value, 0) / present.length : null;
};
const pearson = (pairs) => {
  const valid = pairs.filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  const mx = mean(valid.map(([x]) => x));
  const my = mean(valid.map(([, y]) => y));
  const numerator = valid.reduce((sum, [x, y]) => sum + (x - mx) * (y - my), 0);
  const denominator = Math.sqrt(
    valid.reduce((sum, [x]) => sum + (x - mx) ** 2, 0)
    * valid.reduce((sum, [, y]) => sum + (y - my) ** 2, 0),
  );
  return denominator ? numerator / denominator : null;
};
const family = (row) => row.prompt_id.includes("_simple_") ? "simple" : "atypical";
const methodLabel = (id) => id === "diffqrcoder_srpg" ? "safe" : "paper";

function summarize(subset) {
  const attempts = subset.reduce((sum, row) => sum + (number(row.human_scan_attempts) ?? 0), 0);
  const successes = subset.reduce((sum, row) => sum + (number(row.human_scan_successes) ?? 0), 0);
  return {
    rows: subset.length,
    accepted: subset.filter((row) => row.status === "accepted").length,
    phone_scannable_images: subset.filter((row) => row.human_scan_result === "scannable").length,
    phone_successes: successes,
    phone_attempts: attempts,
    phone_success_rate: attempts ? successes / attempts : null,
    aesthetic_ok: subset.filter((row) => bool(row.aesthetic_ok)).length,
    aesthetic_mean: mean(subset.map((row) => row.aesthetic_score)),
    ssr_mean: mean(subset.map((row) => row.scan_pass_rate)),
    phone_proxy_mean: mean(subset.map((row) => row.quality_phone_proxy_normalized_pass_rate)),
    original_decoder_passed_mean: mean(subset.map((row) => row.quality_validation_original_passed)),
    original_decoder_total_mean: mean(subset.map((row) => row.quality_validation_original_total)),
    mer_mean: mean(subset.map((row) => row.module_error_rate)),
    clip_aesthetic_mean: mean(subset.map((row) => row.quality_clip_aesthetic)),
    clip_score_mean: mean(subset.map((row) => row.quality_clip_score)),
    generation_seconds_mean: mean(subset.map((row) => number(row.generation_ms) / 1000)),
    guard_warnings: subset.filter((row) => number(row.quality_diffqrcoder_guard_warning) === 1).length,
    guard_divergences: subset.filter((row) => number(row.quality_diffqrcoder_guard_diverged) === 1).length,
    changed_pixel_ratio_mean: mean(subset.map((row) => row.quality_diffqrcoder_stage2_changed_pixel_ratio)),
    mean_absolute_change_mean: mean(subset.map((row) => row.quality_diffqrcoder_stage2_mean_absolute_change)),
    qart_target_original_pass_rate_mean: mean(subset.map((row) => row.quality_diffqrcoder_qart_target_original_pass_rate)),
  };
}

const summary = {};
for (const method of ["safe", "paper"]) {
  summary[method] = {};
  for (const group of ["all", "simple", "atypical"]) {
    summary[method][group] = summarize(rows.filter((row) =>
      methodLabel(row.method_id) === method && (group === "all" || family(row) === group),
    ));
  }
}

const pairs = [];
for (const promptId of [...new Set(rows.map((row) => row.prompt_id))].sort()) {
  const safe = rows.find((row) => row.prompt_id === promptId && methodLabel(row.method_id) === "safe");
  const paper = rows.find((row) => row.prompt_id === promptId && methodLabel(row.method_id) === "paper");
  pairs.push({
    prompt_id: promptId,
    family: family(safe),
    stage1_pair_ok: paper.stage1_source_run_id === safe.generation_run_id
      && paper.provenance_stage1_image_sha256 === safe.provenance_stage1_image_sha256,
    safe_phone: `${safe.human_scan_successes}/${safe.human_scan_attempts}`,
    paper_phone: `${paper.human_scan_successes}/${paper.human_scan_attempts}`,
    safe_ssr: number(safe.scan_pass_rate),
    paper_ssr: number(paper.scan_pass_rate),
    safe_aesthetic: number(safe.aesthetic_score),
    paper_aesthetic: number(paper.aesthetic_score),
    safe_original: `${safe.quality_validation_original_passed}/${safe.quality_validation_original_total}`,
    paper_original: `${paper.quality_validation_original_passed}/${paper.quality_validation_original_total}`,
    safe_mer: number(safe.module_error_rate),
    paper_mer: number(paper.module_error_rate),
    paper_warning: number(paper.quality_diffqrcoder_guard_warning),
  });
}

const correlations = {};
for (const method of ["safe", "paper", "all"]) {
  const subset = rows.filter((row) => method === "all" || methodLabel(row.method_id) === method);
  const phonePairs = subset.map((row) => {
    const attempts = number(row.human_scan_attempts);
    return [attempts ? number(row.human_scan_successes) / attempts : NaN, row];
  });
  correlations[method] = {
    human_vs_ssr: pearson(phonePairs.map(([human, row]) => [human, number(row.scan_pass_rate)])),
    human_vs_phone_proxy: pearson(phonePairs.map(([human, row]) => [human, number(row.quality_phone_proxy_normalized_pass_rate)])),
  };
}

console.log(JSON.stringify({ summary, correlations, pairs }, null, 2));
