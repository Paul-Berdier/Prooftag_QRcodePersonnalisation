import fs from "node:fs/promises";
import { Workbook } from "@oai/artifact-tool";

const csvPath = process.argv[2];
const csvText = await fs.readFile(csvPath, "utf8");
const workbook = await Workbook.fromCSV(csvText, { sheetName: "E022" });

const overview = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 3000,
  tableMaxRows: 4,
  tableMaxCols: 12,
  tableMaxCellChars: 80,
});
console.log("OVERVIEW");
console.log(overview.ndjson);

const sheet = workbook.worksheets.getItem("E022");
const used = sheet.getUsedRange(true);
const values = used.values;
const headers = values[0].map((value) => String(value ?? ""));
console.log("DIMENSIONS", JSON.stringify({ rows: values.length - 1, columns: headers.length }));
console.log("HEADERS", JSON.stringify(headers));

const wanted = [
  "prompt_id", "method_id", "seed", "status", "selected_variant",
  "stage1_reused", "stage1_source_run_id", "scan_pass_rate",
  "exact_payload_match", "module_error_rate", "generation_ms",
  "aesthetic_score", "aesthetic_ok", "human_scan_result",
  "human_scan_attempts", "human_scan_successes", "human_scan_device",
  "prompt_fidelity_score", "qr_discretion_score", "overall_score",
  "quality_clip_aesthetic", "quality_clip_score", "quality_clip_similarity",
  "quality_diffqrcoder_stage2_effective_steps",
  "quality_diffqrcoder_stage2_control_target_exact",
  "quality_diffqrcoder_stage2_control_target_qart",
  "quality_diffqrcoder_guard_diverged",
  "quality_diffqrcoder_guard_warning",
  "provenance_stage1_source_run_id", "provenance_final_image_sha256",
];
const indices = wanted.map((name) => [name, headers.indexOf(name)]).filter(([, index]) => index >= 0);
const rows = values.slice(1).map((row) => Object.fromEntries(indices.map(([name, index]) => [name, row[index]])));
console.log("ROWS");
console.log(JSON.stringify(rows));
