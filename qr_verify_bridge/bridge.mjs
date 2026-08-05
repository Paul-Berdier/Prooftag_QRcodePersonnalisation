import readline from 'node:readline'
import process from 'node:process'

import { scan } from 'qr-scanner-wechat'
import sharp from 'sharp'

// qr-verify 0.2.0 uses these exact values in its "high" tolerance mode.
// Each preset starts from a fresh Sharp pipeline: unlike the interactive CLI,
// the benchmark is deterministic and does not shuffle or accumulate filters.
const presets = [
  { id: 'original', contrast: 1, brightness: 1, blur: 0 },
]

for (const contrast of [6, 3, 1.5]) {
  for (const blur of [0.5, 1, 1.5, 2]) {
    for (const brightness of [0.9, 1.2, 1.4]) {
      presets.push({
        id: `c${contrast}_b${brightness}_r${blur}`,
        contrast,
        brightness,
        blur,
      })
    }
  }
}

function preprocess(buffer, preset) {
  let image = sharp(buffer)
    .resize({ withoutEnlargement: true, width: 300, height: 300, fit: 'outside' })
    .ensureAlpha()
    .grayscale()
  if (preset.contrast !== 1)
    image = image.linear(preset.contrast, -(128 * preset.contrast) + 128)
  if (preset.brightness !== 1)
    image = image.modulate({ brightness: preset.brightness })
  if (preset.blur)
    image = image.blur(preset.blur)
  return image
}

async function verify(imageBase64) {
  const source = Buffer.from(imageBase64, 'base64')
  const attempts = []
  for (const preset of presets) {
    const started = performance.now()
    let text = ''
    let error = null
    try {
      const { data, info } = await preprocess(source, preset).raw().toBuffer({
        resolveWithObject: true,
      })
      const result = await scan({
        width: info.width,
        height: info.height,
        data: Uint8ClampedArray.from(data),
      })
      text = result?.text || ''
    }
    catch (caught) {
      error = `${caught?.name || 'Error'}: ${caught?.message || String(caught)}`
    }
    attempts.push({
      preset: preset.id,
      contrast: preset.contrast,
      brightness: preset.brightness,
      blur: preset.blur,
      text,
      latency_ms: performance.now() - started,
      error,
    })
  }
  return { engine: 'qr-verify@0.2.0', preset_count: presets.length, attempts }
}

const lines = readline.createInterface({ input: process.stdin, crlfDelay: Infinity })
for await (const line of lines) {
  if (!line.trim())
    continue
  let response
  try {
    const request = JSON.parse(line)
    response = { id: request.id, ok: true, ...(await verify(request.image_base64)) }
  }
  catch (error) {
    response = {
      id: null,
      ok: false,
      error: `${error?.name || 'Error'}: ${error?.message || String(error)}`,
    }
  }
  process.stdout.write(`${JSON.stringify(response)}\n`)
}
