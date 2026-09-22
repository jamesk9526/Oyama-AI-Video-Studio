import type { ModelFile } from '../types'
import type { ComfyPrompt } from './workflow'

export type Music3ModelSelection = { diffusion: string; textEncoder: string; vae: string }
export type Music3GenerationOptions = { caption: string; lyrics: string; duration: number; seed: number; tiledDecode: boolean; filenamePrefix: string }
export const MUSIC3_REQUIRED_NODES = ['UNETLoader', 'CLIPLoader', 'VAELoader', 'MiniMaxMusic3TextEncode', 'EmptyMiniMaxMusic3LatentAudio', 'ConditioningZeroOut', 'KSampler', 'VAEDecodeAudioTiled', 'SaveAudioAdvanced'] as const

export function inferMusic3Selection(models: ModelFile[]): Music3ModelSelection {
  const find = (kind: ModelFile['kind'], pattern: RegExp) => models.find((model) => model.kind === kind && pattern.test(model.name))?.name ?? ''
  return { diffusion: find('diffusion_models', /(?:minimax[_-]?)?music[_-]?3.*(?:int8|convrot)|music3.*int8/i) || find('diffusion_models', /music[_-]?3/i), textEncoder: find('text_encoders', /music[_-]?3.*(?:text|encoder)/i), vae: find('vae', /music[_-]?3.*(?:dav|vae)/i) }
}

export function buildMusic3Workflow(options: Music3GenerationOptions, models: Music3ModelSelection): ComfyPrompt {
  const graph: ComfyPrompt = {
    '1': { class_type: 'UNETLoader', inputs: { unet_name: models.diffusion, weight_dtype: 'default' } },
    '2': { class_type: 'CLIPLoader', inputs: { clip_name: models.textEncoder, type: 'minimax', device: 'default' } },
    '3': { class_type: 'VAELoader', inputs: { vae_name: models.vae } },
    '4': { class_type: 'MiniMaxMusic3TextEncode', inputs: { clip: ['2', 0], caption: options.caption, lyrics: options.lyrics, seed: options.seed, max_duration: Math.max(4, Math.min(300, options.duration)), cfg_scale: 1.7, top_k: 50 } },
    '5': { class_type: 'ConditioningZeroOut', inputs: { conditioning: ['4', 0] } },
    '6': { class_type: 'EmptyMiniMaxMusic3LatentAudio', inputs: { seconds: ['4', 1], batch_size: 1 } },
    '7': { class_type: 'KSampler', inputs: { model: ['1', 0], positive: ['4', 0], negative: ['5', 0], latent_image: ['6', 0], seed: options.seed, steps: 30, cfg: 1.7, sampler_name: 'euler', scheduler: 'simple', denoise: 1 } },
    // Current ComfyUI exposes MP3 quality as a DynamicCombo sub-input (`format.quality`); `bitrate` is kept for older builds.
    '9': { class_type: 'SaveAudioAdvanced', inputs: { audio: ['8', 0], filename_prefix: options.filenamePrefix, format: 'mp3', 'format.quality': 'V0', bitrate: 'V0' } },
  }
  graph['8'] = options.tiledDecode ? { class_type: 'VAEDecodeAudioTiled', inputs: { samples: ['7', 0], vae: ['3', 0], tile_size: 1536, overlap: 64 } } : { class_type: 'VAEDecodeAudio', inputs: { samples: ['7', 0], vae: ['3', 0] } }
  return graph
}

export function buildMusic3Caption(sections: { metadata: string; vocals: string; arrangement: string }) {
  return [
    `### Global Metadata\n${sections.metadata.trim() || 'Unspecified; preserve the user’s explicit musical constraints.'}`,
    `### Vocal Details\n${sections.vocals.trim() || 'Vocal configuration unspecified; use a conservative treatment appropriate to the requested style.'}`,
    `### Arrangement\n${sections.arrangement.trim() || 'Develop a coherent section-by-section timeline around the tagged lyric structure.'}`,
  ].join('\n\n')
}
