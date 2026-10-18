import { invoke } from '@tauri-apps/api/core';
import type { HFModelDetail, AppSettings, SystemInfo } from '../stores/appStore';

interface CodeGenerationResponse {
  code: string;
  analysis: string;
  dependencies?: string[];
}

interface RawGenerationResponse extends Partial<CodeGenerationResponse> {
  error?: string;
  errorType?: string;
  traceback?: string;
  stderr?: string;
  debugStderr?: string;
  exitCode?: string;
}

interface AnthropicMessagesResponse {
  text?: string;
}

interface ClaudeFixContext {
  currentCode: string;
  stderrOutput: string;
  installedModules: string;
  executionError?: string | null;
}

const CLAUDE_SONNET_MODEL = 'claude-sonnet-4-6';

function previewText(raw: string, max = 2200): string {
  const trimmed = raw.trim();
  if (trimmed.length <= max) return trimmed;
  return `${trimmed.slice(0, max)}\n...[truncated ${trimmed.length - max} chars]`;
}

function parseRunnerResponse(raw: string): RawGenerationResponse {
  const candidates: string[] = [];
  const trimmed = raw.trim();
  if (trimmed) {
    candidates.push(trimmed);
    const lines = trimmed
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
    if (lines.length > 1) {
      candidates.push(lines[lines.length - 1]);
    }
    const firstBrace = trimmed.indexOf('{');
    const lastBrace = trimmed.lastIndexOf('}');
    if (firstBrace >= 0 && lastBrace > firstBrace) {
      candidates.push(trimmed.slice(firstBrace, lastBrace + 1));
    }
  }

  for (const candidate of candidates) {
    try {
      return JSON.parse(candidate) as RawGenerationResponse;
    } catch {
      // try next candidate
    }
  }
  throw new Error(
    `Runner output is not valid JSON. Raw output preview:\n${previewText(raw)}`
  );
}

function normalizeDependencyName(raw: string): string {
  return raw.trim().toLowerCase().replace(/^['"`]+|['"`]+$/g, '');
}

function dedupeDependencies(dependencies: string[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();

  for (const dep of dependencies) {
    const trimmed = dep.trim();
    const key = normalizeDependencyName(trimmed);
    if (!trimmed || !key || seen.has(key)) continue;
    seen.add(key);
    out.push(trimmed);
  }

  return out;
}

async function fetchModelReadme(modelId: string, hfToken?: string): Promise<string> {
  const headers: HeadersInit = {};
  if (hfToken) headers['Authorization'] = `Bearer ${hfToken}`;

  const response = await fetch(`https://huggingface.co/${modelId}/raw/main/README.md`, {
    headers,
  });
  if (!response.ok) {
    return '';
  }
  return response.text();
}

export function buildCacheIdentity(model: HFModelDetail, system: SystemInfo): string {
  // We use model ID, RAM size, GPU availability, and OS as factors that might affect script output.
  return `local-worker-v2_${model.modelId || model.id}_${system.totalRam}_gpu:${system.gpuName ? 'yes' : 'no'}_${system.os}`;
}

export async function generateCodeLocally(
  model: HFModelDetail,
  settings: AppSettings,
  _system: SystemInfo,
  ggufFile?: string | null
): Promise<CodeGenerationResponse> {
  const modelId = model.modelId || model.id;

  if (!modelId) {
    throw new Error('Model ID is required to generate code.');
  }

  // Execute the python auto runner to get JSON code analysis payload
  try {
    const rawResponse = await invoke<string>('generate_python_code_local', {
      modelId: modelId,
      ggufFile: ggufFile ?? null,
      hfToken: settings.hfToken?.trim() ? settings.hfToken.trim() : null,
    });

    let parsed: RawGenerationResponse;
    try {
      parsed = parseRunnerResponse(rawResponse);
    } catch (parseError) {
      throw new Error(
        `Failed parsing generated response JSON: ${String(parseError)}`
      );
    }

    if (parsed.debugStderr) {
      console.info('[hf_auto_runner][debugStderr]', parsed.debugStderr);
    }

    if (parsed.error) {
      const details = [
        `hf_auto_runner execution failed: ${parsed.error}`,
        parsed.errorType ? `Type: ${parsed.errorType}` : '',
        parsed.exitCode ? `Exit: ${parsed.exitCode}` : '',
        parsed.stderr ? `Stderr:\n${previewText(parsed.stderr)}` : '',
        parsed.traceback ? `Traceback:\n${previewText(parsed.traceback)}` : '',
      ]
        .filter(Boolean)
        .join('\n');
      throw new Error(details);
    }

    if (!parsed.code || !parsed.analysis) {
      throw new Error(
        `Invalid output received from hf_auto_runner python generator.\nParsed payload preview:\n${previewText(
          JSON.stringify(parsed, null, 2)
        )}`
      );
    }

    return {
      code: parsed.code,
      analysis: parsed.analysis
    };

  } catch (error) {
     console.error("Local code generation failed:", error);
     throw new Error(`Could not generate local script: ${error}`);
  }
}

export async function generateCodeWithAssistant(
  model: HFModelDetail,
  settings: AppSettings,
  system: SystemInfo
): Promise<CodeGenerationResponse> {
  const modelId = model.modelId || model.id;
  const claudeApiKey = settings.claudeApiKey?.trim();

  if (!modelId) {
    throw new Error('Model ID is required to generate code.');
  }
  if (!claudeApiKey && !['codex-cli', 'claude-code'].includes(settings.codeGenerationProvider)) {
    throw new Error('Anthropic API key is required for Claude Sonnet generation.');
  }
  try {
    const readme = (model.readme?.trim() ? model.readme : await fetchModelReadme(modelId, settings.hfToken?.trim() || undefined)).trim();
    const siblingNames = (model.siblings ?? []).map((item) => item.rfilename).slice(0, 200);
    const systemSummary = {
      totalRamGb: Number((system.totalRam / 1024 ** 3).toFixed(1)),
      availableRamGb: Number((system.availableRam / 1024 ** 3).toFixed(1)),
      gpuName: system.gpuName,
      gpuVramGb: system.gpuVram,
      os: system.os,
    };

    const prompt = [
      'Read the Hugging Face model metadata and README below and generate a single-file Python script for local inference.',
      '',
      'Return strict JSON only with this exact shape:',
      '{',
      '  "analysis": "short explanation of the runtime choice and constraints",',
      '  "code": "full python script as a string",',
      '  "dependencies": ["pip package spec", "..."]',
      '}',
      '',
      'Requirements for the Python script:',
      '- It must be a single file.',
      '- Use model_source(model_id) from hf_auto_runner.local_runtime for local weights and load_once(loader, ...) for reusable model objects.',
    '- All downloads are prepared separately; do not download weights in the generated script.',
      '- It must read user input from environment variable HB_INPUT.',
      '- HB_INPUT contains the user-provided input from the app and should be treated as the primary runtime input contract.',
      '- For text models, HB_INPUT is plain text.',
      '- For question-answering, HB_INPUT may contain a JSON payload encoded as a string.',
      '- For image models, HB_INPUT may be an absolute local file path, an HTTP/HTTPS URL, or a data URL. The app may also prefix image uploads with __HBIMG__:.',
      '- For audio models, HB_INPUT is usually an absolute local file path to the user-selected audio file.',
      '- For Sprint 8 multimodal code, treat HB_INPUT as text only. Use HB_IMAGE_PATH for uploaded images, HB_DOCUMENT_PATH for uploaded document images or PDFs, and HB_MULTIMODAL_TASK for the selected multimodal task.',
      '- The script must handle missing or empty HB_INPUT gracefully with a sensible fallback message or default behavior.',
        '- It must read the Hugging Face token from environment variable HF_TOKEN when available, but fall back to no token if HF_TOKEN is unset or empty.',
        '- If it writes an audio output file, it must print HB_OUTPUT_AUDIO:<absolute_path>.',
        '- For multimodal outputs, print HB_MULTIMODAL_TEXT:<text> or HB_MULTIMODAL_JSON:<json>, and print HB_REFERENCE_IMAGE:<absolute_path> for the actual image fed to the model.',
        '- It must be runnable directly with python script.py inside a model workspace.',
        '- Prefer repository-documented libraries over generic transformers if the README indicates a custom runtime.',
        '- Only include dependencies that are actually needed by the generated script.',
        '- Keep dependencies pip-installable. If the README mentions non-pip system packages, mention them in analysis instead of dependencies.',
        '- The script must be verbose about progress logging.',
        '- Add print statements before and after every major step: input parsing, dependency/runtime selection, model loading, processor/tokenizer loading, preprocessing, inference, post-processing, and file output.',
        '- For long-running operations, print a clear status message immediately before the call so the app does not appear stuck.',
        '- All progress logs must flush immediately. Use print(..., flush=True) or equivalent.',
        '- Preserve machine-readable output markers like HB_OUTPUT_AUDIO:<absolute_path>, but keep the rest of the script logs human-readable and explicit.',
        '- Never download or invent a sample image when multimodal input is missing. Fail clearly instead.',
        '',
        `Model ID: ${modelId}`,
      `Pipeline Tag: ${model.pipeline_tag ?? 'unknown'}`,
      `Author: ${model.author ?? 'unknown'}`,
      `Tags: ${(model.tags ?? []).join(', ')}`,
      `Description: ${model.description ?? ''}`,
      `Sibling Files: ${JSON.stringify(siblingNames)}`,
      `System Info: ${JSON.stringify(systemSummary)}`,
      '',
      'README:',
      readme || '(README not available)',
    ].join('\n');

    const payload = await requestAssistant(settings, {
      model: CLAUDE_SONNET_MODEL,
      apiKey: claudeApiKey,
      prompt,
    });
    const text = payload.text?.trim() ?? '';

    if (!text) {
      throw new Error('Assistant returned no text content.');
    }

    const parsed = parseRunnerResponse(text);
    if (!parsed.code || !parsed.analysis) {
      throw new Error(
        `Assistant returned an invalid payload.\nPayload preview:\n${previewText(text)}`
      );
    }

    return {
      code: parsed.code,
      analysis: parsed.analysis,
      dependencies: dedupeDependencies(
        Array.isArray(parsed.dependencies)
          ? parsed.dependencies.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
          : []
      ),
    };
  } catch (error) {
    throw error;
  }
}

export async function fixCodeWithAssistant(
  model: HFModelDetail,
  settings: AppSettings,
  system: SystemInfo,
  context: ClaudeFixContext
): Promise<CodeGenerationResponse> {
  const modelId = model.modelId || model.id;
  const claudeApiKey = settings.claudeApiKey?.trim();

  if (!modelId) {
    throw new Error('Model ID is required to fix code.');
  }
  if (!claudeApiKey && !['codex-cli', 'claude-code'].includes(settings.codeGenerationProvider)) {
    throw new Error('Anthropic API key is required for Claude Sonnet fix mode.');
  }

  const readme = (model.readme?.trim() ? model.readme : await fetchModelReadme(modelId, settings.hfToken?.trim() || undefined)).trim();
  const siblingNames = (model.siblings ?? []).map((item) => item.rfilename).slice(0, 200);
  const systemSummary = {
    totalRamGb: Number((system.totalRam / 1024 ** 3).toFixed(1)),
    availableRamGb: Number((system.availableRam / 1024 ** 3).toFixed(1)),
    gpuName: system.gpuName,
    gpuVramGb: system.gpuVram,
    os: system.os,
  };

  const prompt = [
    'Fix the current HuggingBox Python script using the runtime failure context below.',
    'Use the supplied model documentation to verify runtime, dependency requirements and input format. State uncertainty explicitly.',
    '',
    'Return strict JSON only with this exact shape:',
    '{',
    '  "analysis": "short explanation of what was wrong and what was changed",',
    '  "code": "full corrected python script as a string",',
    '  "dependencies": ["pip package spec", "..."]',
    '}',
    '',
    'Requirements for the corrected Python script:',
    '- It must be a single file.',
    '- Use model_source(model_id) from hf_auto_runner.local_runtime for local weights and load_once(loader, ...) for reusable model objects.',
    '- All downloads are prepared separately; do not download weights in the generated script.',
    '- Preserve HuggingBox environment contracts. Use HB_MODEL_DIR for local weights; never silently download another copy.',
    '- HB_INPUT is the primary text/runtime input contract unless the model is multimodal.',
    '- For Sprint 8 multimodal code, use HB_INPUT for text only, HB_IMAGE_PATH for uploaded images, HB_DOCUMENT_PATH for uploaded document images or PDFs, and HB_MULTIMODAL_TASK for the selected multimodal task.',
    '- It must read HF_TOKEN from the environment when available, but fall back to no token if missing.',
    '- It must be verbose about progress logging and flush logs immediately.',
    '- Preserve machine-readable markers like HB_OUTPUT_AUDIO, HB_MULTIMODAL_TEXT, HB_MULTIMODAL_JSON, and HB_REFERENCE_IMAGE when applicable.',
    '- Never invent or download sample inputs.',
    '- Only include dependencies actually needed by the corrected script.',
    '- If the fix requires non-pip system packages, explain them in analysis rather than dependencies.',
    '',
    `Model ID: ${modelId}`,
    `Pipeline Tag: ${model.pipeline_tag ?? 'unknown'}`,
    `Author: ${model.author ?? 'unknown'}`,
    `Tags: ${(model.tags ?? []).join(', ')}`,
    `Description: ${model.description ?? ''}`,
    `Sibling Files: ${JSON.stringify(siblingNames)}`,
    `System Info: ${JSON.stringify(systemSummary)}`,
    '',
    'Current code:',
    context.currentCode || '(empty)',
    '',
    'Execution error:',
    context.executionError?.trim() || '(none captured)',
    '',
    'Green console / stderr output:',
    context.stderrOutput?.trim() || '(none captured)',
    '',
    'Installed modules from pip freeze:',
    context.installedModules?.trim() || '(no modules captured)',
    '',
    'README:',
    readme || '(README not available)',
  ].join('\n');

  const payload = await requestAssistant(settings, {
    model: CLAUDE_SONNET_MODEL,
    apiKey: claudeApiKey,
    prompt,
  });
  const text = payload.text?.trim() ?? '';

  if (!text) {
    throw new Error('Assistant returned no text content.');
  }

  const parsed = parseRunnerResponse(text);
  if (!parsed.code || !parsed.analysis) {
    throw new Error(
      `Assistant returned an invalid fix payload.\nPayload preview:\n${previewText(text)}`
    );
  }

  return {
    code: parsed.code,
    analysis: parsed.analysis,
    dependencies: dedupeDependencies(
      Array.isArray(parsed.dependencies)
        ? parsed.dependencies.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
        : []
    ),
  };
}

async function requestAssistant(settings: AppSettings, request: {model: string; apiKey: string; prompt: string}): Promise<AnthropicMessagesResponse> {
  if (settings.codeGenerationProvider === 'codex-cli' || settings.codeGenerationProvider === 'claude-code') {
    let prompt = request.prompt;
    for (const secret of [settings.hfToken, settings.claudeApiKey]) {
      if (secret?.trim()) prompt = prompt.split(secret).join('[REDACTED]');
    }
    return invoke('generate_code_with_cli', {
      provider: settings.codeGenerationProvider,
      executablePath: (settings.codeGenerationProvider === 'codex-cli' ? settings.codexCliPath : settings.claudeCodePath) || null,
      prompt,
    });
  }
  return invoke('generate_code_with_claude', request);
}
