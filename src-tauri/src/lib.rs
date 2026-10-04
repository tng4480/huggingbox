use std::sync::{Arc, Mutex, OnceLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::process::Stdio;
use std::path::PathBuf;
use reqwest::Client;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use sysinfo::{ProcessesToUpdate, System};
use tauri::{AppHandle, Emitter, Manager, State};
use tokio::io::{AsyncBufReadExt, AsyncReadExt};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;
const HF_TRANSFORMERS_GIT_URL: &str = "git+https://github.com/huggingface/transformers.git";

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

static GPU_INFO_CACHE: OnceLock<Option<(String, u64)>> = OnceLock::new();

#[cfg(target_os = "windows")]
fn apply_no_window_std(command: &mut std::process::Command) {
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(target_os = "windows"))]
fn apply_no_window_std(_command: &mut std::process::Command) {}

#[cfg(target_os = "windows")]
fn apply_no_window_tokio(command: &mut tokio::process::Command) {
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(target_os = "windows"))]
fn apply_no_window_tokio(_command: &mut tokio::process::Command) {}

// ─── Managed State ────────────────────────────────────────────────────────────

pub struct ExecutionHandle {
    pub execution_pid: Arc<Mutex<Option<u32>>>,
    pub download_pid: Arc<Mutex<Option<u32>>>,
}

impl ExecutionHandle {
    pub fn new() -> Self {
        Self {
            execution_pid: Arc::new(Mutex::new(None)),
            download_pid: Arc::new(Mutex::new(None)),
        }
    }
}

// ─── GPU vendor classification ─────────────────────────────────────────────────

/// Classify a GPU name string into a backend identifier.
/// Returns one of: "cuda", "amd", "intel", "cpu".
/// "cuda" is used for NVIDIA so the value maps directly to HB_GPU_BACKEND.
fn classify_vendor(name: &str) -> String {
    let lower = name.to_lowercase();
    if lower.contains("nvidia")
        || lower.contains("geforce")
        || lower.contains("quadro")
        || lower.contains("rtx")
        || lower.contains("gtx")
        || lower.contains("tesla")
    {
        "cuda".to_string()
    } else if lower.contains("amd")
        || lower.contains("radeon")
        || lower.contains("rx ")
        || lower.contains("vega")
        || lower.contains("navi")
        || lower.contains("polaris")
    {
        "amd".to_string()
    } else if lower.contains("intel")
        || lower.contains(" arc ")
        || lower.contains("iris")
        || lower.contains("uhd graphics")
        || lower.contains("hd graphics")
    {
        "intel".to_string()
    } else {
        "cpu".to_string()
    }
}

// ─── Payload types ────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct SystemInfoPayload {
    total_ram: u64,
    available_ram: u64,
    gpu_name: Option<String>,
    gpu_vram: Option<u64>,
    /// One of: "cuda" (NVIDIA), "amd", "intel", "cpu" (unknown/no GPU).
    /// Matches the HB_GPU_BACKEND env var values passed to Python subprocesses.
    gpu_vendor: Option<String>,
    os_name: String,
}

#[derive(Serialize)]
pub struct PythonInfo {
    path: String,
    version: String,
    ready: bool,
}

#[derive(Serialize, Clone)]
struct StreamPayload {
    text: String,
}

#[derive(Serialize, Clone)]
struct DonePayload {
    exit_code: i32,
}

#[derive(Serialize, Clone)]
struct ExecutionStatsPayload {
    pid: u32,
    rss_bytes: u64,
    virtual_memory_bytes: u64,
    gpu_memory_bytes: Option<u64>,
    elapsed_ms: u64,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct ShellCommandResult {
    stdout: String,
    stderr: String,
    exit_code: i32,
    cwd: String,
}

#[derive(Serialize, Clone)]
struct DownloadStatsPayload {
    percent: f64,
    downloaded_bytes: u64,
    total_bytes: u64,
    speed_bps: f64,
    eta_seconds: Option<f64>,
    phase: String,
    files_done: usize,
    files_total: usize,
    filename: Option<String>,
}

#[derive(Deserialize)]
struct DownloadJsonLine {
    percent: f64,
    downloaded_bytes: u64,
    total_bytes: u64,
    speed_bps: f64,
    eta_seconds: Option<f64>,
    phase: String,
    files_done: usize,
    files_total: usize,
    filename: Option<String>,
}

#[derive(Clone, Default)]
struct DownloadTelemetrySnapshot {
    total_bytes: u64,
    files_done: usize,
    files_total: usize,
    filename: Option<String>,
}

#[derive(Serialize)]
struct CodeCacheRecord {
    code: String,
}

#[derive(Serialize, Deserialize)]
struct ClaudeGenerationResponse {
    text: String,
}

#[derive(Serialize)]
struct PythonBootstrapInfo {
    ready: bool,
    path: String,
    message: String,
}

#[derive(Serialize)]
struct DownloadedModelPayload {
    id: String,
    name: String,
    pipeline_tag: String,
    size_bytes: u64,
    last_used: String,
    storage_path: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelEnvironmentPayload {
    model_id: String,
    python_path: String,
    size_bytes: Option<u64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GpuInfoPayload {
    id: String,
    name: String,
    vram_gb: Option<u64>,
    backend: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelWorkspaceEntryPayload {
    name: String,
    relative_path: String,
    is_dir: bool,
    size_bytes: Option<u64>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelDependencyProbeResult {
    #[serde(default)]
    missing_packages: Vec<String>,
    #[serde(default)]
    required_packages: Vec<String>,
    #[serde(default)]
    compatibility_error: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase", default)]
struct AppSettingsPayload {
    // Kept for back-compat with old settings.json files; no longer used in UI
    #[serde(default)]
    claude_api_key: String,
    hf_token: String,
    model_storage_path: String,
    env_storage_path: String,
    preferred_device: String,
    selected_gpu_id: Option<String>,
    #[serde(default)]
    code_generation_provider: String,
    #[serde(default)]
    claude_auto_install_dependencies: bool,
    theme: String,
}

// ─── Helper: find Python executable ──────────────────────────────────────────

fn bundled_python_candidates(app: &AppHandle) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(data_dir) = app.path().app_data_dir() {
        let py_root = data_dir.join("python");
        #[cfg(target_os = "windows")]
        {
            paths.push(py_root.join("python.exe"));
            paths.push(py_root.join("Scripts").join("python.exe"));
        }
        #[cfg(not(target_os = "windows"))]
        {
            paths.push(py_root.join("bin").join("python3"));
            paths.push(py_root.join("bin").join("python"));
        }
    }
    paths
}

fn app_python_root(app: &AppHandle) -> Option<PathBuf> {
    app.path().app_data_dir().ok().map(|d| d.join("python"))
}

fn validate_model_id(model_id: &str) -> Result<(), String> {
    if model_id.trim().is_empty() {
        return Err("Model id is required.".to_string());
    }
    if model_id.contains("..") || model_id.contains('\\') || model_id.starts_with('/') {
        return Err("Invalid model id path.".to_string());
    }
    Ok(())
}

async fn find_python(app: &AppHandle) -> Result<String, String> {
    for candidate in bundled_python_candidates(app) {
        if !candidate.exists() {
            continue;
        }
        let mut cmd = tokio::process::Command::new(&candidate);
        apply_no_window_tokio(&mut cmd);
        if let Ok(output) = cmd.arg("--version").output().await {
            if output.status.success() {
                return Ok(candidate.to_string_lossy().to_string());
            }
        }
    }

    for candidate in ["python3", "python"] {
        let mut cmd = tokio::process::Command::new(candidate);
        apply_no_window_tokio(&mut cmd);
        if let Ok(output) = cmd.arg("--version").output().await {
            if output.status.success() {
                return Ok(candidate.to_string());
            }
        }
    }
    Err("Python not found. Install Python 3.11+ or bundle Python into the app data directory.".to_string())
}

/// Returns the directory that *contains* the `hf_auto_runner` package so it
/// can be prepended to PYTHONPATH. In a production bundle Tauri copies the
/// folder into the resource dir; during `tauri dev` the source tree itself is
/// used as a fallback.
fn hf_auto_runner_parent_dir(app: &AppHandle) -> Option<PathBuf> {
    // Production: Tauri puts bundled resources next to the binary.
    if let Ok(res_dir) = app.path().resource_dir() {
        let candidate = res_dir.join("hf_auto_runner");
        if candidate.is_dir() {
            // parent of hf_auto_runner/
            if let Some(parent) = candidate.parent() {
                return Some(parent.to_path_buf());
            }
        }
    }

    // Dev fallback: the source tree is two levels above src-tauri/
    // (i.e., the workspace root where hf_auto_runner/ lives).
    if let Ok(exe) = std::env::current_exe() {
        // During `tauri dev` the exe is at target/debug/huggingbox(.exe)
        // Walk up from the exe to find the workspace root containing hf_auto_runner.
        let mut dir = exe.as_path();
        for _ in 0..8 {
            if let Some(parent) = dir.parent() {
                if parent.join("hf_auto_runner").is_dir() {
                    return Some(parent.to_path_buf());
                }
                dir = parent;
            } else {
                break;
            }
        }
    }

    // Last resort: current working directory
    if let Ok(cwd) = std::env::current_dir() {
        if cwd.join("hf_auto_runner").is_dir() {
            return Some(cwd);
        }
    }

    None
}

fn ffmpeg_binary_name() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "ffmpeg.exe"
    }
    #[cfg(not(target_os = "windows"))]
    {
        "ffmpeg"
    }
}

fn resolve_ffmpeg_path(app: &AppHandle) -> Option<String> {
    let binary = ffmpeg_binary_name();

    if let Ok(res_dir) = app.path().resource_dir() {
        let candidate = res_dir.join("bin").join(binary);
        if candidate.is_file() {
            return Some(candidate.to_string_lossy().to_string());
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.as_path();
        for _ in 0..8 {
            if let Some(parent) = dir.parent() {
                let candidate = parent.join("src-tauri").join("bin").join(binary);
                if candidate.is_file() {
                    return Some(candidate.to_string_lossy().to_string());
                }
                dir = parent;
            } else {
                break;
            }
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        let candidate = cwd.join("src-tauri").join("bin").join(binary);
        if candidate.is_file() {
            return Some(candidate.to_string_lossy().to_string());
        }
    }

    let mut probe = std::process::Command::new(binary);
    apply_no_window_std(&mut probe);
    if probe
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        return Some(binary.to_string());
    }

    None
}

fn model_venv_dir(app: &AppHandle, model_id: &str) -> Result<PathBuf, String> {
    validate_model_id(model_id)?;
    let data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app data dir: {}", e))?;
    let mut dir = data_dir.join("venvs");
    for segment in model_id.split('/') {
        if !segment.trim().is_empty() {
            dir.push(segment.trim());
        }
    }
    Ok(dir)
}

fn model_venv_dir_from_root(root: &PathBuf, model_id: &str) -> Result<PathBuf, String> {
    validate_model_id(model_id)?;
    let mut dir = root.clone();
    for segment in model_id.split('/') {
        if !segment.trim().is_empty() {
            dir.push(segment.trim());
        }
    }
    Ok(dir)
}

fn venv_python_path(venv_dir: &PathBuf) -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        venv_dir.join("Scripts").join("python.exe")
    }
    #[cfg(not(target_os = "windows"))]
    {
        venv_dir.join("bin").join("python")
    }
}

async fn ensure_model_venv_python(app: &AppHandle, model_id: &str) -> Result<String, String> {
    let venv_dir = model_venv_dir(app, model_id)?;
    ensure_venv_python_at_dir(app, &venv_dir).await
}

async fn ensure_venv_python_at_dir(app: &AppHandle, venv_dir: &PathBuf) -> Result<String, String> {
    let python_path = venv_python_path(&venv_dir);

    if python_path.exists() {
        let mut cmd = tokio::process::Command::new(&python_path);
        apply_no_window_tokio(&mut cmd);
        if let Ok(out) = cmd.arg("--version").output().await {
            if out.status.success() {
                return Ok(python_path.to_string_lossy().to_string());
            }
        }
    }

    if let Some(parent) = venv_dir.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create venv parent directory: {}", e))?;
    }

    let base_python = find_python(app).await?;
    let mut create_cmd = tokio::process::Command::new(&base_python);
    apply_no_window_tokio(&mut create_cmd);
    let create = create_cmd
        .args(["-m", "venv", &venv_dir.to_string_lossy()])
        .output()
        .await
        .map_err(|e| format!("Failed to create model venv: {}", e))?;

    if !create.status.success() {
        let stderr = String::from_utf8_lossy(&create.stderr);
        return Err(format!("Failed to create model venv: {}", stderr.trim()));
    }

    if !python_path.exists() {
        return Err("Model venv was created but Python executable was not found.".to_string());
    }

    // Ensure pip is available in the venv before package checks/installs.
    let mut pip_cmd = tokio::process::Command::new(&python_path);
    apply_no_window_tokio(&mut pip_cmd);
    let pip_ok = pip_cmd
        .args(["-m", "pip", "--version"])
        .output()
        .await
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !pip_ok {
        let mut ensurepip_cmd = tokio::process::Command::new(&python_path);
        apply_no_window_tokio(&mut ensurepip_cmd);
        let _ = ensurepip_cmd
            .args(["-m", "ensurepip", "--upgrade"])
            .output()
            .await;
    }

    Ok(python_path.to_string_lossy().to_string())
}

async fn resolve_python(app: &AppHandle, model_id: Option<&str>) -> Result<String, String> {
    if let Some(id) = model_id {
        if !id.trim().is_empty() {
            return ensure_model_venv_python(app, id).await;
        }
    }
    find_python(app).await
}

/// Returns the root directory under which per-model venvs are stored.
/// Uses the user-configured path when non-empty; otherwise defaults to
/// `<app_data>/venvs`.
fn effective_venv_root(app: &AppHandle, custom_path: Option<&str>) -> Result<PathBuf, String> {
    if let Some(p) = custom_path {
        let trimmed = p.trim();
        if !trimmed.is_empty() {
            let path = PathBuf::from(expand_home_dir(trimmed));
            std::fs::create_dir_all(&path)
                .map_err(|e| format!("Failed to create env storage dir: {}", e))?;
            return Ok(path);
        }
    }
    // Default: AppData/venvs
    let data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app data dir: {}", e))?;
    Ok(data_dir.join("venvs"))
}

fn model_dir_from_id(storage_path: &str, model_id: &str) -> PathBuf {
    let expanded = expand_home_dir(storage_path);
    let mut path = PathBuf::from(expanded);
    for segment in model_id.split('/') {
        path.push(segment);
    }
    path
}

fn normalize_relative_workspace_path(input: &str) -> Result<String, String> {
    let trimmed = input.trim().replace('\\', "/");
    if trimmed.is_empty() {
        return Ok(String::new());
    }
    if trimmed.starts_with('/') {
        return Err("Absolute paths are not allowed.".to_string());
    }
    let mut normalized_segments = Vec::new();
    for raw in trimmed.split('/') {
        let seg = raw.trim();
        if seg.is_empty() || seg == "." {
            continue;
        }
        if seg == ".." {
            return Err("Path traversal is not allowed.".to_string());
        }
        normalized_segments.push(seg.to_string());
    }
    Ok(normalized_segments.join("/"))
}

fn resolve_model_workspace_root(model_id: &str, storage_path: &str) -> Result<PathBuf, String> {
    validate_model_id(model_id)?;
    let root = model_dir_from_id(storage_path, model_id);
    std::fs::create_dir_all(&root)
        .map_err(|e| format!("Failed to create model workspace root: {}", e))?;
    Ok(root)
}

fn resolve_model_workspace_path(root: &PathBuf, relative: &str) -> Result<PathBuf, String> {
    let normalized = normalize_relative_workspace_path(relative)?;
    let mut out = root.clone();
    if !normalized.is_empty() {
        for seg in normalized.split('/') {
            out.push(seg);
        }
    }
    Ok(out)
}

fn workspace_relative_path(root: &PathBuf, path: &PathBuf) -> String {
    path.strip_prefix(root)
        .ok()
        .map(|rel| {
            rel.iter()
                .map(|s| s.to_string_lossy().to_string())
                .collect::<Vec<_>>()
                .join("/")
        })
        .unwrap_or_default()
}

fn compute_dir_size(path: &PathBuf) -> u64 {
    let mut total = 0_u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if let Ok(meta) = std::fs::metadata(&p) {
                if meta.is_file() {
                    total = total.saturating_add(meta.len());
                } else if meta.is_dir() {
                    total = total.saturating_add(compute_dir_size(&p));
                }
            }
        }
    }
    total
}

fn discover_model_envs(
    root: &PathBuf,
    current: &PathBuf,
    acc: &mut Vec<ModelEnvironmentPayload>,
    include_sizes: bool,
) {
    let marker = current.join("pyvenv.cfg");
    let python = venv_python_path(current);
    if marker.exists() && python.exists() {
        if let Ok(rel) = current.strip_prefix(root) {
            let model_id = rel
                .iter()
                .map(|s| s.to_string_lossy().to_string())
                .collect::<Vec<_>>()
                .join("/");
            acc.push(ModelEnvironmentPayload {
                model_id,
                python_path: python.to_string_lossy().to_string(),
                size_bytes: if include_sizes {
                    Some(compute_dir_size(current))
                } else {
                    None
                },
            });
        }
        return;
    }

    if let Ok(entries) = std::fs::read_dir(current) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                discover_model_envs(root, &p, acc, include_sizes);
            }
        }
    }
}

fn discover_models(root: &PathBuf, current: &PathBuf, acc: &mut Vec<DownloadedModelPayload>) {
    let marker = current.join(".huggingbox_complete");
    if marker.exists() {
        if let Ok(rel) = current.strip_prefix(root) {
            let id = rel
                .iter()
                .map(|s| s.to_string_lossy().to_string())
                .collect::<Vec<_>>()
                .join("/");
            let size_bytes = compute_dir_size(current);
            let last_used = std::fs::metadata(&marker)
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs().to_string())
                .unwrap_or_else(|| "unknown".to_string());
            acc.push(DownloadedModelPayload {
                id: id.clone(),
                name: id,
                pipeline_tag: "unknown".to_string(),
                size_bytes,
                last_used,
                storage_path: current.to_string_lossy().to_string(),
            });
        }
        return;
    }

    if let Ok(entries) = std::fs::read_dir(current) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                discover_models(root, &p, acc);
            }
        }
    }
}

fn expand_home_dir(input: &str) -> String {
    if let Some(rest) = input.strip_prefix("~/") {
        #[cfg(target_os = "windows")]
        {
            if let Ok(home) = std::env::var("USERPROFILE") {
                return format!("{}\\{}", home, rest.replace('/', "\\"));
            }
        }
        #[cfg(not(target_os = "windows"))]
        {
            if let Ok(home) = std::env::var("HOME") {
                return format!("{}/{}", home, rest);
            }
        }
    }
    input.to_string()
}

fn kill_pid(pid: u32, force: bool) {
    #[cfg(target_os = "windows")]
    {
        let mut args = vec!["/PID".to_string(), pid.to_string()];
        if force {
            args.insert(0, "/F".to_string());
        }
        let _ = std::process::Command::new("taskkill")
            .args(args)
            .output();
    }
    #[cfg(not(target_os = "windows"))]
    {
        let signal = if force { "-KILL" } else { "-TERM" };
        let _ = std::process::Command::new("kill")
            .args([signal, &pid.to_string()])
            .output();
    }
}

fn detect_gpu_windows() -> Option<(String, u64)> {
    #[cfg(target_os = "windows")]
    {
        GPU_INFO_CACHE
            .get_or_init(|| {
                // Prefer CUDA devices from nvidia-smi when available.
                if let Some(primary_cuda) = detect_cuda_gpus().into_iter().next() {
                    return Some((primary_cuda.name, primary_cuda.vram_gb.unwrap_or(0)));
                }

                let mut command = std::process::Command::new("powershell");
                apply_no_window_std(&mut command);
                let ps_script = r#"
$adapters = Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0*" -ErrorAction SilentlyContinue
$res = @()
foreach ($a in $adapters) {
    if (!$a.DriverDesc -or $a.DriverDesc -match 'Virtual|Basic Render|Remote Display|Parsec') { continue }
    $ram = 0
    if ($a."HardwareInformation.qwMemorySize") {
        $ram = $a."HardwareInformation.qwMemorySize"
    } elseif ($a."HardwareInformation.MemorySize") {
        $val = $a."HardwareInformation.MemorySize"
        if ($val -is [byte[]] -and $val.Length -ge 8) {
            $ram = [BitConverter]::ToInt64($val, 0)
        } elseif ($val -is [byte[]] -and $val.Length -ge 4) {
            $ram = [BitConverter]::ToInt32($val, 0)
        } else {
            try { $ram = [long]$val } catch {}
        }
    }
    if ($ram -gt 0) {
        $res += [PSCustomObject]@{ Name=$a.DriverDesc; AdapterRAM=([long]$ram) }
    }
}
if ($res.Length -eq 0) {
    $gpus = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
    if ($gpus) {
        foreach ($g in $gpus) {
            if ($g.Name -match 'Virtual|Basic Render|Remote Display|Parsec') { continue }
            $res += [PSCustomObject]@{ Name=$g.Name; AdapterRAM=([long]$g.AdapterRAM) }
        }
    }
}
$res | ConvertTo-Json -Compress
"#;
                let output = command
                    .args([
                        "-NoProfile",
                        "-Command",
                        ps_script,
                    ])
                    .output()
                    .ok()?;
                if !output.status.success() {
                    return None;
                }

                let raw = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if raw.is_empty() {
                    return None;
                }

                let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
                let rows: Vec<serde_json::Value> = if value.is_array() {
                    value.as_array()?.clone()
                } else {
                    vec![value]
                };

                let is_virtual = |name: &str| {
                    let lower = name.to_lowercase();
                    lower.contains("virtual")
                        || lower.contains("basic render")
                        || lower.contains("remote display")
                        || lower.contains("parsec")
                };

                let preferred = rows
                    .iter()
                    .find_map(|row| {
                        let name = row.get("Name")?.as_str()?.to_string();
                        if is_virtual(&name) {
                            return None;
                        }
                        let bytes = row.get("AdapterRAM").and_then(|v| v.as_u64()).unwrap_or(0);
                        let vram_gb = if bytes > 0 {
                            (bytes / (1024_u64 * 1024_u64 * 1024_u64)).max(1)
                        } else {
                            0
                        };
                        Some((name, vram_gb))
                    });

                if preferred.is_some() {
                    preferred
                } else {
                    rows.first().and_then(|row| {
                        let name = row.get("Name")?.as_str()?.to_string();
                        let bytes = row.get("AdapterRAM").and_then(|v| v.as_u64()).unwrap_or(0);
                        let vram_gb = if bytes > 0 {
                            (bytes / (1024_u64 * 1024_u64 * 1024_u64)).max(1)
                        } else {
                            0
                        };
                        Some((name, vram_gb))
                    })
                }
            })
            .clone()
    }
    #[cfg(not(target_os = "windows"))]
    {
        None
    }
}

fn detect_cuda_gpus() -> Vec<GpuInfoPayload> {
    let mut command = std::process::Command::new("nvidia-smi");
    apply_no_window_std(&mut command);
    let output = command
        .args(["--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"])
        .output();

    let Ok(output) = output else {
        return vec![];
    };
    if !output.status.success() {
        return vec![];
    }

    let raw = String::from_utf8_lossy(&output.stdout);
    let mut gpus = Vec::new();
    for line in raw.lines() {
        let parts: Vec<&str> = line.split(',').map(|x| x.trim()).collect();
        if parts.len() < 2 {
            continue;
        }
        let id = parts[0].to_string();
        let name = parts[1].to_string();
        let vram_gb = parts
            .get(2)
            .and_then(|mb| mb.parse::<u64>().ok())
            .map(|mb| (mb / 1024).max(1));
        gpus.push(GpuInfoPayload {
            id,
            name,
            vram_gb,
            backend: "cuda".to_string(),
        });
    }
    gpus
}

fn ensure_temp_workspace(app: &AppHandle) -> Result<PathBuf, String> {
    let root = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app data dir: {}", e))?
        .join("temp");
    std::fs::create_dir_all(&root).map_err(|e| format!("Failed to create temp workspace: {}", e))?;
    Ok(root)
}

fn cleanup_old_temp_scripts(app: &AppHandle) {
    if let Ok(root) = ensure_temp_workspace(app) {
        if let Ok(entries) = std::fs::read_dir(&root) {
            for entry in entries.flatten() {
                let path = entry.path();
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
                if name.starts_with("run_") && name.ends_with(".py") {
                    let _ = std::fs::remove_file(path);
                }
            }
        }
    }
}

fn get_db_path(app: &AppHandle) -> Result<PathBuf, String> {
    let data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app data dir: {}", e))?;

    std::fs::create_dir_all(&data_dir)
        .map_err(|e| format!("Failed to create app data dir: {}", e))?;

    Ok(data_dir.join("huggingbox.db"))
}

fn get_settings_path(app: &AppHandle) -> Result<PathBuf, String> {
    let data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app data dir: {}", e))?;

    std::fs::create_dir_all(&data_dir)
        .map_err(|e| format!("Failed to create app data dir: {}", e))?;

    Ok(data_dir.join("settings.json"))
}

fn init_db(conn: &Connection) -> Result<(), String> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS code_cache(
            cache_key TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            code TEXT NOT NULL,
            created_at TEXT NOT NULL
        )",
        [],
    )
    .map_err(|e| format!("Failed to init code_cache table: {}", e))?;

    Ok(())
}

// ─── Commands ─────────────────────────────────────────────────────────────────

#[tauri::command]
fn get_system_info() -> SystemInfoPayload {
    let mut sys = System::new_all();
    sys.refresh_memory();

    let gpu = detect_gpu_windows();
    let gpu_vendor = gpu.as_ref().map(|g| classify_vendor(&g.0));

    SystemInfoPayload {
        total_ram: sys.total_memory(),
        available_ram: sys.available_memory(),
        gpu_name: gpu.as_ref().map(|g| g.0.clone()),
        gpu_vram: gpu.as_ref().map(|g| g.1),
        gpu_vendor,
        os_name: System::long_os_version().unwrap_or_else(|| "Unknown OS".to_string()),
    }
}

#[tauri::command]
fn list_gpus() -> Vec<GpuInfoPayload> {
    let cuda = detect_cuda_gpus();
    if !cuda.is_empty() {
        return cuda;
    }

    #[cfg(target_os = "windows")]
    {
        let mut ps_cmd = std::process::Command::new("powershell");
        apply_no_window_std(&mut ps_cmd);
        let output = ps_cmd
            .args([
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json -Compress",
            ])
            .output();

        let Ok(output) = output else {
            return vec![];
        };
        if !output.status.success() {
            return vec![];
        }

        let raw = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if raw.is_empty() {
            return vec![];
        }

        let value: serde_json::Value = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(_) => return vec![],
        };

        let rows: Vec<serde_json::Value> = if value.is_array() {
            value.as_array().cloned().unwrap_or_default()
        } else {
            vec![value]
        };

        return rows
            .into_iter()
            .enumerate()
            .filter_map(|(idx, row)| {
                let name = row.get("Name")?.as_str()?.to_string();
                let bytes = row.get("AdapterRAM").and_then(|v| v.as_u64()).unwrap_or(0);
                let vram_gb = if bytes > 0 {
                    Some((bytes / (1024_u64 * 1024_u64 * 1024_u64)).max(1))
                } else {
                    None
                };
                Some(GpuInfoPayload {
                    id: idx.to_string(),
                    name,
                    vram_gb,
                    backend: "display".to_string(),
                })
            })
            .collect();
    }

    #[cfg(not(target_os = "windows"))]
    {
        vec![]
    }
}

#[tauri::command]
fn load_app_settings(app: AppHandle) -> Result<Option<AppSettingsPayload>, String> {
    let path = get_settings_path(&app)?;
    if !path.exists() {
        return Ok(None);
    }

    let raw = std::fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read settings file: {}", e))?;
    let parsed = serde_json::from_str::<AppSettingsPayload>(&raw)
        .map_err(|e| format!("Failed to parse settings file: {}", e))?;
    Ok(Some(parsed))
}

#[tauri::command]
fn save_app_settings(app: AppHandle, settings: AppSettingsPayload) -> Result<(), String> {
    let path = get_settings_path(&app)?;
    let json = serde_json::to_string_pretty(&settings)
        .map_err(|e| format!("Failed to serialize settings: {}", e))?;
    std::fs::write(&path, json).map_err(|e| format!("Failed to write settings file: {}", e))?;
    Ok(())
}

#[tauri::command]
async fn bootstrap_python_environment(app: AppHandle) -> Result<PythonBootstrapInfo, String> {
    if let Some(py_root) = app_python_root(&app) {
        std::fs::create_dir_all(&py_root)
            .map_err(|e| format!("Failed to create python environment root: {}", e))?;
    }

    match find_python(&app).await {
        Ok(path) => Ok(PythonBootstrapInfo {
            ready: true,
            path,
            message: "Python runtime available".to_string(),
        }),
        Err(msg) => Ok(PythonBootstrapInfo {
            ready: false,
            path: String::new(),
            message: msg,
        }),
    }
}

#[tauri::command]
async fn detect_python(app: AppHandle) -> PythonInfo {
    for candidate in bundled_python_candidates(&app) {
        if !candidate.exists() {
            continue;
        }
        let mut command = tokio::process::Command::new(&candidate);
        apply_no_window_tokio(&mut command);
        if let Ok(output) = command.arg("--version").output().await
        {
            if output.status.success() {
                let raw = String::from_utf8_lossy(&output.stdout).trim().to_string();
                let version = if raw.is_empty() {
                    String::from_utf8_lossy(&output.stderr).trim().to_string()
                } else {
                    raw
                };
                return PythonInfo {
                    path: candidate.to_string_lossy().to_string(),
                    version,
                    ready: true,
                };
            }
        }
    }

    for candidate in ["python3", "python"] {
        let mut command = tokio::process::Command::new(candidate);
        apply_no_window_tokio(&mut command);
        if let Ok(output) = command.arg("--version").output().await
        {
            if output.status.success() {
                let raw = String::from_utf8_lossy(&output.stdout).trim().to_string();
                let version = if raw.is_empty() {
                    String::from_utf8_lossy(&output.stderr).trim().to_string()
                } else {
                    raw
                };
                return PythonInfo { path: candidate.to_string(), version, ready: true };
            }
        }
    }
    PythonInfo { path: String::new(), version: String::new(), ready: false }
}

#[tauri::command]
async fn generate_python_code_local(
    app: AppHandle,
    model_id: String,
    hf_token: Option<String>,
    gguf_file: Option<String>,
) -> Result<String, String> {
    let python = find_python(&app).await?;
    let mut cmd = tokio::process::Command::new(&python);
    apply_no_window_tokio(&mut cmd);
    cmd.env("HB_DEBUG", "1");
    if let Some(pypath) = hf_auto_runner_parent_dir(&app) {
        cmd.env("PYTHONPATH", pypath.to_string_lossy().to_string());
    }
    // Set token as env var so inspector.py can pick it up
    if let Some(ref token) = hf_token {
        if !token.is_empty() {
            cmd.env("HF_TOKEN", token);
        }
    }
    // Forward GPU vendor + VRAM so the script generator can bake hardware-specific code
    let gpu_info_generate = detect_gpu_windows();
    let gpu_backend_generate = gpu_info_generate
        .as_ref()
        .map(|g| classify_vendor(&g.0))
        .unwrap_or_else(|| "cpu".to_string());
    let gpu_vram_bytes_generate = gpu_info_generate
        .as_ref()
        .map(|g| g.1.saturating_mul(1_073_741_824u64))
        .unwrap_or(0u64);
    cmd.env("HB_GPU_BACKEND", &gpu_backend_generate);
    cmd.env("HB_GPU_VRAM_BYTES", gpu_vram_bytes_generate.to_string());
    // Quantization chosen in the UI; the Python generator uses it instead of auto-picking.
    if let Some(ref file) = gguf_file {
        let trimmed = file.trim();
        if !trimmed.is_empty() {
            cmd.env("HB_GGUF_FILE", trimmed);
        }
    }
    let mut args = vec!["-m".to_string(), "hf_auto_runner".to_string(), "generate".to_string(), model_id.clone()];
    if let Some(ref token) = hf_token {
        if !token.is_empty() {
            args.push("--hf-token".to_string());
            args.push(token.clone());
        }
    }
    let output = cmd
        .args(&args)
        .output()
        .await
        .map_err(|e| format!("Failed to call python hf_auto_runner script logic: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if !output.status.success() {
        // It may fail but still print valid JSON; attach diagnostics when possible.
        if let Ok(mut value) = serde_json::from_str::<serde_json::Value>(stdout.trim()) {
            if let Some(obj) = value.as_object_mut() {
                if !stderr.trim().is_empty() {
                    obj.insert("stderr".to_string(), serde_json::Value::String(stderr.clone()));
                }
                obj.insert(
                    "exitCode".to_string(),
                    serde_json::Value::String(output.status.to_string()),
                );
            }
            return Ok(value.to_string());
        }
        return Err(format!(
            "Python hf_auto_runner failed.\nstatus: {}\nstderr:\n{}\nstdout:\n{}",
            output.status, stderr, stdout
        ));
    }

    // Validate that successful runs return valid JSON and include debug stderr context.
    let mut value = serde_json::from_str::<serde_json::Value>(stdout.trim()).map_err(|e| {
        format!(
            "hf_auto_runner returned non-JSON stdout.\nparse_error: {}\nstderr:\n{}\nstdout:\n{}",
            e, stderr, stdout
        )
    })?;

    if let Some(obj) = value.as_object_mut() {
        if !stderr.trim().is_empty() {
            obj.insert("debugStderr".to_string(), serde_json::Value::String(stderr));
        }
    }

    Ok(value.to_string())
}

#[tauri::command]
async fn generate_code_with_claude(
    model: String,
    api_key: String,
    prompt: String,
) -> Result<ClaudeGenerationResponse, String> {
    let trimmed_key = api_key.trim();
    if trimmed_key.is_empty() {
        return Err("Anthropic API key is required.".to_string());
    }
    if prompt.trim().is_empty() {
        return Err("Claude prompt is empty.".to_string());
    }

    let client = Client::new();
    let body = serde_json::json!({
        "model": model,
        "max_tokens": 5000,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ]
            }
        ],
        "tools": [
            {
                "name": "web_search",
                "type": "web_search_20250305",
                "allowed_domains": [
                    "huggingface.co",
                    "github.com"
                ],
                "max_uses": 3
            }
        ],
        "thinking": {
            "type": "disabled"
        },
        "output_config": {
            "effort": "low"
        }
    });

    let response = client
        .post("https://api.anthropic.com/v1/messages")
        .header("content-type", "application/json")
        .header("x-api-key", trimmed_key)
        .header("anthropic-version", "2023-06-01")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Anthropic request failed: {}", e))?;

    let status = response.status();
    let payload = response
        .text()
        .await
        .map_err(|e| format!("Failed to read Anthropic response: {}", e))?;

    if !status.is_success() {
        return Err(format!(
            "Anthropic API error {}:\n{}",
            status.as_u16(),
            payload
        ));
    }

    let parsed: serde_json::Value = serde_json::from_str(&payload)
        .map_err(|e| format!("Failed to parse Anthropic response JSON: {}\n{}", e, payload))?;

    let text = parsed
        .get("content")
        .and_then(|v| v.as_array())
        .map(|blocks| {
            blocks
                .iter()
                .filter_map(|block| {
                    if block.get("type").and_then(|v| v.as_str()) == Some("text") {
                        block.get("text").and_then(|v| v.as_str()).map(|v| v.to_string())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();

    if text.trim().is_empty() {
        return Err(format!(
            "Anthropic API returned no text blocks.\n{}",
            payload
        ));
    }

    Ok(ClaudeGenerationResponse { text })
}

#[tauri::command]
async fn run_python_code(
    app: AppHandle,
    handle: State<'_, ExecutionHandle>,
    preferred_device: Option<String>,
    selected_gpu_id: Option<String>,
    model_id: Option<String>,
    venv_model_id: Option<String>,
    hf_token: Option<String>,
    user_input: Option<String>,
    env_storage_path: Option<String>,
    storage_path: Option<String>,
    script_relative_path: Option<String>,
    diffusion_mode: Option<String>,
    output_dir: Option<String>,
    negative_prompt: Option<String>,
    steps: Option<u32>,
    guidance_scale: Option<f64>,
    seed: Option<String>,
    num_images: Option<u32>,
    strength: Option<f64>,
    source_image_path: Option<String>,
    mask_image_path: Option<String>,
    multimodal_task: Option<String>,
    image_path: Option<String>,
    document_path: Option<String>,
) -> Result<(), String> {
    {
        let lock = handle.execution_pid.lock().unwrap();
        if lock.is_some() {
            return Err("An execution is already running. Stop it before starting another.".to_string());
        }
    }

    let execution_env_model = venv_model_id
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().to_string())
        .or_else(|| model_id.as_deref().map(|s| s.trim().to_string()));
    if let Some(ref env_model) = execution_env_model {
        validate_model_id(env_model)?;
    }

    let python = resolve_python(&app, execution_env_model.as_deref()).await?;

    let mut command = tokio::process::Command::new(&python);
    apply_no_window_tokio(&mut command);
    command
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .env("HB_DEBUG", "1")
        .env("HB_SKIP_DEP_INSTALL", "1");

    // Ensure isolated venvs can find the hf_auto_runner package
    if let Some(pypath) = hf_auto_runner_parent_dir(&app) {
        command.env("PYTHONPATH", pypath.to_string_lossy().to_string());
    }

    // Set HB_VENV_DIR so the Python env_manager uses the same venv directory
    // that Tauri's resolve_python() created.
    if let Some(ref mid) = execution_env_model {
        if !mid.trim().is_empty() {
            if let Ok(venv_root) = effective_venv_root(&app, env_storage_path.as_deref()) {
                let mut venv_dir = venv_root;
                for segment in mid.split('/') {
                    if !segment.trim().is_empty() {
                        venv_dir.push(segment.trim());
                    }
                }
                command.env("HB_VENV_DIR", venv_dir.to_string_lossy().to_string());
            }
        }
    }

    // Pass HF token as env var (belt-and-suspenders alongside CLI flag)
    if let Some(ref token) = hf_token {
        if !token.is_empty() {
            command.env("HF_TOKEN", token);
        }
    }

    // Forward GPU vendor + VRAM to Python so DependencyManager and ScriptGenerator
    // can install the correct wheels and generate hardware-specific code.
    let gpu_info_run = detect_gpu_windows();
    let gpu_backend_run = gpu_info_run
        .as_ref()
        .map(|g| classify_vendor(&g.0))
        .unwrap_or_else(|| "cpu".to_string());
    let gpu_vram_bytes_run = gpu_info_run
        .as_ref()
        .map(|g| g.1.saturating_mul(1_073_741_824u64))
        .unwrap_or(0u64);
    command.env("HB_GPU_BACKEND", &gpu_backend_run);
    command.env("HB_GPU_VRAM_BYTES", gpu_vram_bytes_run.to_string());

    // Pass user input as env var for the inference script
    if let Some(ref input) = user_input {
        if !input.is_empty() {
            command.env("HB_INPUT", input);
        }
    }

    let ffmpeg_path = resolve_ffmpeg_path(&app);
    if let Some(ref path) = ffmpeg_path {
        command.env("HB_FFMPEG_PATH", path);
    }

    let device = preferred_device.unwrap_or_else(|| "auto".to_string()).to_lowercase();
    if device == "cpu" {
        command.env("CUDA_VISIBLE_DEVICES", "-1");
    } else if let Some(gpu_id) = selected_gpu_id.as_deref() {
        let trimmed = gpu_id.trim();
        if !trimmed.is_empty() {
            command.env("CUDA_VISIBLE_DEVICES", trimmed);
        }
    }

    let model = model_id.unwrap_or_else(|| "".to_string());
    if model.is_empty() {
        return Err("Model ID is required for execution via hf_auto_runner.".to_string());
    }

    let trimmed_script_relative = script_relative_path
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());

    let workspace_root = if trimmed_script_relative.is_some() {
        let storage = storage_path
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or("Storage path is required when running a workspace file.")?;
        Some(resolve_model_workspace_root(&model, storage)?)
    } else {
        None
    };

    if let Some(ref mode) = diffusion_mode {
        if !mode.trim().is_empty() {
            command.env("HB_DIFFUSION_MODE", mode.trim());
        }
    }
    if let Some(ref prompt) = negative_prompt {
        if !prompt.trim().is_empty() {
            command.env("HB_NEGATIVE_PROMPT", prompt);
        }
    }
    if let Some(value) = steps {
        command.env("HB_STEPS", value.to_string());
    }
    if let Some(value) = guidance_scale {
        command.env("HB_GUIDANCE_SCALE", value.to_string());
    }
    if let Some(ref value) = seed {
        if !value.trim().is_empty() {
            command.env("HB_SEED", value.trim());
        }
    }
    if let Some(value) = num_images {
        command.env("HB_NUM_IMAGES", value.to_string());
    }
    if let Some(value) = strength {
        command.env("HB_STRENGTH", value.to_string());
    }
    if let Some(ref value) = source_image_path {
        if !value.trim().is_empty() {
            command.env("HB_SOURCE_IMAGE", value.trim());
        }
    }
    if let Some(ref value) = mask_image_path {
        if !value.trim().is_empty() {
            command.env("HB_MASK_PATH", value.trim());
        }
    }
    if let Some(ref value) = multimodal_task {
        if !value.trim().is_empty() {
            command.env("HB_MULTIMODAL_TASK", value.trim());
        }
    }
    if let Some(ref value) = image_path {
        if !value.trim().is_empty() {
            command.env("HB_IMAGE_PATH", value.trim());
        }
    }
    if let Some(ref value) = document_path {
        if !value.trim().is_empty() {
            command.env("HB_DOCUMENT_PATH", value.trim());
        }
    }

    let resolved_output_dir = if let Some(custom) = output_dir.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        if let Some(root) = &workspace_root {
            let path = resolve_model_workspace_path(root, custom)?;
            std::fs::create_dir_all(&path)
                .map_err(|e| format!("Failed to create output directory: {}", e))?;
            Some(path)
        } else {
            let path = PathBuf::from(custom);
            std::fs::create_dir_all(&path)
                .map_err(|e| format!("Failed to create output directory: {}", e))?;
            Some(path)
        }
    } else if let Some(root) = &workspace_root {
        let path = root.join("outputs");
        std::fs::create_dir_all(&path)
            .map_err(|e| format!("Failed to create output directory: {}", e))?;
        Some(path)
    } else {
        None
    };

    if let Some(ref path) = resolved_output_dir {
        command.env("HB_OUTPUT_DIR", path.to_string_lossy().to_string());
    }

    let run_target_path = if let (Some(root), Some(relative)) = (&workspace_root, &trimmed_script_relative) {
        let path = resolve_model_workspace_path(root, relative)?;
        if !path.exists() || !path.is_file() {
            return Err(format!("Workspace script does not exist: {}", relative));
        }
        Some(path)
    } else {
        None
    };

    let run_args = if let Some(script_path) = &run_target_path {
        vec![script_path.to_string_lossy().to_string()]
    } else {
        let mut args = vec![
            "-m".to_string(), "hf_auto_runner".to_string(), "run".to_string(), model.clone()
        ];
        if let Some(ref token) = hf_token {
            if !token.is_empty() {
                args.push("--hf-token".to_string());
                args.push(token.clone());
            }
        }
        if let Some(ref input) = user_input {
            if !input.is_empty() {
                args.push("--input".to_string());
                args.push(input.clone());
            }
        }
        args
    };

    let input_kind = if let Some(ref input) = user_input {
        if input.starts_with("__HBIMG__:") || input.starts_with("data:image/") {
            "image"
        } else if input.starts_with("http://") || input.starts_with("https://") {
            "url"
        } else if std::path::Path::new(input).exists() {
            "file-path"
        } else {
            "text"
        }
    } else {
        "none"
    };

    let _ = app.emit("execution-stdout", StreamPayload {
        text: format!(
            "[HuggingBox] Launching model execution\n  model: {}\n  env: {}\n  python: {}\n  device: {}\n  input: {}\n  ffmpeg: {}\n",
            model,
            execution_env_model.as_deref().unwrap_or("<default>"),
            python,
            device,
            input_kind,
            ffmpeg_path.as_deref().unwrap_or("<not found>")
        ),
    });
    if let Some(path) = env_storage_path.as_deref() {
        if !path.trim().is_empty() {
            let _ = app.emit("execution-stdout", StreamPayload {
                text: format!("[HuggingBox] Environment storage path: {}\n", path),
            });
        }
    }
    if let Some(script_path) = &run_target_path {
        let _ = app.emit("execution-stdout", StreamPayload {
            text: format!(
                "[HuggingBox] Executing visible editor file\n  workspace: {}\n  script: {}\n",
                workspace_root
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|| "<unknown>".to_string()),
                script_path.to_string_lossy()
            ),
        });
    }
    let _ = app.emit("execution-stdout", StreamPayload {
        text: format!("[HuggingBox] Runner command args: {:?}\n", run_args),
    });

    if let Some(root) = &workspace_root {
        command.current_dir(root);
    }

    let mut child = command
        .args(&run_args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start Python: {}", e))?;

    let started_at = std::time::Instant::now();

    // Store PID for cancellation
    let pid_arc = handle.execution_pid.clone();
    let child_pid = child.id();
    {
        let mut lock = pid_arc.lock().unwrap();
        *lock = child_pid;
    }

    let stdout = child.stdout.take().ok_or("no stdout")?;
    let stderr = child.stderr.take().ok_or("no stderr")?;

    let app_clone = app.clone();

    // Detach all I/O + wait logic into a background task so the command returns immediately
    tokio::spawn(async move {
        let app_out = app_clone.clone();
        let app_err = app_clone.clone();
        let app_done = app_clone.clone();
        let app_stats = app_clone.clone();
        let pid_for_stats = child_pid;
        let stats_started_at = started_at;

        let t_stats = tokio::spawn(async move {
            if let Some(pid_raw) = pid_for_stats {
                let pid = sysinfo::Pid::from_u32(pid_raw);
                let mut system = System::new_all();
                loop {
                    system.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
                    if let Some(process) = system.process(pid) {
                        let _ = app_stats.emit("execution-stats", ExecutionStatsPayload {
                            pid: pid_raw,
                            rss_bytes: process.memory(),
                            virtual_memory_bytes: process.virtual_memory(),
                            gpu_memory_bytes: None,
                            elapsed_ms: stats_started_at.elapsed().as_millis() as u64,
                        });
                    } else {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
                }
            }
        });

        let t_stdout = tokio::spawn(async move {
            let mut reader = tokio::io::BufReader::new(stdout);
            let mut buf = [0u8; 1024];
            loop {
                match reader.read(&mut buf).await {
                    Ok(0) => break,
                    Ok(n) => {
                        let text = String::from_utf8_lossy(&buf[..n]).to_string();
                        let _ = app_out.emit("execution-stdout", StreamPayload { text });
                    }
                    Err(_) => break,
                }
            }
        });

        let t_stderr = tokio::spawn(async move {
            let mut reader = tokio::io::BufReader::new(stderr);
            let mut buf = [0u8; 1024];
            loop {
                match reader.read(&mut buf).await {
                    Ok(0) => break,
                    Ok(n) => {
                        let text = String::from_utf8_lossy(&buf[..n]).to_string();
                        let _ = app_err.emit("execution-stderr", StreamPayload { text });
                    }
                    Err(_) => break,
                }
            }
        });

        let exit_code = match child.wait().await {
            Ok(status) => status.code().unwrap_or(-1),
            Err(_) => -1,
        };

        // Drain remaining I/O
        let _ = tokio::join!(t_stdout, t_stderr, t_stats);

        // Clear PID
        {
            let mut lock = pid_arc.lock().unwrap();
            *lock = None;
        }

        let _ = app_done.emit("execution-done", DonePayload { exit_code });
    });

    Ok(())
}

#[tauri::command]
fn cancel_execution(handle: State<'_, ExecutionHandle>) -> Result<(), String> {
    let pid = { *handle.execution_pid.lock().unwrap() };

    if let Some(pid) = pid {
        kill_pid(pid, false);
        std::thread::sleep(std::time::Duration::from_secs(3));
        let still_running = {
            #[cfg(target_os = "windows")]
            {
                std::process::Command::new("tasklist")
                    .args(["/FI", &format!("PID eq {}", pid)])
                    .output()
                    .ok()
                    .map(|o| String::from_utf8_lossy(&o.stdout).contains(&pid.to_string()))
                    .unwrap_or(false)
            }
            #[cfg(not(target_os = "windows"))]
            {
                std::process::Command::new("kill")
                    .args(["-0", &pid.to_string()])
                    .output()
                    .map(|o| o.status.success())
                    .unwrap_or(false)
            }
        };
        if still_running {
            kill_pid(pid, true);
        }
    }

    Ok(())
}

#[tauri::command]
fn is_model_downloaded(model_id: String, storage_path: String) -> Result<bool, String> {
    let model_dir = model_dir_from_id(&storage_path, &model_id);
    let marker = model_dir.join(".huggingbox_complete");
    Ok(marker.exists())
}

#[tauri::command]
async fn download_model(
    app: AppHandle,
    handle: State<'_, ExecutionHandle>,
    model_id: String,
    storage_path: String,
    hf_token: Option<String>,
) -> Result<(), String> {
    {
        let lock = handle.download_pid.lock().unwrap();
        if lock.is_some() {
            return Err("A model download is already in progress.".to_string());
        }
    }

    let python = resolve_python(&app, Some(&model_id)).await?;
    let model_dir = model_dir_from_id(&storage_path, &model_id);
    std::fs::create_dir_all(&model_dir).map_err(|e| format!("Failed to create model directory: {}", e))?;

    let script = r#"
import os
import sys
import time
import json

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

from huggingface_hub import HfApi, hf_hub_download

model_id = sys.argv[1]
target_dir = sys.argv[2]
token = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None

try:
    import hf_xet  # noqa: F401
    print("[HuggingBox] Xet high-performance downloads enabled.", flush=True)
except Exception:
    print("[HuggingBox] hf_xet not installed; using standard downloader.", flush=True)

api = HfApi(token=token)
info = api.model_info(model_id)
files = []
total = 0
for s in info.siblings:
    name = getattr(s, "rfilename", None)
    if not name:
        continue
    size = getattr(s, "size", None) or 0
    files.append((name, int(size)))
    total += int(size)

downloaded = 0
files_done = 0
files_total = len(files)
start = time.time()

def emit(phase, filename=None):
    elapsed = max(0.001, time.time() - start)
    speed = downloaded / elapsed
    remaining = max(0, total - downloaded)
    eta = (remaining / speed) if speed > 0 else None
    if total > 0:
        pct = (downloaded / total * 100.0)
    elif files_total > 0:
        pct = (files_done / files_total * 100.0)
    else:
        pct = 0.0
    payload = {
        "percent": pct,
        "downloaded_bytes": downloaded,
        "total_bytes": total,
        "speed_bps": speed,
        "eta_seconds": eta,
        "phase": phase,
        "files_done": files_done,
        "files_total": files_total,
        "filename": filename,
    }
    print("HB_DOWNLOAD_JSON:" + json.dumps(payload), flush=True)

print(f"[HuggingBox] Downloading {model_id} into {target_dir}", flush=True)
emit("starting")
for (filename, size) in files:
    hf_hub_download(
        repo_id=model_id,
        filename=filename,
        token=token,
        local_dir=target_dir,
    )
    files_done += 1
    downloaded += max(0, size)
    emit("downloading", filename)

marker = os.path.join(target_dir, ".huggingbox_complete")
with open(marker, "w", encoding="utf-8") as f:
    f.write(str(time.time()))
emit("complete")
print("[HuggingBox] Download complete.", flush=True)
"#;

    let script_path = std::env::temp_dir().join("huggingbox_download.py");
    std::fs::write(&script_path, script).map_err(|e| e.to_string())?;

    let token = hf_token.unwrap_or_default();
    let mut dl_cmd = tokio::process::Command::new(&python);
    apply_no_window_tokio(&mut dl_cmd);
    let mut child = dl_cmd
        .arg(script_path.to_str().ok_or("invalid path")?)
        .arg(model_id.clone())
        .arg(model_dir.to_string_lossy().to_string())
        .arg(token)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start model download: {}", e))?;

    let pid_arc = handle.download_pid.clone();
    {
        let mut lock = pid_arc.lock().unwrap();
        *lock = child.id();
    }

    let stdout = child.stdout.take().ok_or("no stdout")?;
    let stderr = child.stderr.take().ok_or("no stderr")?;
    let app2 = app.clone();
    let scanner_state = Arc::new(Mutex::new(DownloadTelemetrySnapshot::default()));
    let scanner_live = Arc::new(AtomicBool::new(true));

    let scanner_task = {
        let app_scan = app.clone();
        let model_dir_scan = model_dir.clone();
        let scanner_state = scanner_state.clone();
        let scanner_live = scanner_live.clone();
        tokio::spawn(async move {
            let mut last_size = compute_dir_size(&model_dir_scan);
            let mut last_tick = std::time::Instant::now();
            while scanner_live.load(Ordering::Relaxed) {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                if !scanner_live.load(Ordering::Relaxed) {
                    break;
                }
                if last_tick.elapsed() < std::time::Duration::from_secs(10) {
                    continue;
                }

                let current_size = compute_dir_size(&model_dir_scan);
                let now = std::time::Instant::now();
                let elapsed = (now - last_tick).as_secs_f64().max(0.001);
                let delta = current_size.saturating_sub(last_size);
                let speed = delta as f64 / elapsed;
                last_size = current_size;
                last_tick = now;

                let snap = scanner_state.lock().map(|s| s.clone()).unwrap_or_default();
                let percent = if snap.total_bytes > 0 {
                    (current_size as f64 / snap.total_bytes as f64 * 100.0).clamp(0.0, 100.0)
                } else if snap.files_total > 0 {
                    (snap.files_done as f64 / snap.files_total as f64 * 100.0).clamp(0.0, 100.0)
                } else {
                    0.0
                };
                let remaining = snap.total_bytes.saturating_sub(current_size);
                let eta = if speed > 0.0 && snap.total_bytes > 0 {
                    Some(remaining as f64 / speed)
                } else {
                    None
                };

                let _ = app_scan.emit("download-stats", DownloadStatsPayload {
                    percent,
                    downloaded_bytes: current_size,
                    total_bytes: snap.total_bytes,
                    speed_bps: speed,
                    eta_seconds: eta,
                    phase: "scanning".to_string(),
                    files_done: snap.files_done,
                    files_total: snap.files_total,
                    filename: snap.filename.clone(),
                });
            }
        })
    };

    let t1 = tokio::spawn(async move {
        let _ = app.emit("download-progress", StreamPayload {
            text: "[HuggingBox] Download folder-size sampler enabled (10s interval).\n".to_string(),
        });
        let mut lines = tokio::io::BufReader::new(stdout).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if let Some(json_part) = line.strip_prefix("HB_DOWNLOAD_JSON:") {
                if let Ok(stats) = serde_json::from_str::<DownloadJsonLine>(json_part) {
                    if let Ok(mut snap) = scanner_state.lock() {
                        snap.total_bytes = stats.total_bytes;
                        snap.files_done = stats.files_done;
                        snap.files_total = stats.files_total;
                        snap.filename = stats.filename.clone();
                    }
                    let _ = app.emit("download-stats", DownloadStatsPayload {
                        percent: stats.percent,
                        downloaded_bytes: stats.downloaded_bytes,
                        total_bytes: stats.total_bytes,
                        speed_bps: stats.speed_bps,
                        eta_seconds: stats.eta_seconds,
                        phase: stats.phase,
                        files_done: stats.files_done,
                        files_total: stats.files_total,
                        filename: stats.filename,
                    });
                }
            } else {
                let _ = app.emit("download-progress", StreamPayload { text: line + "\n" });
            }
        }
    });

    let t2 = tokio::spawn(async move {
        let mut lines = tokio::io::BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let _ = app2.emit("download-progress", StreamPayload { text: line + "\n" });
        }
    });

    let status = child.wait().await.map_err(|e| e.to_string())?;
    scanner_live.store(false, Ordering::Relaxed);
    let _ = tokio::join!(t1, t2);
    let _ = scanner_task.await;

    {
        let mut lock = pid_arc.lock().unwrap();
        *lock = None;
    }

    if status.success() {
        Ok(())
    } else {
        Err("Model download failed. You can retry to resume partial files.".to_string())
    }
}

#[tauri::command]
fn cancel_download(handle: State<'_, ExecutionHandle>) -> Result<(), String> {
    let pid = { *handle.download_pid.lock().unwrap() };
    if let Some(pid) = pid {
        kill_pid(pid, false);
        std::thread::sleep(std::time::Duration::from_secs(3));
        kill_pid(pid, true);
    }
    Ok(())
}

#[tauri::command]
fn list_downloaded_models(storage_path: String) -> Result<Vec<DownloadedModelPayload>, String> {
    let root = PathBuf::from(expand_home_dir(&storage_path));
    if !root.exists() {
        return Ok(vec![]);
    }
    let mut models = Vec::new();
    discover_models(&root, &root, &mut models);
    Ok(models)
}

#[tauri::command]
fn delete_downloaded_model(model_id: String, storage_path: String) -> Result<(), String> {
    validate_model_id(&model_id)?;
    let model_dir = model_dir_from_id(&storage_path, &model_id);
    if !model_dir.exists() {
        return Ok(());
    }
    std::fs::remove_dir_all(&model_dir)
        .map_err(|e| format!("Failed to remove downloaded model: {}", e))?;
    Ok(())
}

#[tauri::command]
fn list_model_workspace_entries(
    model_id: String,
    storage_path: String,
    directory: Option<String>,
) -> Result<Vec<ModelWorkspaceEntryPayload>, String> {
    let root = resolve_model_workspace_root(&model_id, &storage_path)?;
    let dir_rel = directory.unwrap_or_default();
    let dir_path = resolve_model_workspace_path(&root, &dir_rel)?;
    if !dir_path.exists() {
        std::fs::create_dir_all(&dir_path)
            .map_err(|e| format!("Failed to create workspace directory: {}", e))?;
    }
    if !dir_path.is_dir() {
        return Err("Selected workspace path is not a directory.".to_string());
    }

    let mut out = Vec::new();
    let entries = std::fs::read_dir(&dir_path)
        .map_err(|e| format!("Failed to read workspace directory: {}", e))?;
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(meta) = std::fs::metadata(&path) else {
            continue;
        };
        let name = entry.file_name().to_string_lossy().to_string();
        let relative_path = workspace_relative_path(&root, &path);
        out.push(ModelWorkspaceEntryPayload {
            name,
            relative_path,
            is_dir: meta.is_dir(),
            size_bytes: if meta.is_file() { Some(meta.len()) } else { None },
        });
    }

    out.sort_by(|a, b| {
        if a.is_dir != b.is_dir {
            return b.is_dir.cmp(&a.is_dir);
        }
        a.name.to_lowercase().cmp(&b.name.to_lowercase())
    });
    Ok(out)
}

#[tauri::command]
fn read_model_workspace_file(
    model_id: String,
    storage_path: String,
    relative_path: String,
) -> Result<String, String> {
    let root = resolve_model_workspace_root(&model_id, &storage_path)?;
    let file_path = resolve_model_workspace_path(&root, &relative_path)?;
    if !file_path.exists() {
        return Err("Workspace file does not exist.".to_string());
    }
    if !file_path.is_file() {
        return Err("Selected workspace path is not a file.".to_string());
    }
    std::fs::read_to_string(&file_path)
        .map_err(|e| format!("Failed to read workspace file: {}", e))
}

#[tauri::command]
fn read_binary_file(file_path: String) -> Result<Vec<u8>, String> {
    let path = PathBuf::from(file_path.trim());
    if !path.is_absolute() {
        return Err("Binary file path must be absolute.".to_string());
    }
    if !path.exists() {
        return Err("Binary file does not exist.".to_string());
    }
    if !path.is_file() {
        return Err("Binary path is not a file.".to_string());
    }
    std::fs::read(&path)
        .map_err(|e| format!("Failed to read binary file: {}", e))
}

#[tauri::command]
fn write_model_workspace_file(
    model_id: String,
    storage_path: String,
    relative_path: String,
    content: String,
) -> Result<(), String> {
    let root = resolve_model_workspace_root(&model_id, &storage_path)?;
    let file_path = resolve_model_workspace_path(&root, &relative_path)?;
    let parent = file_path
        .parent()
        .ok_or("Could not resolve workspace file parent directory.")?;
    std::fs::create_dir_all(parent)
        .map_err(|e| format!("Failed to create workspace file directory: {}", e))?;
    std::fs::write(&file_path, content)
        .map_err(|e| format!("Failed to write workspace file: {}", e))
}

#[tauri::command]
fn create_model_workspace_file(
    model_id: String,
    storage_path: String,
    relative_path: String,
) -> Result<(), String> {
    let root = resolve_model_workspace_root(&model_id, &storage_path)?;
    let file_path = resolve_model_workspace_path(&root, &relative_path)?;
    if file_path.exists() {
        return Err("Workspace file already exists.".to_string());
    }
    let parent = file_path
        .parent()
        .ok_or("Could not resolve workspace file parent directory.")?;
    std::fs::create_dir_all(parent)
        .map_err(|e| format!("Failed to create workspace file directory: {}", e))?;
    std::fs::write(&file_path, "")
        .map_err(|e| format!("Failed to create workspace file: {}", e))
}

#[tauri::command]
fn create_model_workspace_directory(
    model_id: String,
    storage_path: String,
    relative_path: String,
) -> Result<(), String> {
    let root = resolve_model_workspace_root(&model_id, &storage_path)?;
    let dir_path = resolve_model_workspace_path(&root, &relative_path)?;
    std::fs::create_dir_all(&dir_path)
        .map_err(|e| format!("Failed to create workspace directory: {}", e))
}

#[tauri::command]
fn list_model_environments(
    app: AppHandle,
    env_storage_path: Option<String>,
    include_sizes: Option<bool>,
) -> Result<Vec<ModelEnvironmentPayload>, String> {
    let root = effective_venv_root(&app, env_storage_path.as_deref())?;
    if !root.exists() {
        return Ok(vec![]);
    }
    let mut envs = Vec::new();
    discover_model_envs(&root, &root, &mut envs, include_sizes.unwrap_or(false));
    envs.sort_by(|a, b| a.model_id.cmp(&b.model_id));
    Ok(envs)
}

#[tauri::command]
async fn get_model_environment_size(
    app: AppHandle,
    model_id: String,
    env_storage_path: Option<String>,
) -> Result<u64, String> {
    let root = effective_venv_root(&app, env_storage_path.as_deref())?;
    let dir = model_venv_dir_from_root(&root, &model_id)?;
    if !dir.exists() {
        return Ok(0);
    }
    tokio::task::spawn_blocking(move || compute_dir_size(&dir))
        .await
        .map_err(|e| format!("Failed to compute model environment size: {}", e))
}

#[tauri::command]
fn delete_model_environment(
    app: AppHandle,
    model_id: String,
    env_storage_path: Option<String>,
) -> Result<(), String> {
    let root = effective_venv_root(&app, env_storage_path.as_deref())?;
    let dir = model_venv_dir_from_root(&root, &model_id)?;
    if !dir.exists() {
        return Ok(());
    }
    std::fs::remove_dir_all(&dir)
        .map_err(|e| format!("Failed to remove model environment: {}", e))?;
    Ok(())
}

#[tauri::command]
async fn check_packages(
    app: AppHandle,
    packages: Vec<String>,
    model_id: Option<String>,
    venv_model_id: Option<String>,
) -> Result<Vec<String>, String> {
    let resolved_env = venv_model_id
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().to_string())
        .or_else(|| model_id.as_deref().map(|s| s.trim().to_string()));
    if let Some(ref env_model) = resolved_env {
        validate_model_id(env_model)?;
    }
    let python = resolve_python(&app, resolved_env.as_deref()).await?;
    let mut missing = Vec::new();

    for pkg in &packages {
        let import_target = import_name_for_requirement(pkg);
        let mut chk_cmd = tokio::process::Command::new(&python);
        apply_no_window_tokio(&mut chk_cmd);
        let result = chk_cmd
            .args(["-c", &format!("import {}", import_target)])
            .output()
            .await
            .map_err(|e| e.to_string())?;

        if !result.status.success() {
            missing.push(pkg.clone());
        }
    }

    Ok(missing)
}

#[tauri::command]
async fn probe_model_dependencies(
    app: AppHandle,
    model_id: String,
    hf_token: Option<String>,
    venv_model_id: Option<String>,
) -> Result<ModelDependencyProbeResult, String> {
    let resolved_env = venv_model_id
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| model_id.clone());
    validate_model_id(&resolved_env)?;
    let python = resolve_python(&app, Some(&resolved_env)).await?;
    let script = r##"
import json
import re
import sys
import shlex
from huggingface_hub import HfApi, hf_hub_download

model_id = sys.argv[1]
token = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None

def call_with_token(fn, *args, **kwargs):
    if token:
        try:
            return fn(*args, token=token, **kwargs)
        except TypeError:
            return fn(*args, use_auth_token=token, **kwargs)
    return fn(*args, **kwargs)

def extract_missing_packages(text: str):
    found = set()
    stop_words = {
        "run", "pip", "pip3", "install", "python", "-m", "m"
    }

    def normalize_pkg(raw: str):
        pkg = (raw or "").strip().strip("`'\".,:;()[]{}")
        if not pkg:
            return None
        if pkg.startswith("-"):
            return None
        if pkg.lower() in stop_words:
            return None
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", pkg):
            return None
        return pkg

    pattern = re.compile(
        r"requires the following packages that were not found in your environment:\s*([^\n]+)",
        re.IGNORECASE,
    )
    for match in pattern.findall(text or ""):
        pip_match = re.search(r"(?:pip|pip3)\s+install\s+([^\n\r`]+)", match, flags=re.IGNORECASE)
        if pip_match:
            for raw in re.split(r"[,\s]+", pip_match.group(1)):
                pkg = normalize_pkg(raw)
                if pkg:
                    found.add(pkg)
            continue
        for raw in re.split(r"[,\s]+", match):
            pkg = normalize_pkg(raw)
            if pkg:
                found.add(pkg)

    for match in re.findall(r"(?:pip|pip3)\s+install\s+([^\n\r]+)", text or "", flags=re.IGNORECASE):
        for raw in re.split(r"[,\s]+", match):
            pkg = normalize_pkg(raw)
            if pkg:
                found.add(pkg)

    # Common dynamic-module failure path:
    # ModuleNotFoundError: No module named 'einops'
    for match in re.findall(r"No module named\s+['\"]?([A-Za-z0-9._-]+)['\"]?", text or "", flags=re.IGNORECASE):
        root = (match or "").split(".")[0]
        pkg = normalize_pkg(root)
        if pkg:
            found.add(pkg)

    return sorted(found)

def extract_repo_requirements(model_id: str, token: str | None):
    found = []
    seen = set()
    stop_words = {
        "pip", "pip3", "python", "install", "-m", "uv", "poetry", "conda", "mamba"
    }

    def add_req(raw: str):
        req = (raw or "").strip()
        if not req:
            return
        if req.startswith("#"):
            return
        req = req.split("#", 1)[0].strip()
        if not req:
            return
        if req.startswith("-r") or req.startswith("--"):
            return
        if "://" in req:
            return
        req = req.strip("`'\"").strip()
        if not req:
            return
        if req.lower() in stop_words:
            return
        if req.endswith("\\"):
            req = req[:-1].strip()
        if not req:
            return
        # Validate requirement-ish tokens while preserving version specifiers.
        if not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9._-]*(?:\[[^\]]+\])?(?:\s*(?:==|~=|>=|<=|!=|>|<)\s*[A-Za-z0-9*+_.-]+)?(?:\s*;[^\n]+)?",
            req,
        ):
            return
        # Keep full requirement spec (pins/ranges) for pip install.
        if req not in seen:
            seen.add(req)
            found.append(req)

    def parse_requirements_from_text(text: str):
        if not text:
            return

        # Capture explicit pip install commands from docs.
        for cmd in re.findall(
            r"(?:^|\n)\s*(?:python\s+-m\s+)?pip(?:3)?\s+install\s+([^\n\r]+)",
            text,
            flags=re.IGNORECASE,
        ):
            try:
                parts = shlex.split(cmd)
            except Exception:
                parts = re.split(r"\s+", cmd.strip())
            for tok in parts:
                if not tok or tok.startswith("-"):
                    continue
                add_req(tok)

        req_with_version_pattern = r"[A-Za-z][A-Za-z0-9._-]*(?:\[[^\]]+\])?\s*(?:==|~=|>=|<=|!=|>|<)\s*[A-Za-z0-9*+_.-]+(?:\s*;[^\n]+)?"

        # Parse fenced code blocks where dependency pins are often listed.
        for block in re.findall(r"```(?:bash|sh|shell|zsh|txt|python)?\s*\n(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
            for line in block.splitlines():
                l = line.strip()
                if not l or l.startswith("#"):
                    continue
                # Handle pip install commands explicitly so CLI flags are ignored.
                if re.search(r"^(?:python\s+-m\s+)?pip(?:3)?\s+install\s+", l, flags=re.IGNORECASE):
                    cmd = re.sub(r"^(?:python\s+-m\s+)?pip(?:3)?\s+install\s+", "", l, flags=re.IGNORECASE)
                    try:
                        parts = shlex.split(cmd)
                    except Exception:
                        parts = re.split(r"\s+", cmd.strip())
                    for tok in parts:
                        if not tok or tok.startswith("-"):
                            continue
                        add_req(tok)
                    continue
                if re.search(req_with_version_pattern, l):
                    for token in re.findall(req_with_version_pattern, l):
                        add_req(token)
                elif re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]*(?:\[[^\]]+\])?", l):
                    add_req(l)

        # Inline backticked version pins, e.g. `transformers==4.41.2`
        for token in re.findall(
            r"`([A-Za-z][A-Za-z0-9._-]*(?:\[[^\]]+\])?\s*(?:==|~=|>=|<=|!=|>|<)\s*[A-Za-z0-9*+_.-]+(?:\s*;[^\n`]+)?)`",
            text,
            flags=re.IGNORECASE,
        ):
            add_req(token)

    try:
        api = HfApi(token=token or None)
        files = api.list_repo_files(repo_id=model_id, repo_type="model")
        lower_to_real = {f.lower(): f for f in files}
        candidates = []
        for key in [
            "requirements.txt",
            "requirement.txt",
            "requirements/requirements.txt",
            "requirements/base.txt",
        ]:
            if key in lower_to_real:
                candidates.append(lower_to_real[key])
        if not candidates:
            for f in files:
                lf = f.lower()
                if lf.endswith(".txt") and "requirements" in lf:
                    candidates.append(f)
                    if len(candidates) >= 3:
                        break

        for filename in candidates:
            try:
                path = hf_hub_download(repo_id=model_id, filename=filename, token=token or None)
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        add_req(line)
            except Exception:
                continue

        # Parse README guidance for pinned versions when requirements files are absent/incomplete.
        readme_name = lower_to_real.get("readme.md")
        if readme_name:
            try:
                readme_path = hf_hub_download(repo_id=model_id, filename=readme_name, token=token or None)
                with open(readme_path, "r", encoding="utf-8") as rh:
                    parse_requirements_from_text(rh.read())
            except Exception:
                pass
    except Exception:
        return []

    return found

compatibility_error = None
required = extract_repo_requirements(model_id, token)
try:
    from transformers import AutoConfig, AutoModel
except Exception as import_err:
    err_text = str(import_err)
    missing = extract_missing_packages(err_text)
    if not missing:
        missing = ["transformers"]
    print("HB_PROBE_JSON:" + json.dumps({
        "missingPackages": missing,
        "requiredPackages": required,
        "compatibilityError": None,
    }), flush=True)
    sys.exit(0)

try:
    cfg = call_with_token(AutoConfig.from_pretrained, model_id, trust_remote_code=True)
    call_with_token(AutoModel.from_config, cfg, trust_remote_code=True)
    missing = []
except Exception as err:
    err_text = str(err)
    missing = extract_missing_packages(err_text)
    lower = err_text.lower()
    if not missing and "cannot import name" in lower and "transformers.models" in lower:
        transformer_pins = [r for r in required if r.lower().startswith("transformers")]
        if transformer_pins:
            hint = f" Install model-declared requirement: {transformer_pins[0]}"
        else:
            hint = " Install model-declared requirements (if present) or try pinning transformers to a model-compatible version."
        compatibility_error = (
            "This model's custom code is incompatible with your installed transformers version. "
            f"{hint}"
        )
    elif not missing and "sigalrm" in lower and "trust_remote_code" in lower:
        compatibility_error = (
            "Windows trust_remote_code prompt failed. Regenerate code and ensure trust_remote_code=True is used."
        )

print("HB_PROBE_JSON:" + json.dumps({
    "missingPackages": missing,
    "requiredPackages": required,
    "compatibilityError": compatibility_error,
}), flush=True)
"##;

    let script_path = std::env::temp_dir().join("huggingbox_probe_deps.py");
    std::fs::write(&script_path, script).map_err(|e| e.to_string())?;

    let token = hf_token.unwrap_or_default();
    let mut probe_cmd = tokio::process::Command::new(&python);
    apply_no_window_tokio(&mut probe_cmd);
    let output = probe_cmd
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .arg(script_path.to_string_lossy().to_string())
        .arg(model_id)
        .arg(token)
        .output()
        .await
        .map_err(|e| format!("Failed to run dependency probe: {}", e))?;

    let _ = std::fs::remove_file(&script_path);
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if let Some(raw) = line.strip_prefix("HB_PROBE_JSON:") {
            let parsed = serde_json::from_str::<ModelDependencyProbeResult>(raw)
                .map_err(|e| format!("Failed to parse dependency probe result: {}", e))?;
            return Ok(parsed);
        }
    }

    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if output.status.success() {
        Ok(ModelDependencyProbeResult {
            missing_packages: vec![],
            required_packages: vec![],
            compatibility_error: None,
        })
    } else {
        Err(format!("Dependency probe failed: {}", stderr))
    }
}

fn requirement_name(requirement: &str) -> String {
    let trimmed = requirement.trim().to_lowercase();
    if trimmed.starts_with("git+https://github.com/huggingface/transformers") {
        return "transformers".to_string();
    }

    let mut end = requirement.len();
    for (idx, ch) in requirement.char_indices() {
        if ['<', '>', '=', '!', '~', '[', ';', ' '].contains(&ch) {
            end = idx;
            break;
        }
    }
    requirement[..end].trim().to_lowercase()
}

fn import_name_for_requirement(requirement: &str) -> String {
    let name = requirement_name(requirement);
    match name.as_str() {
        "pillow" => "PIL".to_string(),
        "scikit-learn" => "sklearn".to_string(),
        "opencv-python" | "opencv-python-headless" => "cv2".to_string(),
        _ => name.replace('-', "_"),
    }
}

fn extract_shell_marker(output: &str, marker_prefix: &str) -> (String, Option<String>) {
    let mut cleaned = Vec::new();
    let mut marker: Option<String> = None;
    for line in output.lines() {
        if let Some(rest) = line.strip_prefix(marker_prefix) {
            marker = Some(rest.trim().to_string());
            continue;
        }
        cleaned.push(line);
    }
    let mut joined = cleaned.join("\n");
    if output.ends_with('\n') {
        joined.push('\n');
    }
    (joined, marker)
}

fn normalize_requirement_for_install(requirement: &str) -> String {
    if requirement_name(requirement) == "transformers" {
        return HF_TRANSFORMERS_GIT_URL.to_string();
    }
    requirement.to_string()
}

async fn run_pip_install_with_progress(
    app: &AppHandle,
    python: &str,
    args: Vec<String>,
) -> Result<(), String> {
    let mut pip_install_cmd = tokio::process::Command::new(python);
    apply_no_window_tokio(&mut pip_install_cmd);
    let mut child = pip_install_cmd
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| e.to_string())?;

    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let app2 = app.clone();
    let app_out = app.clone();

    let t1 = tokio::spawn(async move {
        let mut lines = tokio::io::BufReader::new(stdout).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let _ = app_out.emit("install-progress", StreamPayload { text: line + "\n" });
        }
    });

    let t2 = tokio::spawn(async move {
        let mut lines = tokio::io::BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let _ = app2.emit("install-progress", StreamPayload { text: line + "\n" });
        }
    });

    let status = child.wait().await.map_err(|e| e.to_string())?;
    let _ = tokio::join!(t1, t2);
    if status.success() {
        Ok(())
    } else {
        Err("Package installation failed".to_string())
    }
}

#[tauri::command]
async fn install_packages(
    app: AppHandle,
    packages: Vec<String>,
    model_id: Option<String>,
    venv_model_id: Option<String>,
) -> Result<(), String> {
    let execution_env_model = venv_model_id
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().to_string())
        .or_else(|| model_id.as_deref().map(|s| s.trim().to_string()));
    if let Some(ref env_model) = execution_env_model {
        validate_model_id(env_model)?;
    }

    let python = resolve_python(&app, execution_env_model.as_deref()).await?;
    let mut torch_family = Vec::new();
    let mut other_packages = Vec::new();
    let mut flash_attn_packages = Vec::new();
    let mut requested_torch_name = false;
    let mut requested_torchvision_or_audio = false;
    let mut requested_flash_attn = false;
    let mut requested_transformers = false;
    for pkg in packages {
        let name = requirement_name(&pkg);
        let normalized_pkg = normalize_requirement_for_install(&pkg);
        if name == "transformers" {
            requested_transformers = true;
        }
        if name == "flash-attn" || name == "flash_attn" {
            requested_flash_attn = true;
            flash_attn_packages.push(normalized_pkg);
            continue;
        }
        if name == "torch" || name == "torchvision" || name == "torchaudio" {
            if name == "torch" {
                requested_torch_name = true;
            }
            if name == "torchvision" || name == "torchaudio" {
                requested_torchvision_or_audio = true;
            }
            torch_family.push(normalized_pkg);
        } else {
            other_packages.push(normalized_pkg);
        }
    }

    if requested_transformers {
        let mut dropped_conflicts: Vec<String> = Vec::new();
        other_packages.retain(|pkg| {
            let name = requirement_name(pkg);
            let is_conflict = name == "tokenizers" || name == "huggingface-hub";
            if is_conflict {
                dropped_conflicts.push(pkg.clone());
            }
            !is_conflict
        });

        if !dropped_conflicts.is_empty() {
            let _ = app.emit("install-progress", StreamPayload {
                text: format!(
                    "[HuggingBox] Dropping conflicting pins with HF transformers source: {}\n",
                    dropped_conflicts.join(", ")
                ),
            });
        }
    }

    // torchvision/torchaudio wheels must match the torch build.
    // If callers request torchvision/torchaudio alone, include torch to keep versions aligned.
    if requested_torchvision_or_audio && !requested_torch_name {
        torch_family.insert(0, "torch".to_string());
    }
    // If a model pins torch (and uses flash-attn), force a torchvision reinstall
    // in the same transaction so torch/torchvision versions stay aligned.
    if requested_torch_name && !requested_torchvision_or_audio && requested_flash_attn {
        torch_family.push("torchvision".to_string());
    }

    if requested_transformers {
        let _ = app.emit("install-progress", StreamPayload {
            text: format!(
                "[HuggingBox] Preferring Hugging Face transformers source: {}\n",
                HF_TRANSFORMERS_GIT_URL
            ),
        });
    }

    let has_cuda_gpu = !detect_cuda_gpus().is_empty();
    if has_cuda_gpu && !torch_family.is_empty() {
        let _ = app.emit("install-progress", StreamPayload {
            text: "[HuggingBox] CUDA GPU detected. Preferring CUDA PyTorch wheels.\n".to_string(),
        });
        let mut cuda_args = vec![
            "-m".to_string(),
            "pip".to_string(),
            "install".to_string(),
            "--index-url".to_string(),
            "https://download.pytorch.org/whl/cu124".to_string(),
        ];
        cuda_args.extend(torch_family.clone());

        if run_pip_install_with_progress(&app, &python, cuda_args).await.is_err() {
            let _ = app.emit("install-progress", StreamPayload {
                text: "[HuggingBox] CUDA wheel install failed. Falling back to default PyPI packages.\n".to_string(),
            });
            let mut fallback_args = vec!["-m".to_string(), "pip".to_string(), "install".to_string()];
            fallback_args.extend(torch_family);
            run_pip_install_with_progress(&app, &python, fallback_args).await?;
        }
    } else if !torch_family.is_empty() {
        let mut args = vec!["-m".to_string(), "pip".to_string(), "install".to_string()];
        args.extend(torch_family);
        run_pip_install_with_progress(&app, &python, args).await?;
    }

    if !other_packages.is_empty() {
        let mut args = vec!["-m".to_string(), "pip".to_string(), "install".to_string()];
        args.extend(other_packages);
        run_pip_install_with_progress(&app, &python, args).await?;
    }

    if !flash_attn_packages.is_empty() {
        if cfg!(target_os = "windows") {
            let _ = app.emit("install-progress", StreamPayload {
                text: "[HuggingBox] Skipping optional flash-attn on Windows (build support is limited). Continuing without it.\n".to_string(),
            });
        } else {
            // Ensure common build tooling exists before attempting flash-attn.
            let bootstrap_args = vec![
                "-m".to_string(),
                "pip".to_string(),
                "install".to_string(),
                "--upgrade".to_string(),
                "wheel".to_string(),
                "setuptools".to_string(),
            ];
            let _ = run_pip_install_with_progress(&app, &python, bootstrap_args).await;

            let _ = app.emit("install-progress", StreamPayload {
                text: "[HuggingBox] Installing flash-attn with --no-build-isolation.\n".to_string(),
            });
            let mut args = vec![
                "-m".to_string(),
                "pip".to_string(),
                "install".to_string(),
                "--no-build-isolation".to_string(),
            ];
            args.extend(flash_attn_packages);
            if let Err(e) = run_pip_install_with_progress(&app, &python, args).await {
                let _ = app.emit("install-progress", StreamPayload {
                    text: format!(
                        "[HuggingBox] Optional dependency flash-attn failed to install: {}. Continuing without it.\n",
                        e
                    ),
                });
            }
        }
    }

    Ok(())
}

#[tauri::command]
async fn run_model_shell_command(
    app: AppHandle,
    model_id: String,
    venv_model_id: Option<String>,
    command: String,
    model_storage_path: Option<String>,
    env_storage_path: Option<String>,
    cwd: Option<String>,
    hf_token: Option<String>,
) -> Result<ShellCommandResult, String> {
    validate_model_id(&model_id)?;
    let trimmed = command.trim();
    if trimmed.is_empty() {
        return Err("Command is empty.".to_string());
    }

    let execution_env_model = venv_model_id
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| model_id.clone());
    validate_model_id(&execution_env_model)?;

    let venv_root = effective_venv_root(&app, env_storage_path.as_deref())?;
    let venv_dir = model_venv_dir_from_root(&venv_root, &execution_env_model)?;
    let _ = ensure_venv_python_at_dir(&app, &venv_dir).await?;
    let workspace_root = model_storage_path
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| resolve_model_workspace_root(&model_id, s))
        .transpose()?;

    let venv_python = venv_python_path(&venv_dir);
    let venv_bin = venv_python
        .parent()
        .ok_or("Could not resolve venv scripts directory.")?
        .to_path_buf();

    let command_cwd = cwd
        .as_deref()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .filter(|p| p.exists() && p.is_dir())
        .unwrap_or_else(|| workspace_root.unwrap_or_else(|| venv_dir.clone()));

    let current_path = std::env::var("PATH").unwrap_or_default();
    let sep = if cfg!(target_os = "windows") { ";" } else { ":" };
    let path_with_venv = format!("{}{}{}", venv_bin.to_string_lossy(), sep, current_path);
    let cwd_marker = "__HB_CWD__:";

    #[cfg(target_os = "windows")]
    let output = {
        let script = "$cmd = $env:HB_SHELL_COMMAND; \
function global:python { & $env:HB_VENV_PYTHON @Args }; \
function global:pip { & $env:HB_VENV_PYTHON -m pip @Args }; \
function global:pip3 { & $env:HB_VENV_PYTHON -m pip @Args }; \
Invoke-Expression $cmd; \
$hbExit = $LASTEXITCODE; \
if ($null -eq $hbExit) { $hbExit = 0 }; \
Write-Output ('__HB_CWD__:' + (Get-Location).Path); \
exit $hbExit";
        {
            let mut shell_cmd = tokio::process::Command::new("powershell");
            apply_no_window_tokio(&mut shell_cmd);
            shell_cmd
                .args(["-NoProfile", "-Command", script])
                .env("PATH", &path_with_venv)
                .env("VIRTUAL_ENV", venv_dir.to_string_lossy().to_string())
                .env("HB_VENV_PYTHON", venv_python.to_string_lossy().to_string())
                .env("HB_SHELL_COMMAND", trimmed)
                .env("HF_TOKEN", hf_token.unwrap_or_default())
                .current_dir(&command_cwd)
                .output()
                .await
                .map_err(|e| format!("Failed to execute shell command: {}", e))?
        }
    };

    #[cfg(not(target_os = "windows"))]
    let output = {
        let script = r#"python() { "$HB_VENV_PYTHON" "$@"; }
pip() { "$HB_VENV_PYTHON" -m pip "$@"; }
pip3() { "$HB_VENV_PYTHON" -m pip "$@"; }
cmd="$HB_SHELL_COMMAND"
eval "$cmd"
hb_ec=$?
printf "__HB_CWD__:%s\n" "$PWD"
exit $hb_ec"#;
        tokio::process::Command::new("bash")
            .args(["-lc", script])
            .env("PATH", &path_with_venv)
            .env("VIRTUAL_ENV", venv_dir.to_string_lossy().to_string())
            .env("HB_VENV_PYTHON", venv_python.to_string_lossy().to_string())
            .env("HB_SHELL_COMMAND", trimmed)
            .env("HF_TOKEN", hf_token.unwrap_or_default())
            .current_dir(&command_cwd)
            .output()
            .await
            .map_err(|e| format!("Failed to execute shell command: {}", e))?
    };

    let raw_stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let raw_stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let (stdout, extracted_cwd) = extract_shell_marker(&raw_stdout, cwd_marker);
    let exit_code = output.status.code().unwrap_or(-1);
    let resolved_cwd = extracted_cwd.unwrap_or_else(|| command_cwd.to_string_lossy().to_string());

    Ok(ShellCommandResult {
        stdout,
        stderr: raw_stderr,
        exit_code,
        cwd: resolved_cwd,
    })
}

#[tauri::command]
fn get_cached_code(
    app: AppHandle,
    cache_key: String,
) -> Result<Option<CodeCacheRecord>, String> {
    let db_path = get_db_path(&app)?;
    let conn = Connection::open(db_path).map_err(|e| format!("DB open failed: {}", e))?;
    init_db(&conn)?;

    let mut stmt = conn
        .prepare("SELECT code FROM code_cache WHERE cache_key = ?1")
        .map_err(|e| format!("DB query prepare failed: {}", e))?;

    let mut rows = stmt
        .query(params![cache_key])
        .map_err(|e| format!("DB query failed: {}", e))?;

    if let Some(row) = rows.next().map_err(|e| format!("DB row read failed: {}", e))? {
        let code: String = row.get(0).map_err(|e| format!("DB value read failed: {}", e))?;
        Ok(Some(CodeCacheRecord { code }))
    } else {
        Ok(None)
    }
}

#[tauri::command]
fn upsert_cached_code(
    app: AppHandle,
    cache_key: String,
    model_id: String,
    code: String,
) -> Result<(), String> {
    let db_path = get_db_path(&app)?;
    let conn = Connection::open(db_path).map_err(|e| format!("DB open failed: {}", e))?;
    init_db(&conn)?;

    conn.execute(
        "INSERT INTO code_cache(cache_key, model_id, code, created_at)
         VALUES(?1, ?2, ?3, datetime('now'))
         ON CONFLICT(cache_key)
         DO UPDATE SET
           model_id = excluded.model_id,
           code = excluded.code,
           created_at = datetime('now')",
        params![cache_key, model_id, code],
    )
    .map_err(|e| format!("DB upsert failed: {}", e))?;

    Ok(())
}

// ─── Entry point ─────────────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Best-effort startup cleanup for stale temp scripts.
    // Any active process from a prior session would have exited with the app.
    // This keeps app_data/temp tidy.
    // Failures here should not block app startup.
    // Intentionally ignored.
    // cleanup_old_temp_scripts requires app handle, so we call it via setup below.
    tauri::Builder::default()
        .manage(ExecutionHandle::new())
        .setup(|app| {
            cleanup_old_temp_scripts(app.handle());
            Ok(())
        })
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            load_app_settings,
            save_app_settings,
            get_system_info,
            list_gpus,
            bootstrap_python_environment,
            detect_python,
            generate_python_code_local,
            generate_code_with_claude,
            run_python_code,
            run_model_shell_command,
            cancel_execution,
            is_model_downloaded,
            download_model,
            cancel_download,
            list_downloaded_models,
            delete_downloaded_model,
            list_model_workspace_entries,
            read_model_workspace_file,
            read_binary_file,
            write_model_workspace_file,
            create_model_workspace_file,
            create_model_workspace_directory,
            list_model_environments,
            get_model_environment_size,
            delete_model_environment,
            check_packages,
            probe_model_dependencies,
            install_packages,
            get_cached_code,
            upsert_cached_code,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
