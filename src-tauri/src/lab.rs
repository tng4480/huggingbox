use serde_json::{json, Value};
use std::{
    path::PathBuf,
    process::Stdio,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::{SystemTime, UNIX_EPOCH},
};
use tauri::{AppHandle, Manager, State};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

#[derive(Default)]
pub struct LabState {
    active: Arc<Mutex<Option<(String, u32)>>>,
    cancelled: Arc<AtomicBool>,
    draft: Mutex<()>,
}
impl LabState {
    pub fn shutdown(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        if let Some((_, pid)) = self.active.lock().unwrap().take() {
            if pid != 0 {
                super::kill_pid(pid, true);
            }
        }
    }
}
fn root(app: &AppHandle) -> Result<PathBuf, String> {
    let p = app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?
        .join("lab/runs");
    std::fs::create_dir_all(&p).map_err(|e| e.to_string())?;
    Ok(p)
}
fn run_path(app: &AppHandle, id: &str) -> Result<PathBuf, String> {
    if id.is_empty() || !id.chars().all(|c| c.is_ascii_digit() || c == '-') {
        return Err("Invalid Lab run ID".into());
    }
    Ok(root(app)?.join(id))
}

#[tauri::command]
pub fn lab_read_dataset(path: String) -> Result<String, String> {
    let p = PathBuf::from(path);
    let size = std::fs::metadata(&p).map_err(|e| e.to_string())?.len();
    if size > 8_000_000 {
        return Err("Dataset must be at most 8 MB".into());
    }
    std::fs::read_to_string(p).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn lab_runs(app: AppHandle, state: State<'_, LabState>) -> Result<Vec<Value>, String> {
    let active = state.active.lock().unwrap().as_ref().map(|v| v.0.clone());
    let mut paths = std::fs::read_dir(root(&app)?)
        .map_err(|e| e.to_string())?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect::<Vec<_>>();
    paths.sort();
    paths.reverse();
    Ok(paths
        .into_iter()
        .take(50)
        .filter_map(|p| {
            let mut value: Value =
                serde_json::from_str(&std::fs::read_to_string(p.join("status.json")).ok()?).ok()?;
            if value["status"] == "running" && active.as_deref() != value["id"].as_str() {
                value["status"] = json!("interrupted");
            }
            value["checkpointAvailable"] = json!(
                p.join("adapter/adapters.safetensors").is_file()
                    || p.join("adapter/adapter_model.safetensors").is_file()
            );
            Some(value)
        })
        .collect())
}

#[tauri::command]
pub fn lab_run_detail(app: AppHandle, id: String) -> Result<Value, String> {
    use std::io::{Read, Seek, SeekFrom};
    let p = run_path(&app, &id)?;
    let read = |name: &str| -> Value {
        std::fs::read_to_string(p.join(name))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(Value::Null)
    };
    let mut log = String::new();
    if let Ok(mut file) = std::fs::File::open(p.join("log.txt")) {
        let size = file.metadata().map_err(|e| e.to_string())?.len();
        file.seek(SeekFrom::Start(size.saturating_sub(64000)))
            .map_err(|e| e.to_string())?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
        log = String::from_utf8_lossy(&bytes).to_string();
    }
    Ok(
        json!({"result":read("result.json"),"error":read("error.json"),"manifest":read("manifest.json"),"log":log,"path":p}),
    )
}

#[tauri::command]
pub fn lab_cancel(state: State<'_, LabState>) -> Result<(), String> {
    state.cancelled.store(true, Ordering::SeqCst);
    if let Some((_, pid)) = state.active.lock().unwrap().as_ref() {
        if *pid != 0 {
            super::kill_pid(*pid, true);
        }
    }
    Ok(())
}

#[tauri::command]
pub async fn lab_start(
    app: AppHandle,
    state: State<'_, LabState>,
    handle: State<'_, super::ExecutionHandle>,
    mut config: Value,
) -> Result<String, String> {
    let action = config["action"]
        .as_str()
        .ok_or("Missing action")?
        .to_string();
    if ![
        "validate", "setup", "train", "evaluate", "inspect", "fit_lens",
    ]
    .contains(&action.as_str())
    {
        return Err("Unsupported Lab action".into());
    }
    if config.to_string().len() > 8_500_000 {
        return Err("Request is too large".into());
    }
    let backend = config["backend"].as_str().unwrap_or("torch").to_string();
    if !["mlx", "torch"].contains(&backend.as_str()) {
        return Err("Unknown backend".into());
    }
    if backend == "mlx" && !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return Err("MLX requires Apple Silicon. Choose PyTorch on this machine.".into());
    }
    if ["inspect", "fit_lens"].contains(&action.as_str()) && backend != "torch" {
        return Err(
            "Inspection currently requires a Transformers model and PyTorch environment.".into(),
        );
    }
    let mut worker = handle.worker.lock().await;
    if handle.execution_pid.lock().unwrap().is_some()
        || handle.download_pid.lock().unwrap().is_some()
        || handle.preparation_pid.lock().unwrap().is_some()
    {
        return Err("Finish the current model job before starting Lab work.".into());
    }
    if handle.lab_busy.swap(true, Ordering::SeqCst) {
        return Err("A Lab job is already running.".into());
    }
    if let Some(old) = worker.take() {
        super::kill_pid(old.pid, true);
    }
    drop(worker);
    state.cancelled.store(false, Ordering::SeqCst);
    let busy = handle.lab_busy.clone();
    let active = state.active.clone();
    let cancelled = state.cancelled.clone();
    let id = format!(
        "{}-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis(),
        std::process::id()
    );
    *active.lock().unwrap() = Some((id.clone(), 0));
    let started=async{
        let p=run_path(&app,&id)?;std::fs::create_dir(&p).map_err(|e|e.to_string())?;
        std::fs::write(p.join("status.json"),json!({"id":id,"action":action,"backend":backend,"task":config["task"].as_str().unwrap_or("text-generation"),"status":"running","path":p}).to_string()).map_err(|e|e.to_string())?;
        if let Some(source)=config["run_id"].as_str(){config["run_path"]=json!(run_path(&app,source)?);}
        if let Some(source)=config["resume_id"].as_str().filter(|s|!s.is_empty()){config["resume_path"]=json!(run_path(&app,source)?);}
        if let Some(source)=config["inspection_run_id"].as_str().filter(|s|!s.is_empty()){config["inspection_run_path"]=json!(run_path(&app,source)?);}
        let override_path=config["python_path"].as_str().filter(|s|!s.trim().is_empty());
        let python=if action=="validate"{super::find_python(&app).await?}else if let Some(path)=override_path{
            if action=="setup"{return Err("Setup installs only into HuggingBox's isolated Lab environment. Clear the Python override first.".into())}
            if !PathBuf::from(path).is_absolute()||!PathBuf::from(path).is_file(){return Err("Choose an existing Python executable.".into())}path.to_string()
        }else{let env=app.path().app_data_dir().map_err(|e|e.to_string())?.join(format!("lab/env-{backend}"));super::ensure_venv_python_at_dir(&app,&env).await?};
        let mut command=tokio::process::Command::new(&python);super::apply_no_window_tokio(&mut command);
        if let Some(parent)=super::hf_auto_runner_parent_dir(&app){command.env("PYTHONPATH",parent);}
        command.env("PYTHONUNBUFFERED","1").env("PYTHONUTF8","1").env("TOKENIZERS_PARALLELISM","false").env("WANDB_DISABLED","true");
        for key in ["HF_TOKEN","ANTHROPIC_API_KEY","OPENAI_API_KEY"]{command.env_remove(key);}
        if action=="setup"{
            command.args(["-m","pip","install"]);
            if backend=="mlx"{command.args(["mlx-lm[train]>=0.32,<0.33"]);}else{command.args(["torch>=2.6,<3","transformers>=5.5,<6","peft>=0.17,<1","pillow>=10,<13","jlens @ git+https://github.com/anthropics/jacobian-lens.git@581d398613e5602a5af361e1c34d3a92ea82ba8e"]);}
        }else{
            command.env("HF_HUB_OFFLINE","1").env("TRANSFORMERS_OFFLINE","1");
            std::fs::write(p.join("config.json"),config.to_string()).map_err(|e|e.to_string())?;
            command.args(["-u","-m","hf_auto_runner.lab"]).arg(p.join("config.json"));
        }
        if cancelled.load(Ordering::SeqCst){return Err("Lab start cancelled".into())}
        let mut child=command.current_dir(&p).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true).spawn().map_err(|e|e.to_string())?;
        let pid=child.id().ok_or("No process ID")?;*active.lock().unwrap()=Some((id.clone(),pid));
        if cancelled.load(Ordering::SeqCst){super::kill_pid(pid,true);}
        let stdout=child.stdout.take().unwrap();let stderr=child.stderr.take().unwrap();
        let status=json!({"id":id,"action":action,"backend":backend,"task":config["task"].as_str().unwrap_or("text-generation"),"status":"running","path":p});
        std::fs::write(p.join("status.json"),status.to_string()).map_err(|e|e.to_string())?;
        let log=Arc::new(tokio::sync::Mutex::new(tokio::fs::File::create(p.join("log.txt")).await.map_err(|e|e.to_string())?));
        let read_log=log.clone();let read_err=log;
        tokio::spawn(async move{
            let out=tokio::spawn(async move{let mut lines=tokio::io::BufReader::new(stdout).lines();while let Ok(Some(line))=lines.next_line().await{let _=read_log.lock().await.write_all(format!("{line}\n").as_bytes()).await;}});
            let err=tokio::spawn(async move{let mut lines=tokio::io::BufReader::new(stderr).lines();while let Ok(Some(line))=lines.next_line().await{let _=read_err.lock().await.write_all(format!("{line}\n").as_bytes()).await;}});
            let result=child.wait().await;let _=out.await;let _=err.await;
            let mut status=status;status["status"]=json!(if cancelled.load(Ordering::SeqCst){"cancelled"}else if result.is_ok_and(|s|s.success()){ "completed" }else{"failed"});
            let _=std::fs::write(p.join("status.json"),status.to_string());
            *active.lock().unwrap()=None;busy.store(false,Ordering::SeqCst);
        });
        Ok(id.clone())
    }.await;
    if let Err(ref error) = started {
        if let Ok(p) = run_path(&app, &id) {
            let _=std::fs::write(p.join("status.json"),json!({"id":id,"action":action,"backend":backend,"task":config["task"].as_str().unwrap_or("text-generation"),"status":if state.cancelled.load(Ordering::SeqCst){"cancelled"}else{"failed"},"path":p}).to_string());
            let _ = std::fs::write(p.join("error.json"), json!({"message":error}).to_string());
        }
        *state.active.lock().unwrap() = None;
        handle.lab_busy.store(false, Ordering::SeqCst);
    }
    started
}

#[tauri::command]
pub fn lab_dataset_draft(
    app: AppHandle,
    state: State<'_, LabState>,
    text: Option<String>,
    task: Option<String>,
) -> Result<String, String> {
    let _guard = state.draft.lock().unwrap();
    let task = task.unwrap_or_else(|| "text-generation".into());
    if ![
        "text-generation",
        "text-classification",
        "image-classification",
    ]
    .contains(&task.as_str())
    {
        return Err("Unsupported task".into());
    }
    let path =
        root(&app)?
            .parent()
            .ok_or("Lab directory missing")?
            .join(if task == "text-generation" {
                "dataset-draft.jsonl".to_string()
            } else {
                format!("dataset-{task}.jsonl")
            });
    if let Some(text) = text {
        if text.len() > 8_000_000 {
            return Err("Dataset must be at most 8 MB".into());
        }
        let temporary = path.with_extension("tmp");
        std::fs::write(&temporary, &text).map_err(|e| e.to_string())?;
        std::fs::rename(temporary, &path).map_err(|e| e.to_string())?;
        return Ok(text);
    }
    if path.exists() {
        std::fs::read_to_string(path).map_err(|e| e.to_string())
    } else {
        Ok(String::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    #[test]
    fn shutdown_terminates_active_process() {
        let mut child = std::process::Command::new("/bin/sleep")
            .arg("30")
            .spawn()
            .unwrap();
        let state = LabState::default();
        *state.active.lock().unwrap() = Some(("running".into(), child.id()));
        let started = std::time::Instant::now();
        state.shutdown();
        assert!(!child.wait().unwrap().success());
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
    }
    #[test]
    fn shutdown_cancels_even_before_process_spawn() {
        let state = LabState::default();
        *state.active.lock().unwrap() = Some(("pending".into(), 0));
        state.shutdown();
        assert!(state.cancelled.load(Ordering::SeqCst));
        assert!(state.active.lock().unwrap().is_none());
    }
}

#[tauri::command]
pub fn lab_import_image_folder(path: String) -> Result<String, String> {
    let root = PathBuf::from(path)
        .canonicalize()
        .map_err(|e| e.to_string())?;
    let mut rows = Vec::new();
    let mut classes = std::fs::read_dir(&root)
        .map_err(|e| e.to_string())?
        .flatten()
        .collect::<Vec<_>>();
    classes.sort_by_key(|e| e.file_name());
    for class in classes {
        if !class.file_type().map_err(|e| e.to_string())?.is_dir() {
            continue;
        }
        let label = class.file_name().to_string_lossy().to_string();
        if label.starts_with('.') {
            continue;
        }
        let mut files = std::fs::read_dir(class.path())
            .map_err(|e| e.to_string())?
            .flatten()
            .collect::<Vec<_>>();
        files.sort_by_key(|e| e.file_name());
        for file in files {
            let p = file.path();
            let ext = p
                .extension()
                .unwrap_or_default()
                .to_string_lossy()
                .to_lowercase();
            if p.is_file() && ["jpg", "jpeg", "png", "webp", "bmp"].contains(&ext.as_str()) {
                rows.push(json!({"image":p,"label":label}).to_string());
                if rows.len() > 10000 {
                    return Err("Maximum 10,000 images".into());
                }
            }
        }
    }
    if rows.is_empty() {
        return Err("Choose a folder containing one subfolder per class, with images directly inside each class folder.".into());
    }
    Ok(rows.join("\n"))
}
