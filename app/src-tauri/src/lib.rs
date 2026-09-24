// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::Manager;

struct SidecarState(Mutex<Option<Child>>);

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

/// Dev-time only: assumes the source tree layout (`../../engine` relative to
/// src-tauri's cwd). Phase 6 packaging replaces this with a bundled sidecar
/// binary instead of shelling out to `uv run` against source.
fn engine_dir() -> PathBuf {
    std::env::current_dir()
        .expect("current dir")
        .join("..")
        .join("..")
        .join("engine")
}

/// Kill a process and its whole descendant tree.
///
/// `Child::kill()` alone only terminates that one process. On Windows, `uv
/// run` doesn't replace its own process image the way Unix `exec` does --
/// it spawns the actual `python.exe` as a *child* of `uv.exe` -- so killing
/// just the `uv` handle we hold leaves that python process (our FastAPI
/// server) orphaned and still bound to the port. `taskkill /T` kills the
/// whole tree instead.
fn kill_process_tree(pid: u32) {
    #[cfg(target_os = "windows")]
    {
        let _ = Command::new("taskkill")
            .args(["/F", "/T", "/PID", &pid.to_string()])
            .output();
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = Command::new("kill").args(["-9", &pid.to_string()]).output();
    }
}

/// Launch `syncaudio serve` (the FastAPI sidecar) as a child process. A
/// failure here (e.g. `uv` missing) is logged, not fatal: the GUI window
/// still opens, it just can't reach the engine until the sidecar is fixed
/// and the app restarted.
fn spawn_sidecar() -> Option<Child> {
    let dir = engine_dir();
    match Command::new("uv")
        .args(["run", "syncaudio", "serve", "--port", "8756"])
        .current_dir(&dir)
        .spawn()
    {
        Ok(child) => {
            println!("[sidecar] démarré (uv run syncaudio serve) dans {:?}", dir);
            Some(child)
        }
        Err(err) => {
            eprintln!("[sidecar] échec du démarrage dans {:?} : {}", dir, err);
            None
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(SidecarState(Mutex::new(spawn_sidecar())))
        .invoke_handler(tauri::generate_handler![greet])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            if let tauri::RunEvent::ExitRequested { .. } = event {
                let state = app_handle.state::<SidecarState>();
                let mut guard = state.0.lock().unwrap();
                if let Some(child) = guard.take() {
                    kill_process_tree(child.id());
                }
            }
        });
}
