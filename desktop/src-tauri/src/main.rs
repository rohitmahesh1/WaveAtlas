#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::Deserialize;
use std::{
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
    time::{Duration, Instant},
};
use tauri::{Manager, WebviewUrl, WebviewWindowBuilder, WindowEvent};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons};

#[derive(Clone, Deserialize)]
struct Ready {
    origin: String,
    url: String,
    token: String,
}

struct Backend {
    child: Mutex<Option<Child>>,
    ready: Mutex<Option<Ready>>,
    closing: AtomicBool,
    workspace: PathBuf,
}

impl Drop for Backend {
    fn drop(&mut self) {
        if let Ok(child) = self.child.get_mut() {
            if let Some(child) = child.as_mut() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

fn show_error(app: &tauri::AppHandle, message: &str) {
    if let Some(window) = app.get_webview_window("loading") {
        let _ = window.eval(&format!(
            "document.getElementById('message').textContent = {}",
            serde_json::to_string(message).unwrap()
        ));
        let _ = window.show();
    } else {
        app.dialog()
            .message(message)
            .title("WaveAtlas")
            .show(|_| {});
    }
}

fn stop_backend(app: &tauri::AppHandle) {
    let backend = app.state::<Backend>();
    // Keep the child alive briefly for cooperative cancellation and checkpointing.
    if let Ok(mut guard) = backend.child.lock() {
        if let Some(child) = guard.as_mut() {
            if let Some(stdin) = child.stdin.as_mut() {
                let _ = stdin.write_all(b"{\"command\":\"shutdown\"}\n");
                let _ = stdin.flush();
            }
        }
    }
    let deadline = Instant::now() + Duration::from_secs(17);
    loop {
        let mut guard = backend.child.lock().unwrap();
        let Some(child) = guard.as_mut() else { break };
        if child.try_wait().ok().flatten().is_some() {
            break;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            break;
        }
        drop(guard);
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn close_requested(app: tauri::AppHandle) {
    let backend = app.state::<Backend>();
    if backend.closing.swap(true, Ordering::SeqCst) {
        return;
    }
    std::thread::spawn(move || {
        let ready = app.state::<Backend>().ready.lock().unwrap().clone();
        let active = ready
            .and_then(|ready| {
                reqwest::blocking::Client::builder()
                    .timeout(Duration::from_secs(3))
                    .build()
                    .ok()?
                    .get(format!("{}/api/desktop/status", ready.origin))
                    .bearer_auth(ready.token)
                    .send()
                    .ok()?
                    .error_for_status()
                    .ok()?
                    .json::<serde_json::Value>()
                    .ok()
            })
            .map(|status| !status["active_job"].is_null());
        // If a running backend cannot answer, ask rather than assume work is idle.
        let running = app
            .state::<Backend>()
            .child
            .lock()
            .unwrap()
            .as_mut()
            .map(|child| child.try_wait().ok().flatten().is_none())
            .unwrap_or(false);
        if active.unwrap_or(running) {
            let stop = app.dialog().message("An analysis may still be running. Stop it and exit? Available checkpoints will be kept for your next launch.")
                .title("Close WaveAtlas")
                .buttons(MessageDialogButtons::OkCancelCustom("Stop and exit".into(), "Keep working".into()))
                .blocking_show();
            if !stop {
                app.state::<Backend>()
                    .closing
                    .store(false, Ordering::SeqCst);
                return;
            }
        }
        if let Some(window) = app.get_webview_window("main") {
            let _ = window.set_title("WaveAtlas — closing…");
        }
        stop_backend(&app);
        app.exit(0);
    });
}

fn start_backend(app: tauri::AppHandle) -> Result<(), String> {
    let resources = app.path().resource_dir().map_err(|e| e.to_string())?;
    let executable = resources.join("backend/waveatlas-backend.exe");
    let workspace = app.state::<Backend>().workspace.clone();
    std::fs::create_dir_all(workspace.join("logs")).map_err(|e| e.to_string())?;
    let stderr =
        std::fs::File::create(workspace.join("logs/launcher.log")).map_err(|e| e.to_string())?;
    let mut command = Command::new(executable);
    command
        .arg("--workspace")
        .arg(&workspace)
        .env("PYTHONUTF8", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(stderr);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000); // CREATE_NO_WINDOW; the pipes remain available.
    }
    let mut child = command
        .spawn()
        .map_err(|e| format!("Could not start the analysis engine: {e}. Reinstall WaveAtlas."))?;
    let stdout = child
        .stdout
        .take()
        .ok_or("Missing backend readiness pipe")?;
    *app.state::<Backend>().child.lock().unwrap() = Some(child);
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            match line {
                Ok(line) => {
                    if tx.send(line).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
    let deadline = Instant::now() + Duration::from_secs(180);
    let ready: Ready = loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let line = rx.recv_timeout(remaining).map_err(|_| {
            "The analysis engine did not become ready. See Help → Open Logs.".to_string()
        })?;
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) {
            if value["type"] == "error" {
                return Err(value["message"]
                    .as_str()
                    .unwrap_or("Startup failed")
                    .to_string());
            }
            if value["type"] == "ready" {
                break serde_json::from_value(value).map_err(|e| e.to_string())?;
            }
        }
    };
    let url: tauri::Url = ready
        .url
        .parse()
        .map_err(|_| "Invalid backend launch URL")?;
    if url.scheme() != "http"
        || url.host_str() != Some("127.0.0.1")
        || url.origin().ascii_serialization() != ready.origin
    {
        return Err("Unexpected backend address".into());
    }
    let origin = ready.origin.clone();
    *app.state::<Backend>().ready.lock().unwrap() = Some(ready);
    let window = WebviewWindowBuilder::new(&app, "main", WebviewUrl::External(url))
        .title("WaveAtlas")
        .inner_size(1400.0, 900.0)
        .min_inner_size(900.0, 600.0)
        .disable_drag_drop_handler()
        .on_navigation(move |url| url.origin().ascii_serialization() == origin)
        .on_new_window(|_, _| tauri::webview::NewWindowResponse::Deny)
        .on_download(|window, event| {
            if let tauri::webview::DownloadEvent::Requested { destination, .. } = event {
                let Ok(downloads) = window.app_handle().path().download_dir() else {
                    return false;
                };
                let Some(name) = destination.file_name() else {
                    return false;
                };
                let mut path = downloads.join(name);
                if path.exists() {
                    let stamp = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos();
                    path = downloads.join(format!("{stamp}-{}", name.to_string_lossy()));
                }
                *destination = path;
            }
            true
        })
        .build()
        .map_err(|e| e.to_string())?;
    let handle = app.clone();
    window.on_window_event(move |event| {
        if let WindowEvent::CloseRequested { api, .. } = event {
            api.prevent_close();
            close_requested(handle.clone());
        }
    });
    if let Some(loading) = app.get_webview_window("loading") {
        let _ = loading.destroy();
    }
    // Watch for unexpected backend termination without ever logging the private
    // readiness message or putting credentials in process arguments.
    loop {
        std::thread::sleep(Duration::from_secs(1));
        if app.state::<Backend>().closing.load(Ordering::SeqCst) {
            break;
        }
        let exited = app
            .state::<Backend>()
            .child
            .lock()
            .unwrap()
            .as_mut()
            .map(|child| child.try_wait().ok().flatten().is_some())
            .unwrap_or(true);
        if exited {
            show_error(&app, "The analysis engine stopped. Close and reopen WaveAtlas to recover saved work. Logs are available under Help → Open Logs.");
            break;
        }
    }
    Ok(())
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _, _| {
            if let Some(window) = app
                .get_webview_window("main")
                .or_else(|| app.get_webview_window("loading"))
            {
                let _ = window.unminimize();
                let _ = window.show();
                let _ = window.set_focus();
            }
        }))
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let workspace = app.path().local_data_dir()?.join("WaveAtlas");
            app.manage(Backend {
                child: Mutex::new(None),
                ready: Mutex::new(None),
                closing: AtomicBool::new(false),
                workspace,
            });
            use tauri::menu::{Menu, MenuItem, Submenu};
            let data = MenuItem::with_id(app, "data", "Open Data Folder", true, None::<&str>)?;
            let logs = MenuItem::with_id(app, "logs", "Open Logs", true, None::<&str>)?;
            let downloads =
                MenuItem::with_id(app, "downloads", "Open Downloads", true, None::<&str>)?;
            let help = Submenu::with_items(app, "Help", true, &[&data, &logs, &downloads])?;
            app.set_menu(Menu::with_items(app, &[&help])?)?;
            app.on_menu_event(|app, event| {
                let root = app.state::<Backend>().workspace.clone();
                let path = match event.id().as_ref() {
                    "data" => Some(root),
                    "logs" => Some(root.join("logs")),
                    "downloads" => app.path().download_dir().ok(),
                    _ => None,
                };
                if let Some(path) = path {
                    // Only these fixed local folders can be opened, never webview input.
                    #[cfg(windows)]
                    {
                        let _ = Command::new("explorer.exe").arg(path).spawn();
                    }
                    #[cfg(not(windows))]
                    {
                        let _ = path;
                    }
                }
            });
            let loading =
                WebviewWindowBuilder::new(app, "loading", WebviewUrl::App("index.html".into()))
                    .title("WaveAtlas")
                    .inner_size(720.0, 420.0)
                    .build()?;
            let handle = app.handle().clone();
            loading.on_window_event(move |event| {
                if let WindowEvent::CloseRequested { api, .. } = event {
                    api.prevent_close();
                    close_requested(handle.clone());
                }
            });
            let handle = app.handle().clone();
            std::thread::spawn(move || {
                if let Err(error) = start_backend(handle.clone()) {
                    show_error(&handle, &error);
                    stop_backend(&handle);
                }
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("Could not initialize WaveAtlas")
        .run(|app, event| {
            if let tauri::RunEvent::ExitRequested { api, .. } = event {
                if !app.state::<Backend>().closing.load(Ordering::SeqCst) {
                    api.prevent_exit();
                    close_requested(app.clone());
                }
            }
        });
}
