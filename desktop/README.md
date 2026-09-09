# WaveAtlas for Windows

The Windows app uses the existing React/FastAPI analysis pipeline, a Tauri desktop window, and a bundled Python runtime. All processing runs on this computer. The supported initial target is **Windows 11 x64, CPU inference**.

## Decisions for version 0.1

- A dedicated application window and per-user NSIS installer. Python/native libraries/models live inside the installation; users receive one installer and a Start-menu shortcut.
- Data lives in `%LOCALAPPDATA%\WaveAtlas`, outside the install directory. Each Windows user has one independent workspace. Copying the installer does not copy research data.
- One active analysis; additional starts receive a readable busy message. No persistent job queue or tray/background service.
- **User-confirmed:** closing during a run asks whether to keep working or stop and exit. Shutdown allows cooperative cancellation, then terminates after a bounded wait. Startup marks interrupted runs resumable.
- A complete, valid extraction checkpoint can resume. If extraction never checkpointed, it can rerun from the input. A partially written/corrupt checkpoint or a different app/model version requires a new run; prior results remain untouched.
- **User-confirmed:** publisher is Rohit Mahesh individually. Signing access is still to be arranged. CI produces explicitly unsigned candidates; those are not presented as signed public releases.
- WebView2's offline installer ships with the package. This increases download size and avoids requiring internet at installation. Windows/university policy may still require IT to provision prerequisites.
- Exports go to Downloads, with collision protection. Help provides Open Downloads, Open Data Folder, and Open Logs. External web pages cannot replace the application window.
- Updates are manual and preserve data. Automatic updates, shared folders, cloud imports, and macOS/Linux support are deferred.

## Build on Windows

Build prerequisites: Git, Python **3.11 x64**, Node **22**, Rust **1.98.1**, and Visual Studio Build Tools with the Desktop development with C++ workload and Windows SDK. End users do not install these tools. Use a dedicated virtual environment for the build; the script installs the checked-in hashed dependency locks into the active Python environment.

From a PowerShell window at the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
.\scripts\build-windows.ps1 -FetchModels
```

The script installs locked dependencies, builds the frontend, verifies the pinned model hashes, generates build metadata/notices/icon, runs tests, freezes Python, loads all four packaged models, exercises all three analysis modes through the packaged API, verifies relaunch persistence, and builds the installer. It fails on any unsuccessful native command.

Outputs:

- `desktop/src-tauri/target/release/bundle/nsis/*-setup.exe`
- `desktop/src-tauri/target/release/bundle/nsis/SHA256SUMS.txt`
- `desktop/build-manifest.json` (source commit, app/runtime versions, model release and hashes)
- `desktop/THIRD_PARTY_NOTICES.txt`

The `.github/workflows/windows.yml` workflow runs these steps on Windows and attaches unsigned candidates to the workflow run. It does not publish a GitHub Release or change the hosted service.

Models come from the fixed release and checksums in `desktop/models.json`. Updating that file is an explicit model-version change. Runtime and build dependencies are locked separately for Python 3.11/Windows. Regenerate locks deliberately using `uv pip compile --python-version 3.11 --python-platform windows --generate-hashes`; review and test all resulting changes. `desktop/package-lock.json` and `desktop/src-tauri/Cargo.lock` lock the shell build dependencies.

## Signing

After setting up an individual publicly trusted signing identity, provide a signing wrapper executable/script in `WAVEATLAS_SIGN_COMMAND`. It must accept exactly one file path, sign **and timestamp** that file, and exit nonzero on any failure. The same wrapper signs the Python executable and is supplied to Tauri for the shell and installer. If using a PowerShell script, wrap it in a `.cmd` or executable that invokes PowerShell and propagates its exit code; do not put shell fragments or credentials in the environment variable.

```powershell
$env:WAVEATLAS_SIGN_COMMAND = 'C:\BuildTools\sign-waveatlas.cmd'
.\scripts\build-windows.ps1 -FetchModels -Signed
```

The signed build verifies Authenticode status on the backend, shell, and installer. Configure and test an RFC 3161 timestamp in the wrapper. Keep signing credentials in the signing service/CI secret store, never in the repo. The individual publisher identity must match the certificate. Signing may not immediately eliminate SmartScreen warnings for a new application.

Microsoft's [Artifact Signing integration](https://learn.microsoft.com/en-us/azure/artifact-signing/how-to-signing-integrations) describes SignTool and CI setup; [reputation guidance](https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/smartscreen-reputation) explains first-release prompts.

## Runtime and data

The native shell starts a hidden Python process and receives its readiness URL over a private pipe. Python binds an OS-assigned loopback socket once. A one-use launch nonce establishes an HttpOnly session cookie; saved job ownership comes from `workspace.json`, not that cookie. HTTP, static content, and WebSockets require launch authentication and validate the host/origin. The remote webview has no Tauri IPC capabilities. Only fixed native menu actions can open local folders.

The process holds `workspace.lock` for its lifetime. The database is migrated before serving requests; an upgrade backs up the database under `backups/`. An unknown schema version is rejected. Artifacts use workspace-relative references and atomic file replacement. Scratch files are disposable and cleaned on the next launch; saved checkpoints live under `artifacts/`. Backend logs rotate under `logs/`, and `launcher.log` describes the latest launch. Closing/crashing the shell closes the control pipe so its backend stops too.

To back up or move a workspace, **close WaveAtlas and wait for the process to exit**, then copy the whole `%LOCALAPPDATA%\WaveAtlas` folder, including `workspace.json`, `waveatlas.sqlite`, and `artifacts/`. Never restore only the database or overwrite a running workspace. For a failed upgrade, preserve the failed workspace for diagnosis and restore the matching pre-upgrade database with its associated workspace files before using the old application. Running databases on network shares/actively synchronized folders is unsupported.

The installer's uninstall operation removes application files and preserves research data. Delete the data folder separately only when you intend to erase those runs. Backups and scientific/debug artifacts can grow; use run deletion and periodic closed-workspace backups. Logs and diagnostic bundles should be reviewed before sharing because they may contain local file names.

## Release gates

- Successful Windows build and real packaged-backend smoke tests.
- Install/launch/upload/export/close during analysis/reopen/upgrade/uninstall tested through the actual GUI on clean Windows 11 lab hardware, including a standard-user account.
- Offline prerequisite installation and runtime verified on a clean machine; native ONNX dependencies include the [Visual C++ runtime requirement](https://onnxruntime.ai/docs/install/).
- Measured installer size, peak RAM and runtime on representative lab inputs; no minimum RAM promise is inferred from the hosted allocation.
- Numerical parity reviewed on representative real data in addition to the synthetic regression tests.
- Valid publisher signing identity and timestamp verification, distribution license/model redistribution permission, and reviewed third-party notices. These remain release-owner inputs; the generated notices do not establish model redistribution rights.

See [the scope assessment](../docs/windows-distribution-scope.md) for the original estimate and full acceptance criteria. No cloud resource retirement is part of this branch's implementation.
