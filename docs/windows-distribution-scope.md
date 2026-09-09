WaveAtlas Windows release scope
==============================

Assessment date: September 9, 2026. This document supersedes the preliminary multi-platform assessment. It scopes implementation; no executable has been built, model inference benchmarked, production data migrated, or cloud configuration changed for this assessment.

Proposed first release
----------------------

Ship a signed `WaveAtlas-<version>-windows-x64-setup.exe`. A researcher installs it, opens WaveAtlas from the Start menu, and uses the existing viewer in a dedicated application window. No terminal, Python, Node, Docker, Google account, or model download is required. Models and runtime assets ship with the installer; installation and analysis should work offline once the installer is available. Managed-device policies may still require university IT involvement.

Baseline assumptions: Windows 11 x64; one local workspace per Windows user; CPU inference; one active analysis; all three existing analysis modes; manual updates. The precise supported Windows versions and minimum hardware will be recorded after the feasibility test. Windows 10, ARM64, macOS, and Linux are outside this release's test matrix.

Estimated effort is **20–30 engineer-days (4–6 engineering weeks)** for one engineer familiar with the code, with Windows test/build access and model assets available. Signing verification and lab-review waiting time are additional calendar dependencies. A first usable internal build is targeted around days 5–8, but is not equivalent to a validated lab release. The estimate includes a thin desktop shell and signing integration; do not add the earlier desktop-shell estimate again.

Architecture recommendation
---------------------------

Use the existing React interface and Python analysis engine. Evaluate **Tauri + a PyInstaller-packaged Python backend** during the first milestone. Tauri provides the desktop window and installer; Python retains FastAPI, SQLite, local artifact storage, and the scientific code. This is a candidate architecture until verified with the actual model bundle on Windows.

The shell starts the backend with explicit resource/data paths, waits for readiness, and loads the backend-served UI at a loopback address. Serving the UI and API from the same origin preserves the existing HTTP and WebSocket contract. Restrict native shell permissions and navigation; allow only required desktop operations. The packaged Python runtime can remain a directory of files inside the installation, while users receive one installer and one application shortcut.

Tauri documents [Python API servers packaged as sidecars](https://v2.tauri.app/develop/sidecar/) and [Windows installers/WebView2 distribution](https://v2.tauri.app/distribute/windows-installer/). PyInstaller requires a Windows build environment for Windows output; do not treat this Linux workspace as proof of Windows packaging compatibility. See the [PyInstaller manual](https://pyinstaller.org/en/stable/).

Use `%LOCALAPPDATA%\WaveAtlas\` for user state, separated into database/workspace, artifacts, scratch, logs, and configuration. Discover the Windows user-data path rather than assuming a particular drive or username. Installed assets are immutable. The installer should default to per-user installation where prerequisites and device policy permit, and upgrades/uninstall should preserve research data by default.

Repository findings and work items
---------------------------------

| Work item | Evidence in this repository | Planned result |
| --- | --- | --- |
| Production local entry point | `scripts/dev-backend.sh` is Bash, uses `/tmp`, binds `0.0.0.0`, and enables reload. `app/db.py` constructs an engine at import time. | A frozen entry point configures paths before imports, acquires a workspace lock, runs migrations, and starts one loopback backend without a console window. |
| Bundle resources | Config lookup includes `configs/default.yaml`; model lookup requires an export directory; frontend/docs use source-tree paths. | One resource resolver supplies frontend, default config, docs, migrations, and all four ONNX models from the installation. Writable plotting caches/logs use user storage. |
| Desktop lifecycle | No native shell or installed launcher exists. | Loading/error window, single-instance launch, backend readiness and exit handling, window-close behavior, file selection/export support, and clean child-process shutdown. |
| Local ownership | `app/api/deps.py` assigns run ownership from a browser `sid` cookie; frontend state uses localStorage. | A persistent workspace identity survives browser/webview state loss and changing loopback ports. Short-lived launch authorization remains separate from ownership. |
| Local access protection | Hosted API assumes web session ownership; local runtime is not implemented. | Loopback binding, Host/Origin checks, authenticated launch bootstrap for HTTP/WebSockets, and restricted shell navigation/native commands. Validate the actual webview cookie behavior. |
| Upload behavior | `useJobSession.ts` tries a GCS resumable session, catches any failure, then uploads via API. | Explicit local capabilities select multipart upload directly. User-facing status describes opening/copying a local input, without an expected cloud-upload warning on every run. |
| Job admission and recovery | Jobs use FastAPI `BackgroundTasks`; `claim_resume` accepts failed/cancelled jobs but not abandoned `in_progress` rows. | Enforce one active analysis across API calls, show a clear busy state, reconcile interrupted jobs on startup under the workspace lock, and validate resumable checkpoints. |
| Portable artifacts | `LocalArtifactStore` persists absolute paths and `put_file` reads the whole source into memory. | Workspace-relative artifact keys with compatibility handling; stream large file copies where warranted by benchmarks. Provide closed-workspace backup/restore instructions and Open Data Folder. |
| Dependencies and formats | Scientific native packages are needed. GCS imports are deferred. XLSX/XLS readers request `openpyxl`/`xlrd`, neither listed in `requirements.txt`. | A pinned Windows runtime dependency set with required Excel engines, frozen-import hooks, bundled native libraries, and no unused cloud/Postgres/Wolfram requirement. Validate each advertised input format. |
| Scientific reproducibility | Four models are external; fetch script defaults to the latest release. | Pin model release/checksums and runtime dependencies; record app/model/config versions with results; use fixed inputs and numerical tolerances for parity tests. |
| Packaging and updates | Current distribution is Docker; no desktop build/release files exist. | Windows build automation, versioned installer, executable/installer signing and timestamps, checksums, release notes, diagnostics, and data-preserving manual upgrades. |

Keep the existing analysis algorithms, viewer, run history, filters, regression plots, CSV exports, and configuration documentation. Disable unsupported configuration choices such as Wolfram execution in the packaged local profile with a clear explanation.

Job behavior for version 1
-------------------------

- One analysis runs at a time. Additional start attempts receive a clear busy response; a persistent multi-job queue is deferred.
- During an active run, closing the window offers to keep working or stop and exit. Stop requests cooperative cancellation, waits for a bounded shutdown, and records or recovers an interrupted state if forced termination is required.
- Startup reconciles stale active states only after confirming no other app instance owns the workspace. Resume uses validated durable checkpoints; unavailable extraction checkpoints cause recomputation, with an explanation.
- A backend crash produces a visible error and diagnostic location rather than a blank window. Minimize should preserve work; test sleep/wake separately and do not promise processing while the machine sleeps.
- Start with the current background execution behind an explicit admission guard. If responsiveness or termination tests fail, introduce a supervised worker process within the runtime milestone. If Python multiprocessing is used, handle frozen Windows process startup explicitly, including `freeze_support`; see [PyInstaller process guidance](https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html).

Installer and signing
---------------------

Bundle the Python runtime, native numerical libraries, models, frontend, and configuration/docs. Account for ONNX Runtime's [Visual C++ runtime requirement](https://onnxruntime.ai/docs/install/) through a supported prerequisite deployment approach. Tauri uses WebView2 on Windows: select and test an offline-capable runtime deployment option rather than silently downloading it at first install. Missing runtime prerequisites must be handled in the GUI. Measure the installed size and maintenance implications before finalizing the WebView2 mode.

Plan public-trust signing of the project's executables and final installer, including timestamps and signature verification. Establish lab/university ownership of signing access; recipients only need the signed installer. Microsoft Artifact Signing is one candidate; the lab must supply or establish an eligible signing identity. Signing does not guarantee immediate SmartScreen reputation. See Microsoft's [signing integration](https://learn.microsoft.com/en-us/azure/artifact-signing/how-to-signing-integrations) and [reputation guidance](https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/smartscreen-reputation).

Automate release builds on Windows, with exact dependency/model versions and final artifact hashes. Manual upgrades replace application files and migrate the user database after a consistent backup. Refuse to open a newer unsupported schema with an older application. Uninstall preserves the workspace; deliberate data deletion is a separate user action. Basic support includes version information, log export, install/update instructions, and a named release maintainer.

Milestones and estimates
------------------------

| Milestone | Completion criterion | Engineer-days |
| --- | --- | --- |
| 1. Windows packaging feasibility | Installed shell launches packaged backend; all four ONNX models load; a representative standard run completes on a Windows machine without developer tools. Measure startup, inference, RAM and package size. | 2–3 |
| 2. Local runtime and desktop integration | Stable workspace, resource paths, migrations, local access/session handling, direct local uploads, native window lifecycle and working exports. | 5–7 |
| 3. Reliability and data lifecycle | One-job enforcement, cancellation/close/crash recovery, checkpoint validation, artifact paths, backup/upgrade protection, disk/log handling. | 5–7 |
| 4. Install and release pipeline | Offline prerequisite handling, repeatable Windows installer, signing/timestamp integration, version/checksum metadata and manual updates. | 3–5 |
| 5. Packaged validation and lab pilot | All-mode/format regression checks, clean-machine install/upgrade/recovery tests, lab feedback fixes and release instructions. | 5–8 |
| **Total** | **Windows x64 lab release** | **20–30** |

Milestone 1 is the feasibility gate. Re-estimate if the actual model/native-library bundle cannot load reliably, desktop downloads require substantial adaptation, or the measured memory footprint exceeds lab hardware. Avoid claiming a minimum RAM figure from the hosted 8 GiB allocation: actual usage has not been measured. Model files are not present in this checkout, so installer size and standard-mode packaging remain unverified.

Milestones are sequential estimates, with some practical overlap in setup. Signing-account verification, Windows runner provisioning, and lab review should begin early. A representative Windows lab computer and a clean Windows VM/build runner are required to substantiate release readiness; development-machine success alone is insufficient.

Release acceptance checklist
----------------------------

- Install, launch, and uninstall through Windows UI on the agreed Windows 11 x64 matrix, including a standard-user account and non-ASCII/spaced paths.
- Install and complete analysis with network access disabled, using an installer that includes required prerequisites; no developer runtimes or Google credentials are assumed.
- Exercise CSV, TSV, genuine XLS/XLSX, and advertised image formats; cover standard, ripple-family, and large-wave analysis.
- Compare tracks and metrics against fixed reference outputs from the current analysis version using reviewed tolerances. Validate overlays, track details, configuration editing, original/heatmap downloads, and CSV exports in the desktop webview.
- Preserve history and artifacts across reopening, webview storage loss, manual update, uninstall/reinstall, and a documented workspace backup/restore.
- Test simultaneous start requests, double launch, occupied ports, cancel, close during analysis, backend kill, app kill, sleep/wake, disk-full failure, and recovery without duplicate metric rows or permanently active jobs.
- Test failed/interrupted schema upgrades and rejection of unsupported newer schemas. Perform backups only at a consistent state with no active writer, or use a validated database backup mechanism.
- Measure startup time, peak memory, runtime, installer/installed size, and scratch/artifact growth using representative lab inputs, including the existing roughly 50 MB CSV samples. Small fixtures alone are insufficient.
- Verify shipped signatures, timestamps, and clean-machine behavior. Record any IT policy requirements rather than asking users to disable system protection.

Boundaries and remaining inputs
-------------------------------

This release includes a dedicated window, offline processing, local history, existing scientific features, and manual updates. It excludes cloud-history import, cloud shutdown, shared/network workspaces, complete-run exchange between researchers, automatic updates, GPU acceleration, new analysis algorithms, and support for additional operating systems/CPU architectures. Basic metrics/artifact exports and closed-workspace backup instructions remain included.

For hosted retirement, separately scope a verified database/artifact archive and any historical-run importer before decommissioning. Production backups were disabled when inspected; that is relevant to later retirement, but no cloud changes are authorized by this assessment.

Inputs needed for implementation, without blocking this assessment:

- A Windows x64 build/test environment and representative lab computer; confirm any Windows 10 or managed-device requirement before expanding the matrix.
- The exact four ONNX model artifacts/release used for the reference version, plus provenance and redistribution terms. No repository license file was found during inspection; establish the distribution license and third-party notices before release.
- Representative expected maximum input size and a lab reviewer who can accept numerical output comparisons.
- Lab/university signing identity, installer publisher name, and release-maintainer ownership. Public signing is a release gate, not a prerequisite to the unsigned internal feasibility build.

The next implementation step is milestone 1: prove that an installed Windows app can run the existing standard analysis with the bundled models on a clean machine, then confirm the remainder of the estimate from that result.
