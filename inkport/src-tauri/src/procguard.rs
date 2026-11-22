//! Makes sure no helper process outlives Inkport.
//!
//! The engine watches its stdin and exits when the app goes away, and on Linux the answer
//! engine (llama-server) is tied to the app with PR_SET_PDEATHSIG. Windows has neither for a
//! plain child process: if Inkport is ended from Task Manager or crashes, llama-server would keep
//! running, holding gigabytes of memory until the next restart. A job object with
//! KILL_ON_JOB_CLOSE fixes that: Windows ends every process in the job when the last handle to
//! the job is closed, and the only handle lives in this process.
//!
//! macOS has neither job objects nor PR_SET_PDEATHSIG. There (and on Linux, as a second line) a
//! tiny shell loop watches the app and the helper and ends the helper once the app is gone.

#[cfg(windows)]
mod imp {
    use std::os::windows::io::RawHandle;
    use std::sync::OnceLock;
    use windows_sys::Win32::System::JobObjects::{AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE};

    /// The job handle, stored as an address so the static is Send + Sync (0 = could not be created).
    static JOB: OnceLock<usize> = OnceLock::new();

    fn job() -> usize {
        *JOB.get_or_init(|| {
            // SAFETY: plain Win32 calls with valid arguments; the handle is intentionally never
            // closed (closing it is what ends the children), so it lives as long as the process.
            unsafe {
                let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
                if job as usize == 0 {
                    return 0;
                }
                let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                let ok = SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info as *const _ as *const core::ffi::c_void, std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32);
                if ok == 0 {
                    return 0;
                }
                job as usize
            }
        })
    }

    pub fn adopt_raw(process: RawHandle) {
        let job = job();
        if job == 0 || process.is_null() {
            return;
        }
        // SAFETY: both handles are valid for the duration of the call. Failure (for example a
        // process that already exited) is harmless and ignored.
        unsafe {
            AssignProcessToJobObject(job as _, process as _);
        }
    }
}

/// Tie a child process's lifetime to this process (Windows; a no-op elsewhere).
pub fn adopt_tokio(child: &tokio::process::Child) {
    #[cfg(windows)]
    if let Some(h) = child.raw_handle() {
        imp::adopt_raw(h);
    }
    #[cfg(not(windows))]
    let _ = child;
}

/// The watcher: `$1` is the app, `$2` the helper.
#[cfg(unix)]
const WATCH: &str = r#"( while kill -0 "$1" 2>/dev/null && kill -0 "$2" 2>/dev/null; do sleep 2; done
if ! kill -0 "$1" 2>/dev/null; then kill "$2" 2>/dev/null; sleep 3; kill -9 "$2" 2>/dev/null; fi ) >/dev/null 2>&1 &"#;

/// End `helper_pid` when this process is gone, however it went (macOS and Linux; a no-op on
/// Windows, where the job object does this).
///
/// The watcher is a shell loop in the background, detached from this process: it looks every two
/// seconds whether both are still there, and if the app has disappeared it asks the helper to
/// stop, then makes it. When the helper ends first (the normal case) the watcher just ends too.
pub fn end_with_app(helper_pid: u32) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        use std::process::{Command, Stdio};
        // The outer shell returns at once (so nothing is left to reap here); the loop it started
        // belongs to nobody and survives this process.
        let _ = Command::new("/bin/sh")
            .args(["-c", WATCH, "inkport-watch", &std::process::id().to_string(), &helper_pid.to_string()])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0)
            .status();
    }
    #[cfg(not(unix))]
    let _ = helper_pid;
}

#[cfg(all(test, unix))]
mod tests {
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    fn alive(pid: u32) -> bool {
        // SAFETY: signal 0 only asks whether the process exists.
        unsafe { libc::kill(pid as i32, 0) == 0 }
    }

    /// A stand-in "app" (a shell that starts a helper and the watcher, then is killed outright):
    /// the helper must be gone a few seconds later, although nobody told it to stop.
    #[test]
    fn helper_is_ended_when_the_app_is_killed_outright() {
        let dir = tempfile::tempdir().unwrap();
        let pidfile = dir.path().join("helper.pid");
        let watch = super::WATCH;
        // The "app": starts a long-lived helper, starts the same watcher the real app starts, waits.
        let app = format!("sleep 300 & echo $! > '{}'; /bin/sh -c '{}' inkport-watch $$ $!; sleep 300", pidfile.display(), watch.replace('\'', "'\\''"));
        let mut app = Command::new("/bin/sh").args(["-c", &app]).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap();
        let started = Instant::now();
        while !pidfile.exists() || std::fs::read_to_string(&pidfile).unwrap().trim().is_empty() {
            assert!(started.elapsed() < Duration::from_secs(5));
            std::thread::sleep(Duration::from_millis(50));
        }
        let helper: u32 = std::fs::read_to_string(&pidfile).unwrap().trim().parse().unwrap();
        std::thread::sleep(Duration::from_millis(500));
        assert!(alive(helper), "the helper should be running while the app lives");
        app.kill().unwrap(); // SIGKILL: no chance to tidy up
        app.wait().unwrap();
        let killed = Instant::now();
        while alive(helper) {
            assert!(killed.elapsed() < Duration::from_secs(12), "the helper outlived the app");
            std::thread::sleep(Duration::from_millis(200));
        }
    }
}
