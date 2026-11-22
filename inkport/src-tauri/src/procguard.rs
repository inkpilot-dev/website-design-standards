//! Makes sure no helper process outlives Inkport.
//!
//! The engine watches its stdin and exits when the app goes away, and on Linux the answer
//! engine (llama-server) is tied to the app with PR_SET_PDEATHSIG. Windows has neither for a
//! plain child process: if Inkport is ended from Task Manager or crashes, llama-server would keep
//! running, holding gigabytes of memory until the next restart. A job object with
//! KILL_ON_JOB_CLOSE fixes that: Windows ends every process in the job when the last handle to
//! the job is closed, and the only handle lives in this process.

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
