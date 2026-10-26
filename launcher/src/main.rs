// Self-extracting single-file launcher for music-remover.
//
// The whole runtime (exe + OpenVINO DLLs + model) is embedded from ./payload at build
// time. On first run it is unpacked to %LOCALAPPDATA%\music-remover\rt-<bytes>\ (keyed by
// payload size, so a new build re-extracts). Subsequent runs skip straight to launching
// the inner exe, forwarding all args, stdio, and exit code.

use std::collections::hash_map::DefaultHasher;
use std::ffi::OsString;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

use include_dir::{include_dir, Dir, File};
use rayon::prelude::*;

static PAYLOAD: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/payload");

fn main() {
    let base = cache_base();
    let marker = base.join(".ready");
    if !marker.exists() {
        eprintln!(
            "music-remover: first run — unpacking runtime to {} ...",
            base.display()
        );
        if let Err(e) = std::fs::create_dir_all(&base).and_then(|_| extract(&PAYLOAD, &base)) {
            eprintln!("failed to unpack runtime: {e}");
            std::process::exit(1);
        }
        let _ = std::fs::write(&marker, b"ok");
    }

    let exe = base.join("music-remover.exe");
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
    match Command::new(&exe).args(&args).status() {
        Ok(status) => std::process::exit(status.code().unwrap_or(1)),
        Err(e) => {
            eprintln!("failed to launch {}: {e}", exe.display());
            std::process::exit(1);
        }
    }
}

/// Unpack the whole embedded tree. Directories are created single-threaded first (cheap,
/// and avoids any race between concurrent `create_dir_all` calls on shared parent paths);
/// the file writes — which dominate first-run wall time, since the payload includes a
/// ~100MB model file and several large DLLs — then run in parallel across threads, since
/// each file's write is fully independent of the others.
fn extract(dir: &Dir, base: &Path) -> std::io::Result<()> {
    create_dirs(dir, base)?;
    collect_files(dir).par_iter().try_for_each(|f| std::fs::write(base.join(f.path()), f.contents()))
}

fn create_dirs(dir: &Dir, base: &Path) -> std::io::Result<()> {
    for d in dir.dirs() {
        std::fs::create_dir_all(base.join(d.path()))?;
        create_dirs(d, base)?;
    }
    Ok(())
}

fn collect_files<'a>(dir: &Dir<'a>) -> Vec<&'a File<'a>> {
    let mut out: Vec<&'a File<'a>> = dir.files().collect();
    for d in dir.dirs() {
        out.extend(collect_files(d));
    }
    out
}

fn cache_base() -> PathBuf {
    let root = std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    root.join("music-remover").join(format!("rt-{:016x}", payload_hash(&PAYLOAD)))
}

/// Content hash of every embedded path + file's bytes — used as the cache-dir key so a
/// changed build (even one that happens to total the same byte count as the last one)
/// always extracts fresh instead of reusing a stale runtime.
fn payload_hash(dir: &Dir) -> u64 {
    let mut files = collect_files(dir);
    files.sort_by_key(|f| f.path());
    let mut hasher = DefaultHasher::new();
    for f in files {
        f.path().hash(&mut hasher);
        f.contents().hash(&mut hasher);
    }
    hasher.finish()
}
