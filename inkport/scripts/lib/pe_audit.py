"""Static dependency audit for a Windows payload: nothing may need a DLL that a clean PC lacks.

The clean-machine check runs on build machines that have Visual Studio and the VC++ redistributable
installed system-wide, so a missing runtime DLL would go unnoticed there and only fail on a real
user's fresh Windows install. This audit closes that gap without running anything: it reads the
import table of every .exe/.dll/.pyd in the payload and requires each Microsoft C/C++/OpenMP runtime
DLL to be shipped where the Windows loader will find it (next to the importing file, next to the
program that loads it, or inside the same Python package). Other non-system imports that are not
shipped are reported as warnings.

Pure standard library; works on any OS (it only parses files).
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

# Runtime libraries that are NOT part of Windows and must be shipped app-locally.
REDIST = re.compile(
    r"^(msvcp\d+.*|vcruntime\d+.*|vcomp\d+.*|concrt\d+.*|vccorlib\d+.*|vcamp\d+.*|mfc\d+.*|atl\d+.*"
    r"|libomp\d*.*|libiomp5md|libgomp-.*|libgcc_s.*|libstdc\+\+-.*|libwinpthread-.*)\.dll$"
)
# Present on every supported Windows 10/11 installation (System32), including the universal CRT.
SYSTEM = re.compile(
    r"^(api-ms-win-.*|ext-ms-.*|kernel32|kernelbase|ntdll|user32|gdi32|gdi32full|advapi32|shell32|ole32|oleaut32"
    r"|ws2_32|wsock32|mswsock|crypt32|bcrypt|bcryptprimitives|ncrypt|ucrtbase|msvcrt|shlwapi|version|winmm"
    r"|comctl32|comdlg32|rpcrt4|secur32|sspicli|iphlpapi|psapi|dbghelp|dbgcore|userenv|setupapi|cfgmgr32"
    r"|powrprof|netapi32|dnsapi|wldap32|imm32|dwmapi|uxtheme|propsys|wintrust|pdh|winhttp|wininet|urlmon"
    r"|d3d9|d3d11|d3d12|dxgi|dxcore|opengl32|glu32|dwrite|d2d1|shcore|normaliz|winspool\.drv|ntdsapi|wtsapi32"
    r"|credui|cabinet|msi|imagehlp|winusb|hid|avrt|mfplat|mf|mfreadwrite|profapi|kernel\.appcore|oleacc|usp10"
    r"|gdiplus|msimg32|tdh|wevtapi|winsta|wer|fwpuclnt|rstrtmgr|authz|netutils|srvcli|wkscli|samcli|logoncli"
    r"|mpr|msasn1|cryptsp|cryptbase|devobj|synchronization|combase|windowscodecs|msvfw32|avicap32|avifil32"
    r"|opengl32sw|ksuser|mmdevapi|dsound|winnsi|nsi|dhcpcsvc|sechost|bcp47langs|mscoree|oledlg|esent|wlanapi)(\.dll)?$"
)
# Loaded only when the matching GPU driver is installed; their absence is handled at runtime.
OPTIONAL = re.compile(r"^(vulkan-1|nvcuda|nvml|opencl|cudart64_.*|cublas64_.*|cublaslt64_.*|cudnn.*|nvrtc.*|cufft.*|curand.*|cusparse.*|cusolver.*|nvjitlink.*|caffe2_nvrtc|nvtoolsext64_1|zlibwapi)\.dll$")


def pe_imports(path: Path) -> tuple[list[str], list[str]]:
    """Return (imports, delay_imports) DLL names (lower-case) of a PE file; ([], []) if not PE."""
    data = path.read_bytes()
    if len(data) < 0x40 or data[:2] != b"MZ":
        return [], []
    (pe,) = struct.unpack_from("<I", data, 0x3C)
    if pe + 24 > len(data) or data[pe : pe + 4] != b"PE\0\0":
        return [], []
    nsec, opt_size = struct.unpack_from("<H", data, pe + 6)[0], struct.unpack_from("<H", data, pe + 20)[0]
    opt = pe + 24
    (magic,) = struct.unpack_from("<H", data, opt)
    dirs = opt + (112 if magic == 0x20B else 96)
    sections = []
    for i in range(nsec):
        off = opt + opt_size + 40 * i
        vsize, vaddr, rsize, rptr = struct.unpack_from("<IIII", data, off + 8)
        sections.append((vaddr, max(vsize, rsize), rptr))

    def at(rva: int) -> int | None:
        for vaddr, size, rptr in sections:
            if vaddr <= rva < vaddr + size:
                return rva - vaddr + rptr
        return None

    def cstr(rva: int) -> str | None:
        off = at(rva)
        if off is None or off >= len(data):
            return None
        end = data.find(b"\0", off, off + 512)
        return data[off:end].decode("latin-1").lower() if end > off else None

    def table(index: int, entry: int, name_off: int) -> list[str]:
        if dirs + 8 * index + 8 > opt + opt_size:
            return []
        rva, _size = struct.unpack_from("<II", data, dirs + 8 * index)
        off = at(rva) if rva else None
        names: list[str] = []
        while off is not None and off + entry <= len(data):
            raw = data[off : off + entry]
            if raw == b"\0" * entry:
                break
            (name_rva,) = struct.unpack_from("<I", raw, name_off)
            if not name_rva:
                break
            name = cstr(name_rva)
            if name:
                names.append(name)
            off += entry
        return names

    return table(1, 20, 12), table(13, 32, 4)


def audit(payload: Path) -> tuple[list[str], list[str], int]:
    """Return (errors, warnings, files_scanned) for a Windows payload directory."""
    binaries = [p for p in payload.rglob("*") if p.is_file() and p.suffix.lower() in (".exe", ".dll", ".pyd")]
    by_name: dict[str, list[Path]] = {}
    for p in binaries:
        by_name.setdefault(p.name.lower(), []).append(p)
    py_root = payload / "python"
    site = py_root / "Lib" / "site-packages"

    def search_dirs(importer: Path) -> list[Path]:
        """Where the Windows loader will look for this file's dependencies (besides System32)."""
        dirs = [importer.parent]
        rel = importer.relative_to(payload).parts
        if rel[0] == "llama":
            dirs.append(payload / "llama")  # llama-server.exe's own folder
        elif rel[0] == "python":
            dirs += [py_root, py_root / "DLLs"]  # python.exe's folder and its standard extension folder
            try:
                top = importer.relative_to(site).parts[0]
                # Packages register their own DLL folders (os.add_dll_directory / *.libs).
                # (delvewheel puts uniquely renamed DLLs in "<distribution>.libs", whose name need not match the package).
                dirs += [d for d in [site / top, *site.glob("*.libs")] if d.is_dir()]
                dirs += [d for d in (site / top).rglob("*") if d.is_dir() and d.name.lower() in ("lib", "libs", "bin", ".libs")] if (site / top).is_dir() else []
            except ValueError:
                pass
        return dirs

    errors: list[str] = []
    warnings: dict[str, str] = {}
    for p in binaries:
        try:
            imports, delayed = pe_imports(p)
        except (OSError, struct.error):
            continue
        rel = p.relative_to(payload).as_posix()
        dirs = search_dirs(p)
        for name in imports:
            if SYSTEM.match(name) or OPTIONAL.match(name):
                continue
            shipped = by_name.get(name, [])
            reachable = any(s.parent in dirs for s in shipped)
            if REDIST.match(name):
                if not reachable:
                    errors.append(f"{rel} needs {name}, which is not shipped next to it (a PC without the Visual C++ redistributable could not start it)")
            elif not shipped:
                warnings.setdefault(name, rel)
        for name in delayed:
            if not (SYSTEM.match(name) or OPTIONAL.match(name) or by_name.get(name)):
                warnings.setdefault(name, rel + " (delay-loaded)")
    return errors, [f"{name} (needed by {rel}) is neither shipped nor a known Windows system DLL" for name, rel in sorted(warnings.items())], len(binaries)


if __name__ == "__main__":
    import sys

    errs, warns, n = audit(Path(sys.argv[1]))
    for w in warns:
        print("warning:", w)
    for e in errs:
        print("ERROR:", e)
    print(f"{n} Windows binaries scanned, {len(errs)} error(s), {len(warns)} warning(s)")
    sys.exit(1 if errs else 0)
