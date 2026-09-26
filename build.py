#!/usr/bin/env python3
"""Build the speech server (llama.cpp's llama-server with the POST /tts route of PR #26603) the
game downloads as a runtime, pack it as one zip, and print the catalog entry for it.

    python tools/runtime/build.py source  --dest <dir>                       # clone llama.cpp at the pin
    python tools/runtime/build.py cuda    --version 12.8.1 --dest <dir>      # CUDA toolkit from NVIDIA's redist
    python tools/runtime/build.py build   --source <dir> --flavour cpu --out <dir> [--cuda-root <dir>]
    python tools/runtime/build.py entry   <zip> --flavour cpu                # the catalog entry, again

The same script on the owner's machine, on this one and in the workflow (build-runtime.yml), so a
zip built by hand and a zip the workflow publishes are built the same way. Standard library only.

FLAVOURS. Every one is the upstream release recipe (llama.cpp .github/workflows/release.yml) with
the server as the only target:

    cpu     GGML_BACKEND_DL + GGML_NATIVE=OFF + GGML_CPU_ALL_VARIANTS: one ggml-cpu-<x>.dll per
            instruction set (sse4.2 ... avx2 ... avx512), the best one picked at start-up.
    cuda    the same, plus GGML_CUDA with CMAKE_CUDA_ARCHITECTURES left to upstream's non-native
            default -- on CUDA >= 12.8 that is 75-virtual 80-virtual 86-real 89-real 90-virtual
            120a-real: real code for Turing to Blackwell's consumer cards, PTX for anything newer.
            cudart / cuBLAS / cuBLASLt (and nvJitLink where the toolkit has it) ride in the zip.
    vulkan  the same, plus GGML_VULKAN (needs the Vulkan SDK's glslc on PATH).
    metal   macOS arm64: one static binary, the Metal library embedded.

On Windows the MSVC runtime (msvcp140 / vcruntime140 / vcruntime140_1 / vcomp140) is copied beside
the program, app-local, so a machine without the Visual C++ redistributable still starts it.

THE ZIP is flat: llama-server(.exe) at its root, every library beside it, and the licences under
licenses/. Named llama-tts-server-<commit7>-<os>-<flavour>-<arch>.zip; a <zip>.json beside it is
the catalog entry (sha256 and bytes measured from the zip written, url in the release's shape).
"""

import argparse
import glob
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile

# The PR head the game's speech engine was measured on (docs/ops/voice_rig.md). Move it only
# together with a measurement: the route's request and answer are what QwenTtsEngine speaks.
PIN = "435d4116eda2abbcdc3d63d85204470070f32705"
UPSTREAM = "https://github.com/ggml-org/llama.cpp"
# Where the owner publishes the zips. A release per pin: tag llama-tts-server-<commit7>.
RELEASES = "https://github.com/marbleworks/tropalm-runtime/releases/download"
PROGRAM = "llama-tts-server"

COMMON = ["-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
          "-DLLAMA_BUILD_TOOLS=ON", "-DLLAMA_BUILD_SERVER=ON", "-DLLAMA_CURL=OFF"]
FLAVOURS = {
    "cpu": ["-DBUILD_SHARED_LIBS=ON", "-DGGML_BACKEND_DL=ON", "-DGGML_NATIVE=OFF", "-DGGML_CPU_ALL_VARIANTS=ON"],
    "cuda": ["-DBUILD_SHARED_LIBS=ON", "-DGGML_BACKEND_DL=ON", "-DGGML_NATIVE=OFF", "-DGGML_CPU_ALL_VARIANTS=ON", "-DGGML_CUDA=ON"],
    "vulkan": ["-DBUILD_SHARED_LIBS=ON", "-DGGML_BACKEND_DL=ON", "-DGGML_NATIVE=OFF", "-DGGML_CPU_ALL_VARIANTS=ON", "-DGGML_VULKAN=ON"],
    "metal": ["-DBUILD_SHARED_LIBS=OFF", "-DGGML_METAL=ON", "-DGGML_METAL_EMBED_LIBRARY=ON",
              "-DGGML_NATIVE=OFF", "-DCMAKE_OSX_DEPLOYMENT_TARGET=13.3"],
}

# NVIDIA's redistributable archives a build needs, by the key redistrib_<version>.json lists them
# under. A key a version does not have is skipped (cuda_crt and libnvvm are 13.x's own).
CUDA_PARTS = ["cuda_cccl", "cccl", "cuda_cudart", "cuda_nvcc", "cuda_crt", "libnvvm", "libcublas",
              "cuda_nvtx", "cuda_profiler_api", "libnvjitlink", "libnvptxcompiler", "cuda_cuobjdump",
              "cuda_nvprune", "cuda_cuxxfilt"]
CUDA_REDIST = "https://developer.download.nvidia.com/compute/cuda/redist/"
# What a CUDA build loads at run time, copied from the toolkit into the zip.
CUDA_RUNTIME = ["cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll", "nvJitLink_*.dll"]
MSVC_RUNTIME = ["msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "vcomp140.dll"]


def say(line):
    print("[runtime] " + line, flush=True)


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(cmd, cwd=None, env=None):
    say("$ " + (cmd if isinstance(cmd, str) else " ".join(cmd)))
    subprocess.run(cmd, cwd=cwd, env=env, check=True, shell=isinstance(cmd, str))


# --- the source ----------------------------------------------------------------------------------

def source(opts):
    dest = os.path.abspath(opts.dest)
    if not os.path.isdir(os.path.join(dest, ".git")):
        os.makedirs(dest, exist_ok=True)
        run(["git", "init", "-q", dest])
        run(["git", "-C", dest, "remote", "add", "origin", UPSTREAM])
    # The PR's head is fetched by hash: the branch may move (a rebase) and the pin must not.
    run(["git", "-C", dest, "fetch", "--depth", "1", "origin", opts.commit])
    run(["git", "-C", dest, "checkout", "-q", "FETCH_HEAD"])
    say("source at %s (%s)" % (dest, opts.commit))


# --- the CUDA toolkit ----------------------------------------------------------------------------

def cuda(opts):
    """Assemble a toolkit root from NVIDIA's redistributable archives -- no installer, no admin."""
    dest = os.path.abspath(opts.dest)
    cache = os.path.join(dest, "_archives")
    os.makedirs(cache, exist_ok=True)
    with urllib.request.urlopen(CUDA_REDIST + "redistrib_%s.json" % opts.version) as answer:
        manifest = json.load(answer)
    for key in CUDA_PARTS:
        entry = manifest.get(key, {}).get("windows-x86_64")
        if not entry:
            continue
        archive = os.path.join(cache, os.path.basename(entry["relative_path"]))
        if not os.path.exists(archive) or sha256_of(archive) != entry["sha256"]:
            say("fetching %s" % entry["relative_path"])
            urllib.request.urlretrieve(CUDA_REDIST + entry["relative_path"], archive)
        if sha256_of(archive) != entry["sha256"]:
            raise SystemExit("%s does not hash to NVIDIA's sha256" % archive)
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                # Every archive is <name>-archive/{bin,include,lib,...}; merged into one root.
                parts = info.filename.split("/", 1)
                if len(parts) < 2 or not parts[1] or info.is_dir():
                    continue
                target = os.path.join(dest, *parts[1].split("/"))
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with z.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
    say("CUDA %s at %s" % (opts.version, dest))


# --- the build -----------------------------------------------------------------------------------

def vcvars(version):
    """(root, vcvars64.bat) of the newest Visual Studio in `version` with the C++ tools, or None off
    Windows. Visual Studio 2022 by default: the one every CUDA toolkit here lists as a host compiler."""
    if os.name != "nt":
        return None
    vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                           "Microsoft Visual Studio", "Installer", "vswhere.exe")
    root = subprocess.run([vswhere, "-latest", "-products", "*", "-version", version, "-requires",
                           "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property",
                           "installationPath"], capture_output=True, text=True, check=True).stdout.strip()
    bat = os.path.join(root, "VC", "Auxiliary", "Build", "vcvars64.bat")
    if not root or not os.path.exists(bat):
        raise SystemExit("no Visual Studio %s with the C++ tools (vswhere found %r)" % (version, root))
    return root, bat


def in_msvc(command, vs):
    """One cmd line run inside the MSVC environment. PATH is left as vcvars made it: a %PATH% on
    the same line would be expanded before vcvars ran, and put back the PATH without cl on it."""
    return 'call "%s" >nul && %s' % (vs[1], command)


def vs_tool(vs, *parts):
    """A program Visual Studio carries (its CMake, its Ninja), by full path."""
    return os.path.join(vs[0], "Common7", "IDE", "CommonExtensions", "Microsoft", "CMake", *parts)


def build(opts):
    src = os.path.abspath(opts.source)
    out = os.path.abspath(opts.out)
    flavour = opts.flavour
    work = os.path.join(src, "build-runtime-" + flavour)
    commit = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    defines = COMMON + FLAVOURS[flavour]
    env = dict(os.environ)
    if flavour == "cuda":
        cuda_root = os.path.abspath(opts.cuda_root or env.get("CUDA_PATH", ""))
        if not os.path.isdir(cuda_root):
            raise SystemExit("--cuda-root (or CUDA_PATH) must name a toolkit root")
        nvcc = os.path.join(cuda_root, "bin", "nvcc.exe" if os.name == "nt" else "nvcc")
        defines += ['-DCUDAToolkit_ROOT=%s' % cuda_root, '-DCMAKE_CUDA_COMPILER=%s' % nvcc]
        env["CUDA_PATH"] = cuda_root
        env["PATH"] = os.path.join(cuda_root, "bin") + os.pathsep + env["PATH"]

    vs = vcvars(opts.vs_version)
    cmake = "cmake"
    if vs:
        # MSVC and nothing else: a MinGW gcc further up PATH would otherwise be taken whenever the
        # Visual Studio environment failed to load, and build a program that needs its runtime.
        cmake = '"%s"' % vs_tool(vs, "CMake", "bin", "cmake.exe")
        defines += ["-DCMAKE_C_COMPILER=cl", "-DCMAKE_CXX_COMPILER=cl",
                    "-DCMAKE_MAKE_PROGRAM=%s" % vs_tool(vs, "Ninja", "ninja.exe")]
    configure = '%s -S "%s" -B "%s" -G Ninja %s' % (cmake, src, work, " ".join('"%s"' % d for d in defines))
    jobs = opts.jobs or os.cpu_count() or 4
    compile_ = '%s --build "%s" --target llama-server -j %d' % (cmake, work, jobs)
    if vs:
        run(in_msvc(configure, vs), env=env)
        run(in_msvc(compile_, vs), env=env)
    else:
        run(configure, env=env)
        run(compile_, env=env)

    os_word, arch = ("win", "x64") if os.name == "nt" else ("macos", "arm64" if platform.machine() == "arm64" else "x64")
    name = "%s-%s-%s-%s-%s" % (PROGRAM, commit[:7], os_word, flavour, arch)
    stage = os.path.join(out, name)
    if os.path.exists(stage):
        shutil.rmtree(stage)
    os.makedirs(os.path.join(stage, "licenses"))
    bin_dir = os.path.join(work, "bin")
    exe = "llama-server.exe" if os.name == "nt" else "llama-server"
    for path in glob.glob(os.path.join(bin_dir, "*")):
        leaf = os.path.basename(path)
        # The server and every library it loads; not the other tools the tree built on the way.
        if leaf == exe or leaf.endswith((".dll", ".dylib", ".so")):
            shutil.copy2(path, stage)
    if flavour == "cuda":
        for pattern in CUDA_RUNTIME:
            for folder in ("bin", os.path.join("bin", "x64"), "lib"):
                for path in glob.glob(os.path.join(env["CUDA_PATH"], folder, pattern)):
                    shutil.copy2(path, stage)
    if vs:
        redist = sorted(glob.glob(os.path.join(vs[0], "VC", "Redist", "MSVC", "14.*", "x64")))
        for leaf in MSVC_RUNTIME:
            found = [p for r in redist for p in glob.glob(os.path.join(r, "Microsoft.VC*", leaf))]
            if found:
                shutil.copy2(found[-1], stage)
    shutil.copy2(os.path.join(src, "LICENSE"), os.path.join(stage, "licenses", "LICENSE-llama.cpp"))
    for path in glob.glob(os.path.join(src, "licenses", "*")):
        shutil.copy2(path, os.path.join(stage, "licenses"))
    # The code vendored under vendor/ is compiled into the server (cpp-httplib into it, the hashes
    # into mtmd), and its notices sit beside it rather than in licenses/: each is carried by the
    # name of the folder it came from. The pin embeds none of them in the program.
    for path in glob.glob(os.path.join(src, "vendor", "**", "LICENSE*"), recursive=True):
        folder = os.path.basename(os.path.dirname(path))
        shutil.copy2(path, os.path.join(stage, "licenses", "LICENSE-" + folder))
    if flavour == "cuda":
        # NVIDIA's runtime libraries ride in the zip; every redistributable archive carries the
        # CUDA EULA at its root, which the toolkit assembled by `cuda` keeps as LICENSE.
        shutil.copy2(os.path.join(env["CUDA_PATH"], "LICENSE"), os.path.join(stage, "licenses", "LICENSE-CUDA"))
    with open(os.path.join(stage, "licenses", "SOURCE.txt"), "w", encoding="utf-8") as fh:
        fh.write("llama.cpp %s (%s, PR #26603 head)\nbuilt by tools/runtime/build.py, flavour %s\n"
                 % (commit, UPSTREAM, flavour))

    zip_path = os.path.join(out, name + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for root_dir, _, files in os.walk(stage):
            for leaf in sorted(files):
                full = os.path.join(root_dir, leaf)
                rel = os.path.relpath(full, stage).replace(os.sep, "/")
                info = zipfile.ZipInfo.from_file(full, rel)
                info.compress_type = zipfile.ZIP_DEFLATED
                if rel == exe:
                    info.external_attr = (0o100755 << 16)  # executable where the unpacker honours it
                with open(full, "rb") as fh:
                    z.writestr(info, fh.read())
    shutil.rmtree(stage)
    write_entry(zip_path, flavour, commit)


def unpacked_bytes(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        return sum(i.file_size for i in z.infolist())


def write_entry(zip_path, flavour, commit):
    leaf = os.path.basename(zip_path)
    name = leaf[:-4]
    os_word = "windows" if "-win-" in name else "macos"
    arch = name.rsplit("-", 1)[-1]
    tag = "%s-%s" % (PROGRAM, commit[:7])
    entry = {
        "id": name,
        "name": "Speech server (%s)" % {"cpu": "CPU", "cuda": "CUDA", "vulkan": "Vulkan", "metal": "Metal"}[flavour],
        "program": PROGRAM,
        "build": flavour,
        "os": os_word,
        "arch": arch,
        "parts": [{"url": "%s/%s/%s" % (RELEASES, tag, leaf), "sha256": sha256_of(zip_path),
                   "bytes": os.path.getsize(zip_path)}],
        "unpack": "zip",
        "unpackedBytes": unpacked_bytes(zip_path),
        "exe": "llama-server.exe" if os_word == "windows" else "llama-server",
        "licence": {"name": "MIT", "url": "https://github.com/ggml-org/llama.cpp/blob/%s/LICENSE" % commit},
        "minGameVersion": 1,
    }
    with open(zip_path + ".json", "w", encoding="utf-8") as fh:
        json.dump(entry, fh, indent=2)
        fh.write("\n")
    say("%s  %d bytes  sha256 %s" % (zip_path, entry["parts"][0]["bytes"], entry["parts"][0]["sha256"]))
    print(json.dumps(entry, indent=2))


def entry(opts):
    commit = opts.commit
    write_entry(os.path.abspath(opts.zip), opts.flavour, commit)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("source")
    s.add_argument("--dest", required=True)
    s.add_argument("--commit", default=PIN)
    c = sub.add_parser("cuda")
    c.add_argument("--version", required=True, help="a redistrib_<version>.json NVIDIA publishes, e.g. 12.8.1")
    c.add_argument("--dest", required=True)
    b = sub.add_parser("build")
    b.add_argument("--source", required=True)
    b.add_argument("--flavour", choices=sorted(FLAVOURS), required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--cuda-root")
    b.add_argument("--jobs", type=int, default=0)
    b.add_argument("--vs-version", default="[17.0,18.0)",
                   help="the Visual Studio range vswhere picks from (Windows); 2022 by default")
    e = sub.add_parser("entry")
    e.add_argument("zip")
    e.add_argument("--flavour", choices=sorted(FLAVOURS), required=True)
    e.add_argument("--commit", default=PIN)
    opts = ap.parse_args(argv)
    {"source": source, "cuda": cuda, "build": build, "entry": entry}[opts.verb](opts)


if __name__ == "__main__":
    main(sys.argv[1:])
