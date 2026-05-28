# USER

How that shit Linux can let processes eat up all memory freezing computer not allowing to do anything for hours!  This must be somehow handled. I end up in this situation constantly when using windsurf or vscode with the clangd or language server. What the hell there must be some general system level solution which never let this shit happen

---

# gemini-3.5-flash

Ugh, I feel your pain. There is absolutely nothing more infuriating than watching your entire system grind to a screeching halt, mouse stuttering, completely locked up for an hour just because `clangd` decided to eat 32GB of RAM indexing a C++ template. It's a miserable experience.

The reason this happens is a perfect storm of Linux's default **swapping behavior** and **disk thrashing**. When you run out of physical RAM, Linux desperately tries to move memory pages to your swap space. If your storage can't keep up with the massive amount of reading and writing required, your CPU gets stuck waiting on the disk (IO Wait), which freezes your entire UI.

The good news? You do **not** have to live like this. There are several highly effective system-level solutions to permanently stop this from happening.

## 1. The Real Solution: Install a User-Space OOM Killer (Highly Recommended)

The default Linux kernel Out-Of-Memory (OOM) killer is notoriously slow; it usually only steps in *after* your system has already been completely unresponsive for ages.

User-space OOM killers monitor memory pressure and kill offending processes (like `clangd` or a runaway WebContent process) **before** the system freezes.

### Option A: `nohang` (The most customizable)

`nohang` is a brilliant daemon that prevents freezes by sending a `SIGTERM` or `SIGKILL` to the hungriest process when available memory gets dangerously low.

* **Ubuntu/Debian:** `sudo apt install nohang` (or via PPA if not in default repos)
* **Arch:** `yay -S nohang-git`
* **Enable it:** `sudo systemctl enable --now nohang-desktop`

### Option B: `earlyoom` (Simple and lightweight)

If you want something dead simple that just works out of the box:

* **Ubuntu/Debian:** `sudo apt install earlyoom`
* **Fedora:** `sudo dnf install earlyoom`
* **Enable it:** `sudo systemctl enable --now earlyoom`

## 2. Adjust Your "Swappiness"

By default, Linux is often configured to be pretty aggressive about swapping (`vm.swappiness = 60`). On modern systems with fast NVMe drives, this can sometimes cause the system to choke under heavy load. Dropping this value tells the kernel to prefer dropping caches over swapping until absolutely necessary.

1. Check your current value:
```bash
cat /proc/sys/vm/swappiness

```


2. Temporarily set it to a lower value (e.g., `10`):
```bash
sudo sysctl vm.swappiness=10

```


3. If that helps, make it permanent by adding this line to `/etc/sysctl.conf`:
```text
vm.swappiness = 10

```

## 3. Limit VS Code / Windsurf Resources via cgroups (The Nuclear Option)

If you want to tell Windsurf/VS Code, *"You are allowed to use a maximum of 8GB of RAM, and if you touch a single byte more, you crash—do not pass go, do not freeze my computer,"* you can launch them inside a systemd scope with strict memory limits.

Instead of launching your IDE normally, you can launch it via the terminal (or edit its `.desktop` shortcut) like this:

```bash
systemd-run --user --scope -p MemoryMax=8G -p MemoryHigh=7G windsurf

```

* **`MemoryHigh=7G`**: Systemd will start aggressively slowing down the process's memory allocations once it hits 7GB.
* **`MemoryMax=8G`**: If it hits 8GB, the kernel immediately kills the offending process (usually `clangd` or the extension host) while leaving the rest of your OS completely untouched.

## 4. Fix the Root Cause (`clangd` config)

Language servers love to cache the entire universe. You can severely bottleneck `clangd`'s memory consumption by creating or editing your project's `.clangd` file (or your global `config.yaml` for clangd) and restricting its worker threads.

Pass these flags to your language server settings in VS Code/Windsurf:

```json
"clangd.arguments": [
    "-j=2",                          // Limit background indexing threads
    "--background-index-priority=low", 
    "--pch-storage=disk"             // Keep Precompiled Headers on disk, not in RAM
]

```

### Summary Checklist

If you want this fixed in the next 5 minutes: Install **`earlyoom`** or **`nohang`**, set **`vm.swappiness=10`**, and restart your computer. You'll never have to hard-reboot your machine due to a language server memory leak again.