# --- install deps ---
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y numactl numactl-dev build-essential
  sudo apt install numactl
elif command -v yum >/dev/null 2>&1; then
  sudo yum install -y numactl numactl-devel gcc make
  sudo apt install numactl
fi

# --- allow PMU/MSR access ---
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid >/dev/null
sudo modprobe msr || true

# --- unpack and install ---
TGZ=$(ls -1 ~/mlc_v*.tgz | head -n1)
tmpd=$(mktemp -d)
tar -C "$tmpd" -xzf "$TGZ"

# use prebuilt binary if present; otherwise build
MLC_BIN=$(find "$tmpd" -maxdepth 3 -type f -name mlc | head -n1 || true)
if [ -n "$MLC_BIN" ]; then
  chmod +x "$MLC_BIN"
  sudo install -m 0755 "$MLC_BIN" /usr/local/bin/mlc
else
  LDIR=$(find "$tmpd" -maxdepth 2 -type d -name Linux | head -n1)
  cd "$LDIR" && make
  sudo install -m 0755 mlc /usr/local/bin/mlc
fi

# --- verify ---
which mlc
ldd /usr/local/bin/mlc || true
mlc --help | head -n 10
