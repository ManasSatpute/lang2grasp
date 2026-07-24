#!/bin/bash
# Live view of a MuJoCo rollout running on CSF, from your own machine, via a VNC
# desktop on the compute node. robosuite's on-screen GLFW viewer (render_mode=
# "human") draws into a virtual display that only listens on the node's loopback
# interface (127.0.0.1) -- you reach it by SSH-tunnelling through the login node,
# never exposed on the open network.
#
# Usage (from the repo root):
#   sbatch --export=ALL,RUN_DIR=/scratch/$USER/lang2grasp_runs/lift_123_s0 \
#       src/slurm/rollout_vnc.slurm
#   tail -f logs/lift_rollout_vnc_<jobid>.out   # prints the tunnel command + password
#
# CAVEAT -- this is best-effort, not verified against your actual CSF3 allocation.
# It needs *some* VNC server on the compute node, tried in this order:
#   1. `vncserver` (TurboVNC/TigerVNC) -- via a module if one exists, but a
#      module isn't required: on CSF3 it's a plain OS package already on $PATH
#   2. `Xvfb` + `x11vnc`, if both are already on $PATH
# If neither exists, check CSF3's own docs for a "VNC" / "remote desktop" /
# "interactive GUI session" page -- most HPC centres document one, and running
# the `python -m rl.rollout ... --render` command at the bottom of this script
# inside *their* documented VNC session is more reliable than this auto-detect.
# The `vncserver` non-interactive password setup below also varies by TurboVNC
# version; if it hangs or errors, set the password interactively once per node
# instead (`vncpasswd`) and drop the `< <(printf ...)` redirection.

#SBATCH --job-name=lift_rollout_vnc
#SBATCH --partition=gpuL              # <-- EDIT if your allocation's GPU partition differs
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
source "${SLURM_SUBMIT_DIR}/src/slurm/env.sh"

if [ -z "${RUN_DIR:-}" ]; then
  echo "RUN_DIR is required, e.g.:" >&2
  echo "  sbatch --export=ALL,RUN_DIR=/scratch/\$USER/lang2grasp_runs/lift_123_s0 src/slurm/rollout_vnc.slurm" >&2
  exit 1
fi

EPISODES="${EPISODES:-1000}"        # large on purpose -- keeps the viewer alive until you
                                     # disconnect or the job's time limit ends it
CONNECT_WAIT="${CONNECT_WAIT:-90}"  # seconds to set up the tunnel + connect before rollout starts

# Pick a display/port derived from the job ID so two rollout_vnc jobs landing on
# the same (shared) node don't collide.
DISP_NUM=$(( (SLURM_JOB_ID % 50) + 50 ))
VNC_PORT=$(( 5900 + DISP_NUM ))
export DISPLAY=":${DISP_NUM}"

# The virtual display behind the VNC session has no real GPU attached, so force
# software GL -- MuJoCo's GLFW viewer still needs *a* context to draw into.
export LIBGL_ALWAYS_SOFTWARE=1
export MUJOCO_GL=glfw

VNC_PASS=$(head -c 12 /dev/urandom | base64 | tr -dc 'a-zA-Z0-9' | head -c 10)

start_turbovnc() {
  # A module isn't required -- some systems (e.g. CSF3) ship `vncserver` as a
  # plain OS package with no module to load, so a failed `module load` here
  # isn't fatal on its own; only bail if `vncserver` is nowhere on PATH either.
  module load turbovnc 2>/dev/null || module load tigervnc 2>/dev/null || true
  command -v vncserver >/dev/null 2>&1 || return 1
  vncserver "$DISPLAY" -geometry 1024x768 -depth 24 -localhost -SecurityTypes VncAuth \
      < <(printf '%s\n%s\nn\n' "$VNC_PASS" "$VNC_PASS")
}

start_xvfb_x11vnc() {
  command -v Xvfb >/dev/null 2>&1 && command -v x11vnc >/dev/null 2>&1 || return 1
  Xvfb "$DISPLAY" -screen 0 1024x768x24 &
  sleep 2
  x11vnc -display "$DISPLAY" -rfbport "$VNC_PORT" -localhost -forever -shared \
      -passwd "$VNC_PASS" -bg -o "logs/x11vnc_${SLURM_JOB_ID}.log"
}

if start_turbovnc; then
  echo "VNC backend: TurboVNC/TigerVNC"
elif start_xvfb_x11vnc; then
  echo "VNC backend: Xvfb + x11vnc"
else
  echo "No VNC server found (tried module load turbovnc/tigervnc, and Xvfb+x11vnc on PATH)." >&2
  echo "Check 'module avail vnc' on a CSF3 login node, or CSF3's GUI/remote-desktop docs." >&2
  exit 1
fi

NODE=$(hostname)
echo "=================================================================="
echo "VNC ready: ${NODE}:${VNC_PORT} (display ${DISPLAY}), password: ${VNC_PASS}"
echo "From your machine (Windows PowerShell or a terminal), tunnel in:"
echo "  ssh -L ${VNC_PORT}:localhost:${VNC_PORT} -J <you>@<csf3-login-host> <you>@${NODE}"
echo "Then point a VNC viewer (e.g. TigerVNC Viewer) at localhost:${VNC_PORT}."
echo "This node must accept that second hop while your job is running -- if it's"
echo "refused, CSF3 may not allow direct ssh to compute nodes; ask support."
echo "You have ~${CONNECT_WAIT}s to connect before the rollout starts."
echo "=================================================================="

sleep "$CONNECT_WAIT"

python -m rl.rollout --run-dir "$RUN_DIR" --episodes "$EPISODES" --render
