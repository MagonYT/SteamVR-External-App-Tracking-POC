import tkinter as tk
import threading
import time
from typing import Dict, Tuple, Optional

try:
    import openvr
except Exception as e:
    openvr = None
    raise RuntimeError("openvr not available. install with: pip install openvr") from e

# ---------- config ----------
METERS_TO_PIXELS = 100
ARROW_LENGTH_M = 0.30
TRIGGER_THRESHOLD = 0.30
GRIP_THRESHOLD = 0.30
THUMBTOUCH_THRESHOLD = 0.01
ARROW_FLIP_SIGN = -1.0
POLL_INTERVAL = 0.03

# ---------- exceptions ----------
class SteamVRNotRunningError(RuntimeError):
    pass

# ---------- helpers ----------
def axis_value_safe(axis) -> float:
    if axis is None:
        return 0.0
    try:
        if hasattr(axis, "x"):
            return float(axis.x)
        return float(axis[0])
    except (AttributeError, TypeError, IndexError):
        return 0.0

def detect_fingers_approx(state) -> list[str]:
    if not state or not hasattr(state, "rAxis"):
        return []
    fingers = []
    axes = list(getattr(state, "rAxis", []))

    # index
    trigger_val = axis_value_safe(axes[1]) if len(axes) > 1 else 0.0
    trigger_pressed = bool(getattr(state, "ulButtonPressed", 0) & openvr.ButtonMask.Trigger)
    trigger_touched = bool(getattr(state, "ulButtonTouched", 0) & openvr.ButtonMask.Trigger)
    if trigger_val >= TRIGGER_THRESHOLD or trigger_pressed or trigger_touched:
        fingers.append("Index Finger")

    # thumb
    thumb = bool(getattr(state, "ulButtonTouched", 0) & openvr.ButtonMask.Touchpad)
    if not thumb and axes:
        thumb = abs(axis_value_safe(axes[0])) > THUMBTOUCH_THRESHOLD
    if thumb:
        fingers.append("Thumb")

    # middle/ring/pinky
    mapped = False
    finger_names = ["Middle Finger", "Ring Finger", "Pinky Finger"]
    for idx, name in zip([2, 3, 4], finger_names):
        if len(axes) > idx and axis_value_safe(axes[idx]) >= GRIP_THRESHOLD:
            fingers.append(name)
            mapped = True
    if not mapped:
        grip_val = axis_value_safe(axes[2]) if len(axes) > 2 else 0.0
        grip_pressed = bool(getattr(state, "ulButtonPressed", 0) & openvr.ButtonMask.Grip)
        if grip_val >= GRIP_THRESHOLD or grip_pressed:
            fingers.extend(finger_names)

    # dedupe
    seen, out = set(), []
    for f in fingers:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out

def format_height_ft_in(y_m: float) -> str:
    cm = y_m * 100.0
    total_inches = y_m * 39.3701
    feet, inches = divmod(int(round(total_inches)), 12)
    return f"{cm:.1f} cm / {feet} ft {inches} in"

def forward_from_matrix(m) -> Tuple[float, float, float]:
    try:
        fx, fy, fz = float(m[2][0]), float(m[2][1]), float(m[2][2])
        return (fx * ARROW_FLIP_SIGN, fy * ARROW_FLIP_SIGN, fz * ARROW_FLIP_SIGN)
    except Exception:
        return (0.0, 0.0, 1.0)

def arrow_delta_from_forward(forward, length_m=ARROW_LENGTH_M, scale=METERS_TO_PIXELS):
    fx, _, fz = forward
    dx = fx * length_m * scale
    dy = -fz * length_m * scale
    return dx, dy

# ---------- backend ----------
class VRBackend:
    def __init__(self):
        if openvr is None:
            raise RuntimeError("openvr not imported")
        try:
            openvr.init(openvr.VRApplication_Scene)
        except Exception as e:
            raise SteamVRNotRunningError("failed to init openvr (is SteamVR running?)") from e
        try:
            self.vr_system = openvr.VRSystem()
        except Exception as e:
            raise RuntimeError("could not obtain VRSystem from openvr") from e

        self.devices: Dict[int, Dict] = {}
        self.raw_states: Dict[int, Dict] = {}
        self.left_fingers = "None"
        self.right_fingers = "None"

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        try:
            openvr.shutdown()
        except Exception:
            pass

    def _poll_loop(self):
        while self._running:
            try:
                poses = self.vr_system.getDeviceToAbsoluteTrackingPose(
                    openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
                )
            except Exception:
                try:
                    poses = openvr.VRCompositor().waitGetPoses()[0]
                except Exception:
                    poses = []

            with self._lock:
                present = set()
                for idx, pose in enumerate(poses):
                    if not pose:
                        continue
                    connected = getattr(pose, "bDeviceIsConnected", False) or getattr(pose, "bPoseIsValid", False)
                    if not connected:
                        self.devices.pop(idx, None)
                        self.raw_states.pop(idx, None)
                        continue
                    try:
                        device_class = self.vr_system.getTrackedDeviceClass(idx)
                        m = pose.mDeviceToAbsoluteTracking
                        x, y, z = float(m[0][3]), float(m[1][3]), float(m[2][3])
                    except Exception:
                        continue

                    role = None
                    if device_class == openvr.TrackedDeviceClass_Controller:
                        try:
                            role_id = self.vr_system.getControllerRoleForTrackedDeviceIndex(idx)
                            role = "Left" if role_id == openvr.TrackedControllerRole_LeftHand else "Right"
                        except Exception:
                            pass

                    forward = forward_from_matrix(m)
                    self.devices[idx] = {"x": x, "y": y, "z": z,
                                         "class": device_class, "role": role, "forward": forward}
                    present.add(idx)

                    # controller state
                    if device_class == openvr.TrackedDeviceClass_Controller:
                        state = None
                        try:
                            res = self.vr_system.getControllerStateAndPose(idx)
                            state = res[0] if isinstance(res, tuple) else res
                        except Exception:
                            try:
                                res2 = self.vr_system.getControllerState(idx)
                                state = res2[0] if isinstance(res2, tuple) else res2
                            except Exception:
                                pass
                        if state and hasattr(state, "rAxis"):
                            axes = [(axis_value_safe(a), getattr(a, "y", 0.0)) for a in state.rAxis]
                            pressed = int(getattr(state, "ulButtonPressed", 0))
                            touched = int(getattr(state, "ulButtonTouched", 0))
                            self.raw_states[idx] = {"axes": axes, "pressed": pressed, "touched": touched}
                            fingers = detect_fingers_approx(state)
                            if role == "Left":
                                self.left_fingers = ", ".join(fingers) if fingers else "None"
                            else:
                                self.right_fingers = ", ".join(fingers) if fingers else "None"
                        else:
                            self.raw_states.pop(idx, None)
                            if role == "Left":
                                self.left_fingers = "None"
                            else:
                                self.right_fingers = "None"

                for old in list(self.devices.keys()):
                    if old not in present:
                        self.devices.pop(old, None)
                        self.raw_states.pop(old, None)

            time.sleep(POLL_INTERVAL)

    def snapshot(self):
        with self._lock:
            return {"devices": dict(self.devices),
                    "raw_states": dict(self.raw_states),
                    "left_fingers": self.left_fingers,
                    "right_fingers": self.right_fingers}


# ---------- gui ----------
class VRGui:
    def __init__(self, root: tk.Tk, backend: VRBackend):
        self.root, self.backend = root, backend
        self.root.title("SteamVR External App Tracking Proof of Concept")

        # toggles
        self.show_grid = tk.BooleanVar(master=root, value=True)
        self.show_labels = tk.BooleanVar(master=root, value=True)
        self.show_height = tk.BooleanVar(master=root, value=True)
        self.show_fingers = tk.BooleanVar(master=root, value=True)
        self.show_arrows = tk.BooleanVar(master=root, value=True)
        self.show_debug = tk.BooleanVar(master=root, value=False)

        menubar = tk.Menu(root)
        view = tk.Menu(menubar, tearoff=0)
        for lbl, var in [("Grid", self.show_grid), ("Labels", self.show_labels),
                         ("Height", self.show_height), ("Fingers", self.show_fingers),
                         ("Arrows", self.show_arrows), ("Debug", self.show_debug)]:
            view.add_checkbutton(label=f"Show {lbl}", variable=var)
        menubar.add_cascade(label="View", menu=view)
        root.config(menu=menubar)

        self.canvas = tk.Canvas(root, width=1000, height=720, bg="#0b0d0f")
        self.canvas.pack(fill="both", expand=True)

        self.canvas_items: Dict[int, Dict[str, int]] = {}
        self._running = True
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # persistent grid
        self._draw_grid()
        self.root.bind("<Configure>", lambda e: self._draw_grid(force=True))

        self._tick()

    def _draw_grid(self, force=False):
        w, h = max(200, self.canvas.winfo_width()), max(200, self.canvas.winfo_height())
        if force:
            self.canvas.delete("__grid__")
        if not self.show_grid.get():
            self.canvas.delete("__grid__")
            return
        self.canvas.delete("__grid__")
        spacing = 50
        for gx in range(0, w + spacing, spacing):
            self.canvas.create_line(gx, 0, gx, h, fill="#171a1c", tags="__grid__")
        for gy in range(0, h + spacing, spacing):
            self.canvas.create_line(0, gy, w, gy, fill="#171a1c", tags="__grid__")
        self.canvas.create_oval(w/2 - 6, h/2 - 6, w/2 + 6, h/2 + 6,
                                fill="#9ccf8a", outline="", tags="__grid__")
        self.canvas.tag_lower("__grid__")

    def _ensure_items_for(self, idx: int):
        if idx in self.canvas_items:
            return
        shape = self.canvas.create_oval(0, 0, 0, 0, fill="white")
        label = self.canvas.create_text(0, 0, text="", anchor="w",
                                        fill="white", font=("Segoe UI", 9))
        arrow = self.canvas.create_line(0, 0, 0, 0, fill="white",
                                        width=2, arrow=tk.LAST)
        self.canvas_items[idx] = {"shape": shape, "label": label, "arrow": arrow}

    def _remove_items_for(self, idx: int):
        items = self.canvas_items.pop(idx, None)
        if items:
            for k in items.values():
                self.canvas.delete(k)

    def _tick(self):
        try:
            snap = self.backend.snapshot()
        except SteamVRNotRunningError:
            self.canvas.delete("all")
            self.canvas.create_text(20, 20, anchor="nw",
                                    text="steamvr not running or failed to init.",
                                    fill="red", font=("Segoe UI", 14))
            return

        devices, raw_states = snap["devices"], snap["raw_states"]
        left_fingers, right_fingers = snap["left_fingers"], snap["right_fingers"]
        w, h = max(200, self.canvas.winfo_width()), max(200, self.canvas.winfo_height())
        self._draw_grid()

        present = set()
        for idx, info in devices.items():
            self._ensure_items_for(idx)
            present.add(idx)

            x, y, z = info["x"], info["y"], info["z"]
            dev_class, role = info["class"], info.get("role")
            forward = info.get("forward", (0.0, 0.0, 1.0))
            sx, sy = w/2 + x * METERS_TO_PIXELS, h/2 - z * METERS_TO_PIXELS

            # color + label
            if dev_class == openvr.TrackedDeviceClass_HMD:
                fill, txt, size = "#ff5555", "HMD", 12
            elif dev_class == openvr.TrackedDeviceClass_Controller:
                fill = "#4da6ff" if role == "Left" else "#00ffd1"
                txt, size = f"Controller ({role or '??'})", 10
            elif dev_class == openvr.TrackedDeviceClass_GenericTracker:
                fill, txt, size = "#ffd166", "Tracker", 8
            elif dev_class == openvr.TrackedDeviceClass_TrackingReference:
                fill, txt, size = "#4cff4c", "Base Station", 12
            else:
                fill, txt, size = "white", "Device", 8

            items = self.canvas_items[idx]
            self.canvas.coords(items["shape"], sx-size, sy-size, sx+size, sy+size)
            self.canvas.itemconfig(items["shape"], fill=fill)

            label_text = txt
            if dev_class == openvr.TrackedDeviceClass_HMD and self.show_height.get():
                label_text += f" | {format_height_ft_in(y)}"
            if self.show_labels.get():
                self.canvas.itemconfig(items["label"], text=label_text, fill="white")
                self.canvas.coords(items["label"], sx + size + 4, sy)
                self.canvas.itemconfigure(items["label"], state="normal")
            else:
                self.canvas.itemconfigure(items["label"], state="hidden")

            if dev_class == openvr.TrackedDeviceClass_Controller and self.show_arrows.get():
                dx, dy = arrow_delta_from_forward(forward)
                self.canvas.coords(items["arrow"], sx, sy, sx+dx, sy+dy)
                self.canvas.itemconfig(items["arrow"], fill=fill, state="normal")
            else:
                self.canvas.itemconfig(items["arrow"], state="hidden")

        for old in list(self.canvas_items.keys()):
            if old not in present:
                self._remove_items_for(old)

        # fingers overlay
        self.canvas.delete("__fingers__")
        if self.show_fingers.get():
            self.canvas.create_text(w-12, 12, anchor="ne",
                                    text=f"Left Controller: {left_fingers}",
                                    fill="#cce", font=("Segoe UI", 10),
                                    tags="__fingers__")
            self.canvas.create_text(w-12, 30, anchor="ne",
                                    text=f"Right Controller: {right_fingers}",
                                    fill="#cce", font=("Segoe UI", 10),
                                    tags="__fingers__")

        # debug overlay
        self.canvas.delete("__debug__")
        if self.show_debug.get():
            lines = ["DEBUG: raw controller states"]
            for di in sorted(raw_states.keys()):
                rs = raw_states[di]
                axes_str = ", ".join(f"{a[0]:.2f}" for a in rs.get("axes", [])[:5])
                lines.append(f"Device {di}: axes[{axes_str}] pressed={rs.get('pressed',0)} touched={rs.get('touched',0)}")
            if lines:
                # background rect
                rect_h = 14 * len(lines)
                self.canvas.create_rectangle(5, h-rect_h-5, 400, h-5, fill="#000000", outline="", tags="__debug__")
            for i, ln in enumerate(reversed(lines)):
                self.canvas.create_text(10, h-10-14*i, anchor="sw", text=ln,
                                        fill="#aaa", font=("Consolas", 9), tags="__debug__")

        if self._running:
            self.root.after(int(POLL_INTERVAL*1000), self._tick)

    def _on_close(self):
        self._running = False
        self.backend.stop()
        self.root.after(50, self.root.destroy)

# ---------- main ----------
def main():
    try:
        backend = VRBackend()
    except SteamVRNotRunningError as e:
        root = tk.Tk()
        root.title("SteamVR External App Tracking Proof of Concept")
        tk.Label(root, text=str(e), fg="red").pack(padx=20, pady=20)
        root.mainloop()
        return

    backend.start()
    root = tk.Tk()
    VRGui(root, backend)
    root.mainloop()


if __name__ == "__main__":
    main()
