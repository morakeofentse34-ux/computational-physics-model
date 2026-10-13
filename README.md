# computational-physics-model
DEVELOPMENT OF A COMPUTATIONAL PHYSICS SIMULATION PLATFORM FOR SOLVING CLASSICAL PHYSICS PROBLEMS USING NUMERICAL METHODS

"""
Computational Physics Simulation Platform
Sol Plaatje University - Mini-Capstone Project

Includes:
- ODE solvers: Euler, RK4, Leapfrog
- Projectile motion with/without drag
- Simple pendulum
- Double pendulum
- Spring-mass system
- Gravitational N-body motion
- 2D trajectory, phase-space and energy plots
- Numerical accuracy/stability comparison
"""

import tkinter as tk
from tkinter import ttk, messagebox
import re
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------- Integrators -----------------------------

def euler(f, y0, t):
    y = np.zeros((len(t), len(y0)), dtype=float)
    y[0] = y0
    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        y[i + 1] = y[i] + h * f(t[i], y[i])
    return y


def rk4(f, y0, t):
    y = np.zeros((len(t), len(y0)), dtype=float)
    y[0] = y0
    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        ti, yi = t[i], y[i]
        k1 = f(ti, yi)
        k2 = f(ti + h/2, yi + h*k1/2)
        k3 = f(ti + h/2, yi + h*k2/2)
        k4 = f(ti + h, yi + h*k3)
        y[i + 1] = yi + h*(k1 + 2*k2 + 2*k3 + k4)/6
    return y


def leapfrog_acceleration(accel, r0, v0, t):
    """Velocity-Verlet/leapfrog for second-order systems r''=a(r)."""
    r = np.zeros((len(t), len(r0)), dtype=float)
    v = np.zeros((len(t), len(v0)), dtype=float)
    r[0], v[0] = r0, v0
    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        a = accel(r[i])
        r[i + 1] = r[i] + h*v[i] + 0.5*h*h*a
        a_new = accel(r[i + 1])
        v[i + 1] = v[i] + 0.5*h*(a + a_new)
    return r, v


def solve_ivp(f, y0, t, method="RK4"):
    """Dispatch to a first-order integrator: Euler or RK4.
    Leapfrog is only meaningful for second-order systems, so it falls
    back to RK4 here with a clear note."""
    if method == "Euler":
        return euler(f, y0, t)
    return rk4(f, y0, t)  # RK4 and (fallback) Leapfrog


def solve_second_order(accel, r0, v0, t, method="RK4"):
    """Dispatch for second-order systems r'' = a(r).
    Uses leapfrog when requested; otherwise converts to first-order and
    integrates with Euler or RK4."""
    if method == "Leapfrog":
        return leapfrog_acceleration(accel, r0, v0, t)

    # Convert r''=a(r) to a first-order system [r, v]
    def f(ti, y):
        n = len(y) // 2
        r = y[:n]
        v = y[n:]
        return np.concatenate([v, accel(r)])

    y0 = np.concatenate([np.asarray(r0, dtype=float),
                         np.asarray(v0, dtype=float)])
    y = euler(f, y0, t) if method == "Euler" else rk4(f, y0, t)
    n = len(y0) // 2
    return y[:, :n], y[:, n:]


# ----------------------------- Physics models -----------------------------

G = 9.81


def projectile(v0=20.0, angle_deg=45.0, drag=0.0, t_end=5.0, dt=0.005,
               method="RK4"):
    angle = np.deg2rad(angle_deg)
    y0 = np.array([0.0, 0.0, v0*np.cos(angle), v0*np.sin(angle)])

    def f(t, y):
        x, yy, vx, vy = y
        speed = np.hypot(vx, vy)
        return np.array([vx, vy, -drag*speed*vx, -G-drag*speed*vy])

    t = np.arange(0, t_end + dt, dt)
    return t, solve_ivp(f, y0, t, method)


def pendulum(theta_deg=30.0, length=1.0, t_end=10.0, dt=0.005, damping=0.0,
             method="RK4"):
    theta0 = np.deg2rad(theta_deg)
    y0 = np.array([theta0, 0.0])

    def f(t, y):
        theta, omega = y
        return np.array([omega, -(G/length)*np.sin(theta)-damping*omega])

    t = np.arange(0, t_end + dt, dt)

    if method == "Leapfrog":
        accel = lambda th: -(G/length)*np.sin(th) - damping * 0.0  # damping not supported in this path
        # Leapfrog works on theta only; damping needs the velocity, so we
        # fall back to RK4 when damping is nonzero.
        if damping == 0.0:
            r, v = leapfrog_acceleration(accel, np.array([theta0]),
                                         np.array([0.0]), t)
            return t, np.column_stack([r[:, 0], v[:, 0]])
        # else fall through to RK4
    return t, solve_ivp(f, y0, t, method)


def double_pendulum(theta1_deg=120.0, theta2_deg=20.0, L1=1.0, L2=1.0,
                    m1=1.0, m2=1.0, t_end=10.0, dt=0.005, method="RK4"):
    # Standard double-pendulum equations.
    th1, th2 = np.deg2rad([theta1_deg, theta2_deg])
    y0 = np.array([th1, 0.0, th2, 0.0])

    def f(t, y):
        a1, w1, a2, w2 = y
        delta = a1 - a2
        den1 = (m1+m2)*L1 - m2*L1*np.cos(delta)**2
        den2 = (L2/L1)*den1

        num1 = (m2*L1*w1*w1*np.sin(delta)*np.cos(delta)
                + m2*G*np.sin(a2)*np.cos(delta)
                + m2*L2*w2*w2*np.sin(delta)
                - (m1+m2)*G*np.sin(a1))
        num2 = (-m2*L2*w2*w2*np.sin(delta)*np.cos(delta)
                + (m1+m2)*(G*np.sin(a1)*np.cos(delta)
                - L1*w1*w1*np.sin(delta) - G*np.sin(a2)))
        return np.array([w1, num1/den1, w2, num2/den2])

    t = np.arange(0, t_end + dt, dt)
    return t, solve_ivp(f, y0, t, method)


def spring_mass(m=1.0, k=10.0, x0=1.0, v0=0.0, t_end=10.0, dt=0.005,
                method="RK4"):
    def f(t, y):
        x, v = y
        return np.array([v, -(k/m)*x])

    t = np.arange(0, t_end + dt, dt)

    if method == "Leapfrog":
        accel = lambda x: -(k/m)*x
        r, v = leapfrog_acceleration(accel, np.array([x0]),
                                     np.array([v0]), t)
        return t, np.column_stack([r[:, 0], v[:, 0]])

    return t, solve_ivp(f, np.array([x0, v0]), t, method)


def n_body(n=3, t_end=10.0, dt=0.002, method="Leapfrog"):
    # Simple planar N-body example with normalized units.
    masses = np.ones(n)
    r0 = np.array([
        [-1.0, 0.0], [1.0, 0.0], [0.0, 1.4],
        [0.0, -1.4], [0.8, 0.8]
    ])[:n].astype(float)
    v0 = np.array([
        [0.0, -0.5], [0.0, 0.5], [-0.45, 0.0],
        [0.45, 0.0], [-0.25, 0.25]
    ])[:n].astype(float)

    def accel(r):
        a = np.zeros_like(r)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = r[j] - r[i]
                dist = np.linalg.norm(d) + 1e-9
                a[i] += masses[j] * d / dist**3
        return a

    t = np.arange(0, t_end + dt, dt)
    # N-body is naturally second-order; Leapfrog is the default and best choice.
    if method in ("Leapfrog", "RK4", "Euler"):
        r, v = solve_second_order(accel, r0.ravel(), v0.ravel(), t, method)
    else:
        r, v = solve_second_order(accel, r0.ravel(), v0.ravel(), t, "Leapfrog")
    return t, r.reshape(len(t), n, 2), v.reshape(len(t), n, 2), masses


# ----------------------------- Analysis -----------------------------

def energy_pendulum(data, length=1.0, mass=1.0):
    theta, omega = data[:, 0], data[:, 1]
    return 0.5*mass*(length*omega)**2 + mass*G*length*(1-np.cos(theta))


def energy_spring(data, m=1.0, k=10.0):
    x, v = data[:, 0], data[:, 1]
    return 0.5*m*v*v + 0.5*k*x*x


def energy_double_pendulum(data, L1=1.0, L2=1.0, m1=1.0, m2=1.0):
    th1, w1, th2, w2 = data[:,0], data[:,1], data[:,2], data[:,3]
    KE = (0.5*m1*(L1*w1)**2
          + 0.5*m2*((L1*w1)**2 + (L2*w2)**2
                    + 2*L1*L2*w1*w2*np.cos(th1 - th2)))
    PE = -(m1+m2)*G*L1*np.cos(th1) - m2*G*L2*np.cos(th2)
    return KE + PE


def estimate_period(t, y):
    """Estimate the oscillation period from local maxima of y(t)."""
    peaks = [i for i in range(1, len(y)-1) if y[i-1] < y[i] >= y[i+1]]
    if len(peaks) >= 2:
        return float(np.mean(np.diff(t[peaks])))
    return None


def run_comparison(model="pendulum"):
    """Compare Euler, RK4 and Leapfrog where applicable."""
    t = np.arange(0, 10.0 + 0.01, 0.01)
    if model == "pendulum":
        y0 = np.array([np.deg2rad(30), 0.0])
        def f(ti, y):
            return np.array([y[1], -(G)*np.sin(y[0])])
        e = euler(f, y0, t)
        r = rk4(f, y0, t)
        # Leapfrog on theta, omega using theta''=a(theta)
        lf_r, lf_v = leapfrog_acceleration(
            lambda q: -G*np.sin(q), np.array([y0[0]]), np.array([y0[1]]), t
        )
        return t, e, r, np.column_stack([lf_r[:,0], lf_v[:,0]])
    return None


# ----------------------------- Natural-language question answering -----------------------------
#
# Lightweight, rule-based NLP: no external AI model or internet connection
# needed. A keyword/regex parser extracts (1) which simulation the question
# is about, (2) any numeric parameters mentioned, and (3) what quantity is
# being asked for, then routes to the existing simulation functions above
# and formats the numeric result into an explained answer.

_MODEL_KEYWORDS = [
    ("double pendulum", "double_pendulum"),
    ("double-pendulum", "double_pendulum"),
    ("pendulum", "pendulum"),
    ("spring", "spring_mass"),
    ("oscillator", "spring_mass"),
    ("projectile", "projectile"),
    ("cannonball", "projectile"),
    ("launched", "projectile"),
    ("thrown", "projectile"),
    ("thrown at", "projectile"),
    ("orbit", "n_body"),
    ("n-body", "n_body"),
    ("planets", "n_body"),
    ("bodies", "n_body"),
]

# Ordered most-specific-first so that e.g. "time of flight" wins over "how long"
_INTENT_KEYWORDS = [
    ("maximum height", "height"),
    ("max height", "height"),
    ("how high", "height"),
    ("highest", "height"),
    ("time of flight", "time"),
    ("when does it land", "time"),
    ("how long", "time"),
    ("period", "period"),
    ("how far", "range"),
    ("range", "range"),
    ("distance", "range"),
    ("amplitude", "amplitude"),
    ("energy", "energy"),
    ("speed", "speed"),
    ("velocity", "speed"),
    ("land", "time"),
]

# regex fragment matching a signed float with optional scientific notation
_NUM_RE = r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?)"


def _num_near(text, keywords):
    """Return the first number found close to one of the given keywords.
    Checks both orderings: 'value unit' (e.g. '30 m/s', '40 degrees') and
    'keyword value' (e.g. 'length 2m', 'angle = 30')."""
    t = text.lower()
    for kw in keywords:
        m = re.search(rf"{_NUM_RE}\s*{re.escape(kw)}", t)
        if m:
            return float(m.group(1))
        m = re.search(rf"{re.escape(kw)}\D{{0,12}}?{_NUM_RE}", t)
        if m:
            return float(m.group(1))
    return None


def detect_model(text):
    t = text.lower()
    for kw, model in _MODEL_KEYWORDS:
        if kw in t:
            return model
    return None


def detect_intent(text):
    t = text.lower()
    for kw, intent in _INTENT_KEYWORDS:
        if kw in t:
            return intent
    return "summary"


def extract_params(text, model):
    # Only accept explicit time-unit keywords for t_end to avoid matching
    # unrelated numbers (e.g. "for a pendulum of length 2m").
    p = {"t_end": _num_near(text, ["seconds", "second", "duration", "secs"])}
    if model == "pendulum":
        p["length"] = _num_near(text, ["length", "long"]) or 1.0
        p["angle"] = _num_near(text, ["angle", "degrees", "deg", "theta"]) or 30.0
        p["damping"] = _num_near(text, ["damping"]) or 0.0
        p["t_end"] = p["t_end"] or 10.0
    elif model == "double_pendulum":
        p["theta1"] = _num_near(text, ["theta1", "first angle", "angle1", "angle"]) or 120.0
        p["theta2"] = _num_near(text, ["theta2", "second angle", "angle2"]) or 20.0
        p["t_end"] = p["t_end"] or 10.0
    elif model == "spring_mass":
        p["mass"] = _num_near(text, ["mass", "kg"]) or 1.0
        p["k"] = _num_near(text, ["spring constant", "stiffness", "constant", "n/m"]) or 10.0
        p["x0"] = _num_near(text, ["x0", "displacement", "stretch", "amplitude"]) or 1.0
        p["t_end"] = p["t_end"] or 10.0
    elif model == "projectile":
        p["v0"] = _num_near(text, ["speed", "velocity", "m/s"]) or 20.0
        p["angle"] = _num_near(text, ["angle", "degrees", "deg"]) or 45.0
        p["drag"] = _num_near(text, ["drag"]) or 0.0
        p["t_end"] = p["t_end"] or 5.0
    elif model == "n_body":
        n = _num_near(text, ["bodies", "planets", "objects"])
        p["n"] = int(n) if n else 3
        p["t_end"] = p["t_end"] or 10.0
    return p


def answer_question(text):
    """Parse a plain-English physics question, run the matching simulation,
    and return an explained natural-language answer."""
    model = detect_model(text)
    if model is None:
        return ("I couldn't tell which system you're asking about. Try "
                "mentioning 'pendulum', 'double pendulum', 'spring', "
                "'projectile', or 'n-body' / 'orbit' in your question.")

    intent = detect_intent(text)
    params = extract_params(text, model)
    lines = [f"Detected system: {model.replace('_', ' ')}",
             "Parsed parameters: " + ", ".join(f"{k}={v}" for k, v in params.items()),
             ""]

    if model == "pendulum":
        t, data = pendulum(theta_deg=params["angle"], length=params["length"],
                            t_end=params["t_end"], damping=params["damping"])
        energy = energy_pendulum(data, params["length"])
        amp_deg = float(np.rad2deg(np.max(np.abs(data[:, 0]))))
        period_num = estimate_period(t, data[:, 0])
        period_theory = 2*np.pi*np.sqrt(params["length"]/G)
        drift = 0.0 if energy[0] == 0 else (energy[-1]-energy[0])/abs(energy[0])*100

        if intent == "period":
            lines.append(f"Small-angle theory: T = 2π√(L/g) = {period_theory:.4f} s "
                         f"for L={params['length']:.3g} m.")
            if period_num:
                lines.append(f"The RK4 simulation measures T ≈ {period_num:.4f} s.")
                if params["angle"] > 20:
                    lines.append("The two differ because the small-angle formula "
                                 "breaks down at larger swing angles — the real "
                                 "period grows with amplitude.")
        elif intent == "energy":
            lines.append(f"Total mechanical energy: {energy[0]:.4f} J at t=0, "
                         f"{energy[-1]:.4f} J at t_end ({drift:+.4f}% drift, "
                         f"from RK4's local truncation error).")
        elif intent == "amplitude":
            lines.append(f"Maximum swing angle: {amp_deg:.4f} degrees.")
        else:
            lines += [f"Amplitude: {amp_deg:.4f} deg",
                      f"Small-angle period: {period_theory:.4f} s",
                      f"Numeric period: {period_num:.4f} s" if period_num else "Numeric period: n/a",
                      f"Energy drift: {drift:+.4f} %"]

    elif model == "double_pendulum":
        t, data = double_pendulum(theta1_deg=params["theta1"], theta2_deg=params["theta2"],
                                   t_end=params["t_end"])
        energy = energy_double_pendulum(data)
        amp1 = float(np.rad2deg(np.max(np.abs(data[:, 0]))))
        amp2 = float(np.rad2deg(np.max(np.abs(data[:, 2]))))
        drift = 0.0 if energy[0] == 0 else (energy[-1]-energy[0])/abs(energy[0])*100
        lines.append("This system is chaotic, so there's no simple closed-form "
                     "period — small changes in the starting angles lead to very "
                     "different trajectories.")
        lines.append(f"Max |theta1| = {amp1:.4f} deg, max |theta2| = {amp2:.4f} deg.")
        lines.append(f"Energy drift over the run: {drift:+.4f}% "
                     "(a sanity check on the integrator, not on the chaos itself).")

    elif model == "spring_mass":
        t, data = spring_mass(m=params["mass"], k=params["k"], x0=params["x0"], t_end=params["t_end"])
        energy = energy_spring(data, params["mass"], params["k"])
        amp = float(np.max(np.abs(data[:, 0])))
        period_num = estimate_period(t, data[:, 0])
        period_theory = 2*np.pi*np.sqrt(params["mass"]/params["k"])
        drift = 0.0 if energy[0] == 0 else (energy[-1]-energy[0])/abs(energy[0])*100

        if intent == "period":
            lines.append(f"Theory: T = 2π√(m/k) = 2π√({params['mass']:.3g}/{params['k']:.3g}) "
                         f"= {period_theory:.4f} s.")
            if period_num:
                lines.append(f"The simulation measures T ≈ {period_num:.4f} s — matches "
                             "theory closely, since SHM is exact for a linear spring.")
        elif intent == "energy":
            lines.append(f"Energy at t=0: {energy[0]:.4f} J, at t_end: {energy[-1]:.4f} J "
                         f"({drift:+.4f}% drift).")
        elif intent == "amplitude":
            lines.append(f"Amplitude: {amp:.4f} m.")
        else:
            lines += [f"Amplitude: {amp:.4f} m",
                      f"Theory period: {period_theory:.4f} s",
                      f"Numeric period: {period_num:.4f} s" if period_num else "Numeric period: n/a",
                      f"Energy drift: {drift:+.4f} %"]

    elif model == "projectile":
        t, data = projectile(v0=params["v0"], angle_deg=params["angle"],
                              drag=params["drag"], t_end=params["t_end"])
        land_idx = None
        for i in range(1, len(t)):
            if data[i, 1] <= 0 and data[i-1, 1] > 0:
                land_idx = i
                break
        max_height = float(np.max(data[:, 1]))

        if intent == "height":
            lines.append(f"Launched at {params['v0']:.3g} m/s, {params['angle']:.3g}°, "
                         f"the maximum height reached is {max_height:.4f} m.")
        elif intent == "range":
            if land_idx is not None:
                lines.append(f"Range: {data[land_idx, 0]:.4f} m, reached after "
                             f"{t[land_idx]:.4f} s of flight.")
            else:
                lines.append(f"It hadn't landed within the simulated {t[-1]:.2f} s — "
                             f"at that point x = {data[-1, 0]:.4f} m.")
        elif intent == "time":
            if land_idx is not None:
                lines.append(f"Time of flight: {t[land_idx]:.4f} s.")
            else:
                lines.append(f"Still in the air at t_end = {t[-1]:.2f} s — "
                             "try a longer simulated time.")
        elif intent == "speed":
            idx = land_idx if land_idx is not None else -1
            speed = float(np.hypot(data[idx, 2], data[idx, 3]))
            lines.append(f"Speed {'at landing' if land_idx is not None else 'at t_end'}: "
                         f"{speed:.4f} m/s.")
        else:
            lines.append(f"Max height: {max_height:.4f} m")
            if land_idx is not None:
                lines.append(f"Range: {data[land_idx, 0]:.4f} m, "
                             f"time of flight: {t[land_idx]:.4f} s")

    elif model == "n_body":
        n = min(max(params["n"], 2), 5)
        t, r, v, m = n_body(n=n, t_end=params["t_end"])
        kinetic = 0.5*np.sum(m[None, :, None]*v*v, axis=(1, 2))
        potential = np.zeros(len(t))
        for i in range(r.shape[1]):
            for j in range(i+1, r.shape[1]):
                dist = np.linalg.norm(r[:, j, :]-r[:, i, :], axis=1) + 1e-9
                potential -= m[i]*m[j]/dist
        total = kinetic + potential
        drift = 0.0 if total[0] == 0 else (total[-1]-total[0])/abs(total[0])*100
        lines.append(f"Simulated {n} bodies under mutual gravity (normalized units, G=1).")
        lines.append(f"Total energy drift over the run: {drift:+.4f}% "
                     "(leapfrog usually conserves energy well, but close "
                     "encounters with softened gravity can cause some drift).")
        for i in range(r.shape[1]):
            speed = float(np.linalg.norm(v[-1, i, :]))
            lines.append(f"  Body {i+1} final speed: {speed:.4f}")

    return "\n".join(lines)


# ----------------------------- GUI -----------------------------

class SimulationApp:
    def __init__(self, root):
        self.root = root
        root.title("Computational Physics Simulation Platform")
        root.geometry("1280x700")

        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True)

        sim_tab = ttk.Frame(notebook)
        notebook.add(sim_tab, text="Simulator")

        qa_tab = ttk.Frame(notebook)
        notebook.add(qa_tab, text="Ask a Physics Question")

        left = ttk.Frame(sim_tab, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="Simulation", font=("Arial", 12, "bold")).pack(pady=5)
        self.model = ttk.Combobox(
            left, values=["Projectile", "Pendulum", "Double Pendulum", "Spring-Mass", "N-Body"],
            state="readonly", width=20
        )
        self.model.current(0)
        self.model.pack(pady=5)

        # Keep references to both the Label widgets and the Entry widgets so
        # the label text can be updated dynamically when the model changes.
        self.entries = {}
        self.labels = {}
        self.param_keys = ["p1", "p2", "p3", "t_end", "dt"]
        defaults = [
            ("v0 / mass", "20"),
            ("angle / x0", "45"),
            ("parameter 3", "0"),
            ("time (s)", "10"),
            ("step dt (s)", "0.005"),
        ]
        for key, (name, value) in zip(self.param_keys, defaults):
            lbl = ttk.Label(left, text=name)
            lbl.pack(anchor="w")
            e = ttk.Entry(left, width=22)
            e.insert(0, value)
            e.pack(pady=2)
            self.labels[key] = lbl
            self.entries[key] = e

        ttk.Label(left, text="Integrator", font=("Arial", 11, "bold")).pack(pady=(15, 4))
        self.integrator = ttk.Combobox(left, values=["Euler", "RK4", "Leapfrog"], state="readonly", width=20)
        self.integrator.current(1)
        self.integrator.pack()

        ttk.Button(left, text="Run simulation", command=self.run).pack(fill=tk.X, pady=(18, 5))
        ttk.Button(left, text="Compare integrators", command=self.compare).pack(fill=tk.X, pady=5)

        ttk.Label(left, text="Plots generated:", font=("Arial", 11, "bold")).pack(pady=(20, 4))
        ttk.Label(left, text="• Trajectory / motion\n• Phase space\n• Energy vs time\n• Integrator comparison",
                  justify=tk.LEFT).pack(anchor="w")

        self.status = ttk.Label(left, text="Ready.")
        self.status.pack(pady=20)

        self.model.bind("<<ComboboxSelected>>", self.update_labels)

        # Apply the correct labels for the initially-selected model right away.
        self.update_labels()

        # ---- Results panel (numeric output, separate from the plots) ----
        right = ttk.Frame(sim_tab, padding=10)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(right, text="Numeric results", font=("Arial", 12, "bold")).pack(anchor="w")

        text_frame = ttk.Frame(right)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        self.results_text = tk.Text(
            text_frame, width=60, height=32, font=("Consolas", 10),
            wrap="word", state="disabled"
        )
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.show_results(["Run a simulation to see numeric results here."])

        # ---- Ask-a-question tab ----
        self.build_qa_tab(qa_tab)

    def build_qa_tab(self, parent):
        ttk.Label(parent, text="Ask a physics question in plain English",
                  font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(10, 2))
        ttk.Label(parent, foreground="#555555",
                  text='e.g. "What is the period of a pendulum with length 2m?"'
                  ).pack(anchor="w", padx=10, pady=(0, 10))

        entry_frame = ttk.Frame(parent)
        entry_frame.pack(fill=tk.X, padx=10)
        self.question_entry = ttk.Entry(entry_frame, font=("Arial", 11))
        self.question_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.question_entry.bind("<Return>", lambda e: self.ask_question())
        ttk.Button(entry_frame, text="Ask", command=self.ask_question).pack(side=tk.LEFT, padx=(8, 0))

        examples = [
            "What is the period of a pendulum with length 2m?",
            "How far does a projectile launched at 30 m/s at 40 degrees go?",
            "What is the amplitude of a spring with mass 2kg and spring constant 50?",
            "How chaotic is a double pendulum starting at 150 degrees?",
        ]
        ttk.Label(parent, foreground="#777777", wraplength=1100,
                  text="Examples: " + "  |  ".join(examples)
                  ).pack(anchor="w", padx=10, pady=(0, 10))

        ans_frame = ttk.Frame(parent)
        ans_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.answer_text = tk.Text(ans_frame, font=("Consolas", 11), wrap="word", state="disabled")
        ans_scroll = ttk.Scrollbar(ans_frame, orient="vertical", command=self.answer_text.yview)
        self.answer_text.configure(yscrollcommand=ans_scroll.set)
        self.answer_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ans_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def ask_question(self):
        text = self.question_entry.get().strip()
        if not text:
            return
        try:
            answer = answer_question(text)
        except Exception as ex:
            answer = f"Couldn't answer that question: {ex}"
        self.answer_text.config(state="normal")
        self.answer_text.insert(tk.END, f"Q: {text}\n{answer}\n\n{'='*70}\n\n")
        self.answer_text.see(tk.END)
        self.answer_text.config(state="disabled")

    def show_results(self, lines):
        """Replace the results panel content with the given list of lines."""
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, "\n".join(lines))
        self.results_text.config(state="disabled")

    def vals(self):
        return [float(self.entries[k].get()) for k in self.param_keys]

    def update_labels(self, *_):
        names = ["v0 / mass", "angle / x0", "parameter 3", "time (s)", "step dt (s)"]
        model = self.model.get()
        enabled = [True] * 5
        if model == "Projectile":
            names = ["initial speed (m/s)", "launch angle (deg)", "drag coefficient", "time (s)", "step dt (s)"]
        elif model == "Pendulum":
            names = ["angle (deg)", "length (m)", "damping", "time (s)", "step dt (s)"]
        elif model == "Double Pendulum":
            names = ["theta1 (deg)", "theta2 (deg)", "(unused)", "time (s)", "step dt (s)"]
            enabled[2] = False
        elif model == "Spring-Mass":
            names = ["mass (kg)", "x0 (m)", "spring k", "time (s)", "step dt (s)"]
        elif model == "N-Body":
            names = ["number of bodies", "(unused)", "(unused)", "time (s)", "step dt (s)"]
            enabled[1] = False
            enabled[2] = False
        for key, text, en in zip(self.param_keys, names, enabled):
            self.labels[key].config(text=text)
            self.entries[key].config(state=("normal" if en else "disabled"))

    def run(self):
        try:
            a, b, c, t_end, dt = self.vals()
            if dt <= 0 or t_end <= 0:
                raise ValueError("time and dt must be positive")
            model = self.model.get()
            method = self.integrator.get()

            if model == "Projectile":
                t, data = projectile(a, b, c, t_end, dt, method=method)
                self.plot_projectile(t, data)
                self.show_results(self.results_projectile(t, data))
            elif model == "Pendulum":
                t, data = pendulum(a, b, t_end, dt, c, method=method)
                energy = energy_pendulum(data, b)
                self.plot_ode(t, data, "Pendulum", "angle (rad)", energy)
                self.show_results(self.results_pendulum(t, data, b, energy))
            elif model == "Double Pendulum":
                t, data = double_pendulum(a, b, t_end=t_end, dt=dt, method=method)
                self.plot_double(t, data)
                self.show_results(self.results_double(t, data))
            elif model == "Spring-Mass":
                t, data = spring_mass(a, c, b, 0, t_end, dt, method=method)
                energy = energy_spring(data, a, c)
                self.plot_ode(t, data, "Spring-Mass", "x (m)", energy)
                self.show_results(self.results_spring(t, data, a, c, energy))
            elif model == "N-Body":
                n = int(round(a))
                n = min(max(n, 2), 5)
                t, r, v, m = n_body(n, t_end, dt, method=method)
                self.plot_nbody(t, r, v, m)
                self.show_results(self.results_nbody(t, r, v, m))
            self.status.config(text=f"Completed: {model} using {method}")
        except Exception as ex:
            messagebox.showerror("Simulation error", str(ex))

    # ---- Numeric result summaries (shown in the results panel) ----

    def results_projectile(self, t, d):
        land_idx = None
        for i in range(1, len(t)):
            if d[i, 1] <= 0 and d[i-1, 1] > 0:
                land_idx = i
                break
        max_height = float(np.max(d[:, 1]))
        lines = ["PROJECTILE MOTION", "-"*30]
        lines.append(f"Max height:        {max_height:.4f} m")
        if land_idx is not None:
            speed = float(np.hypot(d[land_idx, 2], d[land_idx, 3]))
            lines.append(f"Time of flight:    {t[land_idx]:.4f} s")
            lines.append(f"Range:             {d[land_idx, 0]:.4f} m")
            lines.append(f"Impact speed:      {speed:.4f} m/s")
        else:
            speed = float(np.hypot(d[-1, 2], d[-1, 3]))
            lines.append(f"Did not land within simulated time ({t[-1]:.2f} s)")
            lines.append(f"Position at t_end: x={d[-1,0]:.4f} m, y={d[-1,1]:.4f} m")
            lines.append(f"Speed at t_end:    {speed:.4f} m/s")
        return lines

    def results_pendulum(self, t, d, length, energy):
        amp_deg = float(np.rad2deg(np.max(np.abs(d[:, 0]))))
        period_num = estimate_period(t, d[:, 0])
        period_theory = 2*np.pi*np.sqrt(length/G)
        drift = 0.0 if energy[0] == 0 else (energy[-1]-energy[0])/abs(energy[0])*100
        lines = ["SIMPLE PENDULUM", "-"*30]
        lines.append(f"Amplitude:            {amp_deg:.4f} deg")
        lines.append(f"Small-angle period:   {period_theory:.4f} s (theory, 2π√(L/g))")
        lines.append(f"Numeric period:       {period_num:.4f} s" if period_num else
                      "Numeric period:       not enough oscillations to measure")
        lines.append(f"Energy drift:         {drift:+.4f} %")
        return lines

    def results_double(self, t, d):
        amp1 = float(np.rad2deg(np.max(np.abs(d[:, 0]))))
        amp2 = float(np.rad2deg(np.max(np.abs(d[:, 2]))))
        energy = energy_double_pendulum(d)
        drift = 0.0 if energy[0] == 0 else (energy[-1]-energy[0])/abs(energy[0])*100
        lines = ["DOUBLE PENDULUM", "-"*30]
        lines.append(f"Max |theta1|:         {amp1:.4f} deg")
        lines.append(f"Max |theta2|:         {amp2:.4f} deg")
        lines.append(f"Energy drift:         {drift:+.4f} %")
        lines.append("(chaotic system — small input changes give large")
        lines.append(" differences in the trajectory)")
        return lines

    def results_spring(self, t, d, m, k, energy):
        amp = float(np.max(np.abs(d[:, 0])))
        period_num = estimate_period(t, d[:, 0])
        period_theory = 2*np.pi*np.sqrt(m/k)
        drift = 0.0 if energy[0] == 0 else (energy[-1]-energy[0])/abs(energy[0])*100
        lines = ["SPRING-MASS SYSTEM", "-"*30]
        lines.append(f"Amplitude:            {amp:.4f} m")
        lines.append(f"Theory period:        {period_theory:.4f} s (2π√(m/k))")
        lines.append(f"Numeric period:       {period_num:.4f} s" if period_num else
                      "Numeric period:       not enough oscillations to measure")
        lines.append(f"Energy drift:         {drift:+.4f} %")
        return lines

    def results_nbody(self, t, r, v, m):
        kinetic = 0.5*np.sum(m[None, :, None]*v*v, axis=(1, 2))
        potential = np.zeros(len(t))
        for i in range(r.shape[1]):
            for j in range(i+1, r.shape[1]):
                dist = np.linalg.norm(r[:, j, :]-r[:, i, :], axis=1) + 1e-9
                potential -= m[i]*m[j]/dist
        total = kinetic + potential
        drift = 0.0 if total[0] == 0 else (total[-1]-total[0])/abs(total[0])*100
        lines = ["N-BODY SYSTEM", "-"*30]
        lines.append(f"Bodies simulated:     {r.shape[1]}")
        lines.append(f"Total energy (t=0):   {total[0]:.6f}")
        lines.append(f"Total energy (t_end): {total[-1]:.6f}")
        lines.append(f"Energy drift:         {drift:+.4f} %")
        lines.append("")
        lines.append("Final speeds:")
        for i in range(r.shape[1]):
            speed = float(np.linalg.norm(v[-1, i, :]))
            lines.append(f"  Body {i+1}: {speed:.4f}")
        return lines

    def plot_projectile(self, t, d):
        fig = plt.figure(figsize=(10, 7))
        ax1 = fig.add_subplot(221)
        ax1.plot(d[:,0], d[:,1])
        ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)"); ax1.set_title("Projectile trajectory")
        ax1.grid(True)
        ax2 = fig.add_subplot(222)
        ax2.plot(t, d[:,2], label="vx"); ax2.plot(t, d[:,3], label="vy")
        ax2.set_xlabel("time (s)"); ax2.set_ylabel("velocity (m/s)"); ax2.legend(); ax2.grid(True)
        ax3 = fig.add_subplot(223)
        ax3.plot(d[:,0], d[:,2]**2 + d[:,3]**2)
        ax3.set_xlabel("x (m)"); ax3.set_ylabel("speed²"); ax3.set_title("Speed² vs position"); ax3.grid(True)
        ax4 = fig.add_subplot(224)
        E = 0.5*(d[:,2]**2+d[:,3]**2) + G*d[:,1]
        ax4.plot(t, E); ax4.set_xlabel("time (s)"); ax4.set_ylabel("specific energy"); ax4.grid(True)
        fig.tight_layout(); plt.show()

    def plot_ode(self, t, d, title, ylabel, energy):
        fig = plt.figure(figsize=(10, 7))
        ax1 = fig.add_subplot(221); ax1.plot(t, d[:,0]); ax1.set_title(title); ax1.set_xlabel("time (s)"); ax1.set_ylabel(ylabel); ax1.grid(True)
        ax2 = fig.add_subplot(222); ax2.plot(d[:,0], d[:,1]); ax2.set_xlabel(ylabel); ax2.set_ylabel("rate of change"); ax2.set_title("Phase space"); ax2.grid(True)
        ax3 = fig.add_subplot(223); ax3.plot(t, energy); ax3.set_xlabel("time (s)"); ax3.set_ylabel("energy"); ax3.set_title("Energy conservation"); ax3.grid(True)
        ax4 = fig.add_subplot(224); ax4.plot(t, d[:,1]); ax4.set_xlabel("time (s)"); ax4.set_ylabel("rate of change"); ax4.set_title("Velocity / angular velocity"); ax4.grid(True)
        fig.tight_layout(); plt.show()

    def plot_double(self, t, d):
        th1, th2 = d[:,0], d[:,2]
        L1 = L2 = 1.0
        x1, y1 = L1*np.sin(th1), -L1*np.cos(th1)
        x2, y2 = x1 + L2*np.sin(th2), y1 - L2*np.cos(th2)
        fig = plt.figure(figsize=(10, 7))
        ax1 = fig.add_subplot(221); ax1.plot(x2, y2); ax1.set_aspect("equal", adjustable="box"); ax1.set_title("Double pendulum path"); ax1.grid(True)
        ax2 = fig.add_subplot(222); ax2.plot(t, th1, label="theta1"); ax2.plot(t, th2, label="theta2"); ax2.legend(); ax2.grid(True); ax2.set_xlabel("time (s)")
        ax3 = fig.add_subplot(223); ax3.plot(th1, d[:,1], label="1"); ax3.plot(th2, d[:,3], label="2"); ax3.legend(); ax3.set_title("Phase space"); ax3.grid(True)
        ax4 = fig.add_subplot(224); ax4.plot(t, np.unwrap(th1)); ax4.set_title("Angular motion"); ax4.set_xlabel("time (s)"); ax4.grid(True)
        fig.tight_layout(); plt.show()

    def plot_nbody(self, t, r, v, m):
        fig = plt.figure(figsize=(10, 7))
        ax1 = fig.add_subplot(221)
        for i in range(r.shape[1]): ax1.plot(r[:,i,0], r[:,i,1], label=f"Body {i+1}")
        ax1.set_aspect("equal", adjustable="box"); ax1.set_title("N-body trajectories"); ax1.legend(); ax1.grid(True)
        ax2 = fig.add_subplot(222)
        for i in range(r.shape[1]): ax2.plot(t, np.linalg.norm(v[:,i,:], axis=1))
        ax2.set_title("Speed vs time"); ax2.set_xlabel("time (s)"); ax2.set_ylabel("speed"); ax2.grid(True)
        ax3 = fig.add_subplot(223)
        kinetic = 0.5*np.sum(m[None,:,None]*v*v, axis=(1,2))
        potential = np.zeros(len(t))
        for i in range(r.shape[1]):
            for j in range(i+1, r.shape[1]):
                dist = np.linalg.norm(r[:,j,:]-r[:,i,:], axis=1) + 1e-9
                potential -= m[i]*m[j]/dist
        ax3.plot(t, kinetic+potential); ax3.set_title("Total energy"); ax3.set_xlabel("time (s)"); ax3.grid(True)
        ax4 = fig.add_subplot(224)
        for i in range(r.shape[1]): ax4.plot(r[:,i,0], r[:,i,1])
        ax4.set_title("Phase/position view"); ax4.set_aspect("equal", adjustable="box"); ax4.grid(True)
        fig.tight_layout(); plt.show()

    def compare(self):
        try:
            t, e, r, lf = run_comparison("pendulum")
            fig = plt.figure(figsize=(10, 7))
            ax1 = fig.add_subplot(211)
            ax1.plot(t, e[:,0], label="Euler")
            ax1.plot(t, r[:,0], label="RK4")
            ax1.plot(t, lf[:,0], label="Leapfrog")
            ax1.set_ylabel("theta (rad)"); ax1.set_title("Integrator accuracy comparison"); ax1.legend(); ax1.grid(True)
            ax2 = fig.add_subplot(212)
            E_e = energy_pendulum(e); E_r = energy_pendulum(r); E_l = energy_pendulum(lf)
            ax2.plot(t, E_e-E_e[0], label="Euler")
            ax2.plot(t, E_r-E_r[0], label="RK4")
            ax2.plot(t, E_l-E_l[0], label="Leapfrog")
            ax2.set_xlabel("time (s)"); ax2.set_ylabel("energy error"); ax2.legend(); ax2.grid(True)
            fig.tight_layout(); plt.show()

            lines = ["INTEGRATOR COMPARISON (pendulum)", "-"*30]
            for name, arr in (("Euler", e), ("RK4", r), ("Leapfrog", lf)):
                E = energy_pendulum(arr)
                lines.append(f"{name:9s} final energy drift: {(E[-1]-E[0]):+.6f}")
            self.show_results(lines)
        except Exception as ex:
            messagebox.showerror("Comparison error", str(ex))


if __name__ == "__main__":
    root = tk.Tk()
    SimulationApp(root)
    root.mainloop()
