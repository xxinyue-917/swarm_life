
import os
# Run pygame headless (no GUI). IMPORTANT: set this BEFORE importing particle_life (which imports pygame).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Import your simulator
import particle_life  # must be in the same folder
from particle_life import Config, ParticleLife

# Metrics
from metrics import (
    avg_radii, kinetic_energy, update_revolutions, mean_spacing_same, mean_spacing_cross
)

ORI_CASES = {
    # A: no orientation coupling
    "A": lambda: (0.0, 0.0),
    # B: same-sign orientation attraction
    "B": lambda: (1.0, 1.0),
    # C: opposite-sign orientation (tends to lanes)
    "C": lambda: (1.0, -1.0),
    # D: one-way orientation attraction (shepherd/satellite)
    "D": lambda: (1.0, 0.0),
}

def build_config(k12_pos, k21_pos, k12_ori, k21_ori, args):
    cfg = Config(
        width=args.width, height=args.height,
        n_species=2, n_particles=args.n_particles,
        dt=args.dt, max_speed=args.max_speed,
        a_rep=args.a_rep, a_att=args.a_att,
        seed=args.seed_base,
        max_angular_speed=args.max_angular_speed,
        a_rot=args.a_rot
    )
    # Diagonals fixed (same-species) = args.C
    pos = np.array([[args.C, k12_pos],
                    [k21_pos, args.C]], dtype=float)
    ori = np.array([[args.C_ori, k12_ori],
                    [k21_ori, args.C_ori]], dtype=float)
    cfg.position_matrix = pos.tolist()
    cfg.orientation_matrix = ori.tolist()
    return cfg

def run_once(cfg, steps_burnin, steps_sample, sample_stride, seed, verbose=False):
    # Each run uses a different seed for initial conditions
    cfg.seed = int(seed)

    sim = ParticleLife(cfg)
    # We won't call sim.run() because that would draw; we only step() here.
    # Prepare accumulators
    species = sim.species.copy()  # [N]
    mask1 = (species == 0)
    mask2 = (species == 1)

    # Revolutions counter
    revs = 0.0
    # Initialize angle using centroids
    c1 = sim.positions[mask1].mean(axis=0)
    c2 = sim.positions[mask2].mean(axis=0)
    prev_angle = math.atan2(*(c1 - c2)[::-1])

    # Burn-in (show progress only in verbose mode)
    if verbose:
        for _ in tqdm(range(steps_burnin), desc="    Burn-in", leave=False):
            sim.step()
    else:
        for _ in range(steps_burnin):
            sim.step()

    # Accumulators
    n_samples = 0
    sum_R1 = 0.0; sum_R2 = 0.0
    sum_K = 0.0
    sum_d11 = 0.0; sum_d22 = 0.0
    sum_d12 = 0.0

    # Sampling (show progress only in verbose mode)
    sample_range = tqdm(range(steps_sample), desc="    Sampling", leave=False) if verbose else range(steps_sample)
    for t in sample_range:
        sim.step()
        if (t % sample_stride) != 0:
            continue

        X = sim.positions  # [N,2]
        V = sim.velocities # [N,2]

        # radii
        R1, R2 = avg_radii(X, mask1, mask2)
        sum_R1 += R1; sum_R2 += R2

        # kinetic energy
        sum_K += kinetic_energy(V)

        # spacing
        d11 = mean_spacing_same(X, mask1)
        d22 = mean_spacing_same(X, mask2)
        d12 = mean_spacing_cross(X, mask1, mask2)
        sum_d11 += d11; sum_d22 += d22; sum_d12 += d12

        # revolutions
        c1 = X[mask1].mean(axis=0)
        c2 = X[mask2].mean(axis=0)
        revs, prev_angle = update_revolutions(c1, c2, prev_angle, revs)

        n_samples += 1

    # Averages
    out = {
        "R1": (sum_R1 / max(n_samples,1)),
        "R2": (sum_R2 / max(n_samples,1)),
        "Rdiff": (sum_R1 - sum_R2) / max(n_samples,1),
        "K": (sum_K / max(n_samples,1)),
        "d11": (sum_d11 / max(n_samples,1)),
        "d22": (sum_d22 / max(n_samples,1)),
        "d12": (sum_d12 / max(n_samples,1)),
        "revs": revs,
        "samples": n_samples,
    }
    return out

def main():
    p = argparse.ArgumentParser(description="Parameter sweep for Particle Life (2 species).")
    # Grid over position couplings
    p.add_argument("--k12-min", type=float, default=0.1)
    p.add_argument("--k12-max", type=float, default=1.0)
    p.add_argument("--k21-min", type=float, default=0.0)
    p.add_argument("--k21-max", type=float, default=1.1)
    p.add_argument("--grid", type=int, default=5, help="grid size per axis")

    # Orientation cases to include
    p.add_argument("--ori-cases", type=str, default="A",
                   help="comma-separated subset of A,B,C,D")

    # Same-species constants
    p.add_argument("--C", type=float, default=0.6, help="diagonal of position matrix")
    p.add_argument("--C-ori", type=float, default=0.0, help="diagonal of orientation matrix")

    # Sim parameters
    p.add_argument("--n-particles", type=int, default=90)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--max-speed", type=float, default=300.0)
    p.add_argument("--a-rep", type=float, default=5.0)
    p.add_argument("--a-att", type=float, default=2.0)
    p.add_argument("--max-angular-speed", type=float, default=20.0)
    p.add_argument("--a-rot", type=float, default=0.0)
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=800)

    # Run control
    p.add_argument("--burnin", type=int, default=100)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--stride", type=int, default=5, help="sample every k steps")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--seed-base", type=int, default=123)

    # Output
    p.add_argument("--out", type=str, default="sweep_out")
    p.add_argument("--verbose", action="store_true", help="Show detailed progress bars for each simulation")

    args = p.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save a copy of arguments for provenance
    with open(outdir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Prepare CSV
    fieldnames = [
        "ori_case", "k12_pos", "k21_pos", "k12_ori", "k21_ori",
        "seed", "R1", "R2", "Rdiff", "K", "d11", "d22", "d12", "revs", "samples"
    ]
    csv_path = outdir / "results.csv"
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        # Build grid
        k12_vals = np.linspace(args.k12_min, args.k12_max, args.grid)
        k21_vals = np.linspace(args.k21_min, args.k21_max, args.grid)

        # Calculate total number of runs for progress bar
        cases_to_run = [c.strip() for c in args.ori_cases.split(",") if c.strip()]
        valid_cases = [c for c in cases_to_run if c in ORI_CASES]
        total_runs = len(valid_cases) * args.grid * args.grid * args.seeds

        print(f"\n📊 Starting parameter sweep:")
        print(f"  - Orientation cases: {', '.join(valid_cases)}")
        print(f"  - Parameter grid: {args.grid}×{args.grid}")
        print(f"  - Seeds per point: {args.seeds}")
        print(f"  - Total simulations: {total_runs}")
        print(f"  - Output directory: {outdir}\n")

        # Main progress bar for overall progress
        start_time = time.time()
        with tqdm(total=total_runs, desc="Overall Progress", unit="sim", miniters=1) as pbar:
            for case_name in cases_to_run:
                if case_name not in ORI_CASES:
                    print(f"Skip unknown orientation case: {case_name}")
                    continue
                k12_ori_val, k21_ori_val = ORI_CASES[case_name]()

                print(f"\n== Case {case_name}: k12_ori={k12_ori_val}, k21_ori={k21_ori_val} ==")
                t0_case = time.time()

                # Nested progress bar for each case
                case_total = args.grid * args.grid * args.seeds
                with tqdm(total=case_total,
                         desc=f"  Case {case_name}",
                         leave=False,
                         unit="sim") as case_pbar:

                    for i, k12 in enumerate(k12_vals):
                        for j, k21 in enumerate(k21_vals):
                            cfg = build_config(k12, k21, k12_ori_val, k21_ori_val, args)

                            # Update progress bar description with current parameters
                            case_pbar.set_postfix({
                                'k12': f'{k12:.2f}',
                                'k21': f'{k21:.2f}',
                                'grid': f'[{i+1},{j+1}]/{args.grid}'
                            })

                            for s in range(args.seeds):
                                seed = args.seed_base + 1000*i + 10*j + s

                                # Time first simulation to estimate total time
                                if pbar.n == 0:
                                    t0_first = time.time()

                                res = run_once(cfg, args.burnin, args.steps, args.stride, seed, args.verbose)

                                # After first simulation, print time estimate
                                if pbar.n == 0:
                                    sim_time = time.time() - t0_first
                                    total_time_est = sim_time * total_runs
                                    print(f"\n  ⏱️  First simulation took {sim_time:.1f}s")
                                    print(f"  📊 Estimated total time: {total_time_est/60:.1f} minutes ({total_time_est/3600:.1f} hours)\n")

                                row = {
                                    "ori_case": case_name,
                                    "k12_pos": float(k12),
                                    "k21_pos": float(k21),
                                    "k12_ori": float(k12_ori_val),
                                    "k21_ori": float(k21_ori_val),
                                    "seed": int(seed),
                                    **res
                                }
                                writer.writerow(row)
                                fcsv.flush()  # Force write to disk

                                # Update both progress bars
                                case_pbar.update(1)
                                pbar.update(1)

                                # Update main progress bar info
                                pbar.set_postfix({
                                    'case': case_name,
                                    'k12': f'{k12:.2f}',
                                    'k21': f'{k21:.2f}'
                                })

                print(f"  ✓ Case {case_name} done in {time.time()-t0_case:.1f}s")

    print(f"\n✅ All done! Results saved to: {csv_path}")
    print(f"📊 Total simulations completed: {total_runs}")
    print(f"📁 Output directory: {outdir}")
    print(f"\n💡 Next step: Use plot_heatmaps.py to visualize the CSV:")
    print(f"   python plot_heatmaps.py --csv {csv_path} --out plots")

if __name__ == "__main__":
    main()
