#!/bin/sh
"exec" "$DGD/lib/.nesl/bin/python3" "-B" "$0" "$@"

"""
A script that builds on the NESL program to 
parallelise some condition building.

Author: Taine J. Rossini (t.rossini@uq.edu.au)
Build Date: March 20, 2026

"""

__version__ = "1.0.0"

from nesl_utils.fast_gas import GasState, GasFlow 
from os import mkdir, makedirs, getpid
from nesl_utils import Helper, Prep
from os.path import dirname, join
from multiprocessing import Pool
from itertools import product
from sys import argv, path
from subprocess import run
from pathlib import Path
from tomli_w import dump
from tomli import load
import numpy as np
import math
import csv

def ceil_sig(x, sig=2) -> float:
    """Round up value to a certain amount of significant figures."""
    if x == 0:
        return 0
    
    power = sig - 1 - math.floor(math.log10(abs(x)))
    factor = 10 ** power
    
    return math.ceil(x * factor) / factor

def get_domain(T, p_or_rho, u, massf, gasm, R, axisymmetric, p_or_rho_type, scale=1.5) -> float:
    """Calculate the domain length based on the maximum possible shock standoff (non-reacting inviscid)."""
    gmodel = Prep.gas(gasm)
    flow = GasFlow(gmodel)
    state_inf = GasState(gmodel)

    if p_or_rho_type == 'p':
        state_inf.p = p_or_rho

    else:
        state_inf.rho = p_or_rho
        
    state_inf.T = T
    state_inf.T_modes = np.array([T])
    state_inf.massf = massf
    state_inf.push()

    if p_or_rho_type == 'p':
        state_inf.update_thermo_from_pT()

    else:
        state_inf.update_thermo_from_rhoT()

    state_inf.update_sound_speed()
    state_inf.pull()

    state2 = GasState(gmodel)
    _, _ = flow.normal_shock(state_inf, u, state2)
    state2.pull()

    epsilon = state_inf.rho / state2.rho
    delta = ceil_sig(scale * Helper.shock_standoff(epsilon, R, axisymmetric))

    return delta

def input_to_arr(config) -> np.ndarray:  # type: ignore
    """Creates an array based on an input from the config file."""
    if isinstance(config, (int, float)):
        return np.array([float(config)])
    
    else:
        lower = config["lower"]
        upper = config["upper"]
        cells = config["cells"]
        range_type = config["bound_type"]

        if range_type == 'linear':
            return np.linspace(lower, upper, cells)

        if range_type == 'log_upper':
            return np.logspace(np.log10(upper), np.log10(lower), cells)

        if range_type == 'log_lower':
            return np.logspace(np.log10(lower), np.log10(upper), cells)

def load_condition_builder_config() -> tuple[float, bool, bool, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
    """Load the condition builder config file."""
    path = argv[2].split("=")[1]

    with open(path, "rb") as f:
        config = load(f)

    if "DOMAIN_SCALE" in config:
        domain_scale = float(config["DOMAIN_SCALE"])

    else:
        domain_scale = 1.5

    if "FIXED_DOMAIN" in config:
        fixed_domain = bool(config["FIXED_DOMAIN"])

    else:
        fixed_domain = False

    if "PERMUTATIONS" in config:
        permutations = config["PERMUTATIONS"]
        
    else:
        permutations = True

    if "P_BOUND" in config:
        if isinstance(config["P_BOUND"], str):
            P_arr = np.loadtxt(config["P_BOUND"])

        else:
            P_arr = input_to_arr(config["P_BOUND"])

    else:
        P_arr = None

    if "RHO_BOUND" in config:
        if isinstance(config["RHO_BOUND"], str):
            Rho_arr = np.loadtxt(config["RHO_BOUND"])

        else:
            Rho_arr = input_to_arr(config["RHO_BOUND"])

    else:
        Rho_arr = None

    if isinstance(config["T_BOUND"], str):
        T_arr = np.loadtxt(config["T_BOUND"])

    else:
        T_arr = input_to_arr(config["T_BOUND"])

    if isinstance(config["U_BOUND"], str):
        U_arr = np.loadtxt(config["U_BOUND"])

    else:
        U_arr = input_to_arr(config["U_BOUND"])

    return domain_scale, fixed_domain, permutations, T_arr, P_arr, Rho_arr, U_arr

def run_nesl(T, p_or_rho, u, domain_list, identifier, template, fixed_domain) -> None:
    """Updates the nesl job file and runs the script."""
    try:
        if not fixed_domain:
            domain = get_domain(T, p_or_rho, u, *domain_list)

        print(f"Starting case {identifier} on processor {getpid()}.")

        path = "runs/case" + str(identifier)
        file = path + "/job.toml"

        makedirs(path)
        run(["cp", template, file])

        with open(file, "rb") as f:
            data = load(f)

        data["T_INF"] = T
        if domain_list[-2] == 'p':
            data["P_INF"] = p_or_rho

        else:
            data["RHO_INF"] = p_or_rho

        data["U_INF"] = u

        if not fixed_domain:
            data["DOMAIN"] = domain

        data["GAS_MODEL"] = "../../" + data["GAS_MODEL"].split("/")[-1]

        if "CHEM_MODEL" in data:
            data["CHEM_MODEL"] = "../../" + data["CHEM_MODEL"].split("/")[-1]

        if "EXCHANGE_MODEL" in data:
            data["EXCHANGE_MODEL"] = "../../" + data["EXCHANGE_MODEL"].split("/")[-1]

        with open(file, "wb") as f:
            dump(data, f)

        run("nesl =" + "job.toml", shell=True, cwd=path)        

        print(f"Ending case {identifier} on processor {getpid()}.")

    except Exception as e:
        print(f"Error: {e}")

def build_result(p_or_rho_type) -> None:
    """Read the results and build into a single csv."""
    root = Path("runs") 

    with open(root / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if p_or_rho_type == 'p':
            writer.writerow(["case", " T_inf (K)", " p_inf (Pa)", " u_inf (m/s)", " standoff (m)", " q_c (W/m^2)", " q_d (W/m^2)", " p (Pa)", " R_s (m)"])

        else:
            writer.writerow(["case", " T_inf (K)", " rho_inf (kg/m^3)", " u_inf (m/s)", " standoff (m)", " q_c (W/m^2)", " q_d (W/m^2)", " p (Pa)", " R_s (m)"])

        for folder in sorted(root.glob("case*")):
            try:

                job = next(folder.glob("*.toml"))
                with open(job, "rb") as tf:
                    cfg = load(tf)

                stag_file = next(folder.glob("stagpoint_*"))
                data = np.loadtxt(stag_file, delimiter=",", skiprows=1)
                data = np.ravel(data)

                if p_or_rho_type == 'p':
                    row = [cfg["T_INF"], cfg["P_INF"], cfg["U_INF"], data[0], data[1], data[2], data[3], data[4]]

                else:
                    row = [cfg["T_INF"], cfg["RHO_INF"], cfg["U_INF"], data[0], data[1], data[2], data[3], data[4]]

                writer.writerow([folder.name[4:]] + [f"{value:.18e}" for value in row])

            except Exception:
                writer.writerow([folder.name[4:], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

def main() -> None:
    """Initialises processes and sets up condition building."""
    nesl_cb_input = argv[1] if len(argv) > 1 else ""
    
    if nesl_cb_input == "--help" or nesl_cb_input == "":
        print("\n|  _   _  ____ ____  ")
        print(r"| | \ | |/ ___| __ ) ")
        print(r"| |  \| | |   |  _ \ ")
        print(r"| | |\  | |___| |_) |   v" + __version__)
        print(r"| |_| \_|\____|____/ ")
        print("|")
        print("| NESL Condition Builder")
        print("\nUsage: nesl_condition_builder [options]\n")
        print("Options:")
        print("    --help        Displays help message and exits.")
        print("    --job=        Template .toml configuration file.")
        print("    --bounds=     Bounds to test specified using a .toml configuration file.")
        print("    --max_cpus=   Number of cores to spread condition building across.")
        print()
        print("Note, '--job=' and '--bounds=' fields must be filled.")
        print()
        quit()

    Prep.toml(globals(), "")

    domain_scale, fixed_domain, permutations, T_arr, P_arr, Rho_arr, U_arr = load_condition_builder_config()

    if permutations:
        if P_arr is None:
            p_or_rho_type = 'rho'
            perms = list(product(T_arr, Rho_arr, U_arr))  # type: ignore
        
        else:
            p_or_rho_type = 'p'
            perms = list(product(T_arr, P_arr, U_arr))
    
    else:
        if P_arr is None:
            p_or_rho_type = 'rho'
            perms = list(zip(T_arr, Rho_arr, U_arr))  # type: ignore
        
        else:
            p_or_rho_type = 'p'
            perms = list(zip(T_arr, P_arr, U_arr))
    
    width = len(str(len(perms) - 1))

    domain_list = [MASS_FRAC, GAS_MODEL, RADIUS, AXISYMMETRIC, p_or_rho_type, domain_scale]  # type: ignore

    template = argv[1].split("=")[1]

    inputs = [(T, P_or_Rho, U, domain_list, str(i).zfill(width), template, fixed_domain) for i, (T, P_or_Rho, U) in enumerate(perms)]

    try:
        mkdir("runs")

    except FileExistsError:
        print(f"'runs' folder already exists.")
        quit()

    no_cpu = 1 

    if len(argv) > 3 and "=" in argv[3]:
        no_cpu = int(argv[3].split("=", 1)[1])

    with Pool(processes=no_cpu) as pool:
        pool.starmap(run_nesl, inputs)

    build_result(p_or_rho_type)

if __name__ == "__main__":
    BASE = dirname(dirname(__file__))
    path.insert(0, join(BASE, "lib"))
    
    main()