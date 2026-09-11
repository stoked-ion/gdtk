# pyright: reportMissingImports=false
from .fast_gas import GasModel, ThermochemicalReactor
from .helper import Helper
from os.path import exists
from .grid import Grid
from tomli import load
from sys import argv
import numpy as np

class Prep:
    """Used to load configuration file, GasModel and ThermochemicalReactor."""
    @staticmethod
    def toml(glob, __version__) -> None:
        """Load the toml file and configure the inputs."""
        # Define dictionary of default configuration parameters.
        defaults = {
            "AXISYMMETRIC": True,  
            "T_WALL": -1.0, 
            "CATALYTIC": "non_catalytic", 
            "WALL_MASS_FRAC": np.array([]),
            "EXCHANGE_MODEL": "",
            "LEWIS_NUM": 1.0, 
            "VISCOSITY": False, 
            "CLUSTERING": False, 
            "CURVATURE": "empirical",
            "T_FROZEN": 1.0,
            "T_END": {"steady": 1e-1},  
            "MAX_ITERS": 1e8,
            "CFL_c": 0.4,  
            "CFL_d": 0.4,  
            "SAVE": False,
            "PLOT": False,
            "VERBOSE": True,
            "SENSOR": True,
            "BETA_SMOOTHING": True,
            "HOT_START": True,
            "LTS": False, 
            "P_INF": None,
            "RHO_INF": None}

        # Define dictionary of critical configuration parameters.
        critical_keys = [
            "GAS_MODEL",
            "MASS_FRAC",
            "T_INF",
            "U_INF",
            "RADIUS",
            "DOMAIN",
            "CELLS"]
        
        # Attempt to input the configuration file.
        try:
            nesl_input = argv[1] if len(argv) > 1 else ""
            
            if nesl_input == "--help" or nesl_input == "":
                print("\n|  _   _ _____ ____  _     ")
                print(r"| | \ | | ____/ ___|| |    ")
                print(r"| |  \| |  _| \___ \| |    v" + __version__)
                print(r"| | |\  | |___ ___) | |___ ")
                print(r"| |_| \_|_____|____/|_____|")
                print("|")
                print("| Non-Equilibrium Stagnation Line")
                print("\nUsage: nesl [options]\n")
                print("Options:")
                print("    --help   Displays help message and exits.")
                print("    --job=   Initialise a simulation using a .toml configuration file.")
                print("    --grid=  Generate grid using a .toml configuration file.\n")
                quit()

            else:
                inputs = nesl_input.split("=")
                run_type = inputs[0]
                path = inputs[1]

        except IndexError:
            print("Error loading configuration file cannot proceed.")
            quit()

        try:
            # Load configuration data as dictionary.
            with open(path, "rb") as f:
                user_cfg = load(f)
                
            # Check to ensure all critical parameters have been provided.
            for key in critical_keys:
                if key not in user_cfg:
                    print(f"Parameter '{key}' not provided cannot proceed.")
                    quit()
            
            if "P_INF" not in user_cfg and "RHO_INF" not in user_cfg:                    
                print("Parameter 'RHO_INF' or 'P_INF' not provided cannot proceed.")
                quit()

            # Create gmodel to infer species names.
            gmodel = Prep.gas(user_cfg["GAS_MODEL"])
            species_names = gmodel.nameSpecies
            num_species = gmodel.numSpecies

            # Update some parameters in user config.
            user_cfg["MASS_FRAC"] = Helper.dict_to_array(user_cfg["MASS_FRAC"], species_names)

            if "RECOM_RATE" in user_cfg:
                user_cfg["RECOM_RATE"] = Helper.dict_to_array(user_cfg["RECOM_RATE"], species_names)
                recom_eff = np.empty(num_species, dtype=np.double)
                recom_products = np.empty(num_species, dtype=int)

                for idx, f in enumerate(user_cfg["RECOM_RATE"]):
                    if isinstance(f, (int, float)):
                        recom_eff[idx] = f
                        recom_products[idx] = -1

                    elif isinstance(f, dict):
                        recom_eff[idx] = f["efficiency"]
                        recom_products[idx] = species_names.index(f["product"])

            else:
                recom_eff = np.zeros(num_species, dtype=np.double)
                recom_products = np.full(num_species, -1, dtype=int)

            if "WALL_MASS_FRAC" in user_cfg:
                user_cfg["WALL_MASS_FRAC"] = Helper.dict_to_array(user_cfg["WALL_MASS_FRAC"], species_names)

            if "CHEM_MODEL" in user_cfg:
                chemistry = True

            else:
                chemistry = False
            
            final_cfg = defaults | user_cfg

            # Update some parameters in final config.
            if isinstance(final_cfg["T_WALL"], (float, int)):
                final_cfg["T_WALL"] = float(final_cfg["T_WALL"])
            
            else:
                final_cfg["T_WALL"] = -1.0 
            
            if final_cfg["AXISYMMETRIC"]:
                alpha = 2

            else:
                alpha = 1

            if isinstance(final_cfg["T_END"], dict):
                residual = final_cfg["T_END"]["steady"]
                final_cfg["T_END"] = -1.0

            else:
                residual = -1.0
                final_cfg["T_END"] = float(final_cfg["T_END"])

            if final_cfg["T_END"] >= 0.0 and final_cfg["LTS"]:
                print("Must use steady state mode if 'LTS' is activated.")
                quit()

            if isinstance(final_cfg["SAVE"], dict) and final_cfg["LTS"]:
                print("Cannot track transient events if 'LTS' is activated.")
                quit()

            final_cfg["MAX_ITERS"] = int(final_cfg["MAX_ITERS"])
            final_cfg["T_INF"] = float(final_cfg["T_INF"])
            final_cfg["T_FROZEN"] = float(final_cfg["T_FROZEN"])
            final_cfg["LEWIS_NUM"] = float(final_cfg["LEWIS_NUM"])

            if final_cfg["P_INF"] is not None:
                final_cfg["P_INF"] = float(final_cfg["P_INF"]) 

            else:
                final_cfg["RHO_INF"] = float(final_cfg["RHO_INF"])

            if isinstance(final_cfg["SAVE"], dict):
                transient_interval = float(final_cfg["SAVE"]["interval"])
                final_cfg["SAVE"] = final_cfg["SAVE"]["path"]

            else:
                transient_interval = -1.0

            final_cfg.update({'chemistry': chemistry, 'alpha': alpha, 'residual': residual,'reactor': None, 
                              'recom_eff': recom_eff, 'recom_products': recom_products, 
                              'transient_interval': transient_interval})

            # Push configuration parameters to global space.
            glob.update(final_cfg)

            # Depending on run_type make grid and quit.
            if run_type == "--grid":
                print("Building grid...")
                Grid(final_cfg["CELLS"], final_cfg["DOMAIN"], final_cfg["CLUSTERING"], num_species).plot()
                print("Grid built and saved under 'grid.png'.")
                quit()

        except Exception as e:
            print(f"Error loading '{path}' : {e}.")
            quit()

    @staticmethod
    def gas(GAS_MODEL) -> GasModel:
        """Initialise a fast_gas GasModel"""
        try:
            if not exists(GAS_MODEL):
                    raise FileNotFoundError
            gmodel = GasModel(GAS_MODEL)
            
        except FileNotFoundError:
            print(f"The file '{GAS_MODEL}' could not be found.")
            quit()

        except Exception as e:
            print(f"There was an error creating the gas model from file '{GAS_MODEL}' : {e}")
            quit()

        return gmodel
    
    @staticmethod
    def chem(CHEM_MODEL, EXCHANGE_MODEL, gmodel) -> ThermochemicalReactor:
        """Initialise a fast_gas ThermochemicalReactor"""
        try:
            required_files = {"CHEM_MODEL": CHEM_MODEL}
            if EXCHANGE_MODEL:
                required_files["EXCHANGE_MODEL"] = EXCHANGE_MODEL

            missing = [f"{name}='{path}'" for name, path in required_files.items() if not exists(path)]
            if missing:
                raise FileNotFoundError("Cannot locate files, " + ", ".join(missing))

            reactor = ThermochemicalReactor(gmodel, CHEM_MODEL, EXCHANGE_MODEL)

        except FileNotFoundError as e:
            print(e)
            quit()

        except Exception as e:
            print(f"There was an error creating the ThermochemicalReactor : {e}")
            quit()

        return reactor