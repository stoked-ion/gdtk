import numpy as np

class Helper:
    """Stores helper functions."""
    @staticmethod
    def shock_standoff(epsilon, radius, axisymmetric = True) -> float:
        """Approximates the shock standoff distance for a blunt body."""
        if axisymmetric:
            delta = radius * epsilon / (1 - epsilon + np.sqrt(8 / 3 * epsilon))

        else:
            delta = radius * epsilon / (1 - epsilon + np.sqrt(8 / 3 * epsilon)) * (0.386 / 0.143) 

        return delta 

    @staticmethod
    def find_idx(array, value) -> int:
        """Find the index of the element in an array closest to a given value."""
        array = np.asarray(array)
        idx = int((np.abs(array - value)).argmin())
        return idx

    @staticmethod
    def dict_to_array(input_dict, species_order) -> np.ndarray:
        """Converts a dictionary to an array based on a specific order."""
        if isinstance(input_dict, dict) and any(isinstance(v, dict) for v in input_dict.values()):
            output = np.array([input_dict.get(species, 0.0) for species in species_order], dtype=object)            

        else:
            output = np.array([input_dict.get(species, 0.0) for species in species_order], dtype=np.double)
        
        return output
