from gdtk.geom.cluster import RobertsFunction
from matplotlib.ticker import LinearLocator
from .plot import TwoDPFormatter
import matplotlib.pyplot as plt
import numpy as np

class Grid:
    """Generates a 1D clustered or linearly spaced grid."""
    def __init__(self, cells, domain, clustering, nsp) -> None:
        self.cells = cells
        self.domain = domain
        self.nsp = nsp

        is_number = isinstance(clustering, (float, int)) and not isinstance(clustering, bool)
        self.clustering = is_number and clustering > 1.0

        if self.clustering:
            self.beta = clustering

        elif clustering is not False:
            print("Invalid 'CLUSTERING', must be a number > 1.")
            quit()

    def roberts_cluster(self) -> np.ndarray:
        """Roberts clustering near the wall."""
        cf = RobertsFunction(False, True, self.beta)
        scale = cf.distribute_parameter_values(self.cells + 1)

        return scale * self.domain

    def linear(self) -> np.ndarray:
        """Linear grid spacing."""
        x_face = np.linspace(0.0, self.domain, self.cells + 1)

        return x_face
    
    def solve(self, x_face) -> tuple[np.ndarray, ...]:
        """Calculates additional arrays based on the spacing of the cell faces."""
        dx = np.diff(x_face)   

        dx_L = dx[0]  
        dx_R = dx[-1]  

        dx = np.insert(dx, [0, len(dx)], [dx_L, dx_R])  
        dx2d = np.full((dx.size, self.nsp + 2 + 1), dx[:, np.newaxis]) 

        x_cen_interior = 0.5 * (x_face[:-1] + x_face[1:])
        x_cen_L = 0 - 0.5 * dx_L  
        x_cen_R = self.domain + 0.5 * dx_R  

        x_cen = np.insert(x_cen_interior, [0, len(x_cen_interior)], [x_cen_L, x_cen_R])  
        
        return x_cen, dx, dx2d

    def make(self) -> tuple[np.ndarray, ...]:
        """Consolidates all methods and makes the grid."""
        if self.clustering:
            x_face = self.roberts_cluster()
            x_cen, dx, dx2d = self.solve(x_face)

        else:
            x_face = self.linear()
            x_cen, dx, dx2d = self.solve(x_face)
        
        return x_cen, dx, dx2d
    
    @staticmethod
    def export_to_vtk(dx, save_name) -> None:
        """Exports the grid as a VTK, where it can be viewed with tools like ParaView."""
        x_face = np.concatenate([[0], np.cumsum(dx[1:-1])])
        height = max(dx)
        
        header = (
            "# vtk DataFile Version 3.0\n"
            "nesl grid\n"
            "ASCII\n"
            "DATASET RECTILINEAR_GRID\n"
            f"DIMENSIONS {len(x_face)} 2 1\n"
            f"X_COORDINATES {len(x_face)} float\n"
        )

        with open(save_name, 'w') as f:
            f.write(header)
            for x in x_face:
                f.write(f"{x:.15g} ")
            f.write("\n")
            f.write("Y_COORDINATES 2 float\n0.0 {:.15g}\n".format(height))
            f.write("Z_COORDINATES 1 float\n0.0\n")

    def plot(self) -> None:
        """Plots the grid."""
        label_size = 9
        tick_size = 8
        line_width = 1.2
        transparency = 0.8

        plt.rcParams['xtick.major.pad'] = 8
        
        x_cen, dx, _ = self.make()

        x = x_cen[1:-1]
        d = x[0] + x[-1] - x

        if self.clustering:
            plt.plot(d, dx[1:-1], linewidth=line_width, color='k')

        else:
            plt.plot(d, np.full(len(d), dx[-2]), 'k')

        ax = plt.gca()

        ax.yaxis.set_major_locator(LinearLocator(numticks=6))
        ax.yaxis.set_major_formatter(TwoDPFormatter())
        ax.yaxis.get_offset_text().set_fontsize(tick_size)

        plt.ylabel(r"$\Delta \ x, \ m$", fontsize=label_size)
        plt.xlabel(r"$x, \ m$", fontsize=label_size)
        plt.grid(True, linestyle=':', alpha=transparency)

        ax.set_xlim(d.max(), d.min())
        ax.xaxis.set_major_locator(LinearLocator(numticks=8))
        ax.xaxis.set_major_formatter(TwoDPFormatter())
        ax.xaxis.get_offset_text().set_fontsize(tick_size)

        ax.tick_params(axis='both', labelsize=tick_size, direction='in')

        plt.tight_layout()
        plt.savefig("grid.png", dpi=300)

        self.export_to_vtk(dx, 'grid.vtk')

