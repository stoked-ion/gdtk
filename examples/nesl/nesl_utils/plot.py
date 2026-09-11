from matplotlib.ticker import ScalarFormatter, LinearLocator
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['xtick.major.pad'] = 8

class TwoDPFormatter(ScalarFormatter):
    """Scientific-notation tick formatter."""
    def __init__(self) -> None:
        super().__init__(useMathText=True)
        self.set_powerlimits((-2, 2))

    def _set_format(self) -> None:
        self.format = '%.2f'

class Plot:
    """Stores plotting methods."""
    title_size = 11
    label_size = 9
    tick_size = 8
    line_width = 1.2
    transparency = 0.8

    colours = [
        "#4063D8", 
        "#389826", 
        "#CB3C33", 
        "#9558B2",
        "#D95319",
        "#1EA896", 
        "#D84C8A", 
        "#ECA521", 
        "#8F6851", 
        "#29C3DE", 
        "#768294" 
    ]

    @classmethod
    def massf(cls, data_path, save_name, species_names) -> None:
        """A simple function to plot the mass fractions."""
        data = np.loadtxt(data_path, skiprows=1, delimiter=',')
        
        x = data[:, 0]
        d = x[0] + x[-1] - x 

        n_plots = len(species_names)

        fig, axs = plt.subplots(n_plots, 1, figsize=(8, n_plots * 1.5), sharex=True)
        
        axs[0].set_title("Stagnation Line Mass Fractions", fontsize=cls.title_size)

        for i, name in enumerate(species_names):
            ax = axs[i]
            
            y_data = data[:, i + data.shape[1] - n_plots]

            ax.plot(d, y_data, linewidth=cls.line_width, color=cls.colours[i % len(cls.colours)])
            
            ax.set_ylabel(rf'${{{name}}}$', fontsize=cls.label_size)
            ax.grid(True, linestyle=':', alpha=cls.transparency)
            ax.tick_params(axis='both', labelsize=cls.tick_size, direction='in')
            
            ax.yaxis.set_major_locator(LinearLocator(numticks=5))
            ax.yaxis.set_major_formatter(TwoDPFormatter())
            ax.yaxis.get_offset_text().set_fontsize(cls.tick_size)
            
        axs[-1].set_xlabel(r"$x, \ m$", fontsize=cls.label_size)
        axs[-1].set_xlim(d.max(), d.min())

        ax = plt.gca()
        ax.xaxis.set_major_locator(LinearLocator(numticks=8))
        ax.xaxis.set_major_formatter(TwoDPFormatter())
        ax.xaxis.get_offset_text().set_fontsize(cls.tick_size)

        plt.tight_layout()
        fig.align_ylabels(axs)
        plt.savefig(save_name + '.png', dpi=300)
        plt.savefig(save_name + '.pdf', format='pdf', dpi=300, bbox_inches="tight")
        plt.close(fig)

    @classmethod
    def prim(cls, data_path, save_name) -> None:
        """A simple function to plot some key primitive variables."""
        data = np.loadtxt(data_path, skiprows=1, delimiter=',')

        x = data[:, 0]
        d = x[0] + x[-1] - x 

        plot_configs = [
            (data[:, 1], r"$u, \ m/s$"),
            (data[:, 2], r"$\rho, \ kg/m^3$"),
            (data[:, 3], r"$T, \ K$"),
            (data[:, 4], r"$p, \ Pa$")
        ]

        n_plots = len(plot_configs)

        fig, axs = plt.subplots(n_plots, 1, figsize=(8, 8), sharex=True)
        
        axs[0].set_title("Stagnation Line Primitive Variables", fontsize=cls.title_size)

        for i, (y_data, ylabel) in enumerate(plot_configs):
            ax = axs[i]

            ax.plot(d, y_data, linewidth=cls.line_width, color=cls.colours[i % len(cls.colours)])
            ax.set_ylabel(ylabel, fontsize=cls.label_size)
            ax.grid(True, linestyle=':', alpha=cls.transparency)
            ax.tick_params(axis='both', labelsize=cls.tick_size, direction='in')
  
            ax.yaxis.set_major_locator(LinearLocator(numticks=5))
            ax.yaxis.set_major_formatter(TwoDPFormatter())
            ax.yaxis.get_offset_text().set_fontsize(cls.tick_size)

        axs[-1].set_xlabel(r"$x, \ m$", fontsize=cls.label_size)
        axs[-1].set_xlim(d.max(), d.min())

        axs[-1].xaxis.set_major_locator(LinearLocator(numticks=8))
        axs[-1].xaxis.set_major_formatter(TwoDPFormatter())
        axs[-1].xaxis.get_offset_text().set_fontsize(cls.tick_size)

        plt.tight_layout()
        fig.align_ylabels(axs)
        plt.savefig(save_name + '.png', dpi=300)
        plt.savefig(save_name + '.pdf', format='pdf', dpi=300, bbox_inches="tight")
        plt.close(fig)

    @classmethod
    def prim_two_T(cls, data_path, save_name) -> None:
        """A simple function to plot some key primitive variables."""
        data = np.loadtxt(data_path, skiprows=1, delimiter=',')

        x = data[:, 0]
        d = x[0] + x[-1] - x 

        plot_configs = [
            (data[:, 1], r"$u, \ m/s$"),
            (data[:, 2], r"$\rho, \ kg/m^3$"),
            (data[:, 3], r"$T, \ K$"),
            (data[:, 4], r"$p, \ Pa$"),
            (data[:, 11], r"$T_{ve}$")
        ]

        n_plots = len(plot_configs) - 1

        fig, axs = plt.subplots(n_plots, 1, figsize=(8, 8), sharex=True)
        
        axs[0].set_title("Stagnation Line Primitive Variables", fontsize=cls.title_size)

        for i, (y_data, ylabel) in enumerate(plot_configs[:-1]):
            ax = axs[i]
            
            if i == 2:
                ax.plot(d, y_data, linewidth=cls.line_width, label=r"$T_{tr}$", color=cls.colours[i % len(cls.colours)])
                ax.plot(d, plot_configs[-1][0], linestyle=':', linewidth=cls.line_width, label=plot_configs[-1][1], color=cls.colours[i % len(cls.colours)])
                ax.legend()

            else:
                ax.plot(d, y_data, linewidth=cls.line_width, label=ylabel, color=cls.colours[i % len(cls.colours)])

            ax.set_ylabel(ylabel, fontsize=cls.label_size)
            ax.grid(True, linestyle=':', alpha=cls.transparency)
            ax.tick_params(axis='both', labelsize=cls.tick_size, direction='in')

            ax.yaxis.set_major_locator(LinearLocator(numticks=5))
            ax.yaxis.set_major_formatter(TwoDPFormatter())
            ax.yaxis.get_offset_text().set_fontsize(cls.tick_size)

        axs[-1].set_xlabel(r"$x, \ m$", fontsize=cls.label_size)
        axs[-1].set_xlim(d.max(), d.min())

        axs[-1].xaxis.set_major_locator(LinearLocator(numticks=8))
        axs[-1].xaxis.set_major_formatter(TwoDPFormatter())
        axs[-1].xaxis.get_offset_text().set_fontsize(cls.tick_size)

        plt.tight_layout()
        fig.align_ylabels(axs)
        plt.savefig(save_name + '.png', dpi=300)
        plt.savefig(save_name + '.pdf', format='pdf', dpi=300, bbox_inches="tight")
        plt.close(fig)

    @classmethod
    def transient(cls, data_path, save_name) -> None:
        """A simple function to plot the transient stagnation point trace."""
        data = np.loadtxt(data_path, skiprows=1, delimiter=',')

        t = data[:, 0]

        plot_configs = [
            (data[:, 1], r"$\delta, \ m$"),
            (data[:, 2], r"$q_c, \ W/m^2$"),
            (data[:, 3], r"$q_d, \ W/m^2$"),
            (data[:, 4], r"$p, \ Pa$")
        ]

        fig, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True)

        axs[0].set_title("Stagnation Point Trace", fontsize=cls.title_size)

        for i, (y_data, ylabel) in enumerate(plot_configs):
            ax = axs[i]

            ax.plot(t, y_data, linewidth=cls.line_width, color=cls.colours[i % len(cls.colours)])

            ax.set_ylabel(ylabel, fontsize=cls.label_size)
            ax.grid(True, linestyle=':', alpha=cls.transparency)
            ax.tick_params(axis='both', labelsize=cls.tick_size, direction='in')

            ax.yaxis.set_major_locator(LinearLocator(numticks=5))
            ax.yaxis.set_major_formatter(TwoDPFormatter())
            ax.yaxis.get_offset_text().set_fontsize(cls.tick_size)

        axs[-1].set_xlabel(r"$t, \ s$", fontsize=cls.label_size)
        axs[-1].set_xlim(t[0], t[-1])
        axs[-1].xaxis.set_major_locator(LinearLocator(numticks=8))
        axs[-1].xaxis.set_major_formatter(TwoDPFormatter())
        axs[-1].xaxis.get_offset_text().set_fontsize(cls.tick_size)

        plt.tight_layout()
        fig.align_ylabels(axs)
        plt.savefig(save_name + '.png', dpi=300)
        plt.savefig(save_name + '.pdf', format='pdf', dpi=300, bbox_inches="tight")
        plt.close(fig)