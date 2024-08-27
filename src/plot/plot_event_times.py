import pandas as pd
import config as cfg
from tools.formatter import Formatter
from tools.data_loader import DataLoader

matplotlib_style = 'default'
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)
plt.rcParams.update({'axes.labelsize': 'x-large',
                     'axes.titlesize': 'x-large',
                     'font.size': 14.0,
                     'text.usetex': True,
                     'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{bm}'})

DATASET_NAME = "xjtu"
AXIS = "X"
BEARING_IDS = [1, 2, 3, 4, 5]

if __name__ == "__main__":
    for condition in cfg.CONDITIONS:
        for pct in cfg.CENSORING_LEVELS:
            dl = DataLoader(DATASET_NAME, AXIS, condition).load_data()
            data = pd.DataFrame()
            for bearing_id in BEARING_IDS:
                df = dl.make_moving_average(bearing_id)
                df = Formatter.add_random_censoring(df, pct)
                df = df.sample(frac=1, random_state=0)
                data = pd.concat([data, df], axis=0)
            data = data.reset_index(drop=True)
            fig, ax = plt.subplots()
            val, bins, patches = plt.hist((data["Survival_time"][data["Event"]],
                                           data["Survival_time"][~data["Event"]]),
                                          bins=10,
                                          stacked=False)
            plt.legend(patches, ["Event time (" + r"$\delta_{i} = 1$" + ")", "Censoring time (" + r"$\delta_{i} = 0$" + ")"])
            plt.xlabel("Time (min)")
            plt.ylabel("Number of occurrences")
            plt.tight_layout()
            ax.grid(which='minor', alpha=0.2)
            ax.grid(which='major', alpha=0.5)
            plt.savefig(f'{cfg.PLOTS_DIR}/event_times_C{condition+1}_cens_{int(pct*100)}.pdf',
                        format='pdf', bbox_inches="tight")
            plt.close()
        