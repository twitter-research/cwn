import os
import sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import seaborn as sns
sns.set_style("whitegrid", {'legend.frameon': False})

from matplotlib import cm
from matplotlib import pyplot as plt
from definitions import ROOT_DIR

def run(exps, codenames, plot_name):

    # Meta
    family_names = [
        'SR(16,6,2,2)',
        'SR(25,12,5,6)',
        'SR(26,10,3,4)',
        'SR(28,12,6,4)',
        'SR(29,14,6,7)',
        'SR(35,16,6,8)',
        'SR(35,18,9,9)',
        'SR(36,14,4,6)',
        'SR(40,12,2,4)']

    # Retrieve results
    base_path = os.path.join(ROOT_DIR, 'exp', 'results')
    results = list()
    for e, exp_path in enumerate(exps):
        path = os.path.join(base_path, exp_path, 'result.txt')
        results.append(dict())
        with open(path, 'r') as handle:
            found = False
            f = 0
            for line in handle:
                if not found:
                    if line.strip().startswith('Mean'):
                        mean = float(line.strip().split(':')[1].strip())
                        found = True
                    else:
                        continue
                else:
                    std = float(line.strip().split(':')[1].strip())
                    results[-1][family_names[f]] = (mean, std)
                    f += 1
                    found = False
            assert f == len(family_names)

    # Set colours
    colors = cm.get_cmap('tab20c').colors[1:4] + cm.get_cmap('tab20c').colors[5:9]
    matplotlib.rc('axes', edgecolor='black', lw=0.25)
    a = np.asarray([83, 115, 171])/255.0 +0.0
    b = np.asarray([209, 135, 92])/255.0 +0.0
    colors = [a, a +0.13, a +0.2, b, b +0.065, b +0.135]

    # Set plotting
    num_families = len(family_names)
    num_experiments = len(results)
    sep = 1.75
    width = 0.7
    disp = num_experiments * width + sep
    xa = np.asarray([i*disp for i in range(num_families)])
    xs = [xa + i*width for i in range(num_experiments//2)] + [xa + i*width + sep*0.25 for i in range(num_experiments//2, num_experiments)]
    plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True
    plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False   
    print(sns.axes_style()) 
    matplotlib.rc('axes', edgecolor='#c4c4c4', linewidth=0.9)

    # Plot
    plt.figure(dpi=300, figsize=(9,6.6))
    plt.grid(axis='x', alpha=0.0)
    for r, res in enumerate(results):
        x = xs[r]
        y = [10+res[family][0] for family in sorted(res)]
        yerr = [res[family][1] for family in sorted(res)]
        plt.bar(x, y, yerr=yerr, bottom=-10, color=colors[r], width=width, 
                label=codenames[r], ecolor='grey', error_kw={'lw': 0.75, 'capsize':0.7},
                edgecolor='white')
                # hatch=('//' if r<3 else '\\\\'))
    plt.axhline(y=1.0, color='indianred', lw=1.5, label='3WL')
    plt.ylim([-0.000005, 2])
    plt.yscale(matplotlib.scale.SymmetricalLogScale(axis='y', linthresh=0.00001))
    plt.xticks(xa+3*width, family_names, fontsize=12, rotation=315, ha='left')
    plt.yticks([0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0], fontsize=12)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1, 4, 2, 5, 3, 6] + [0]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=10, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.15))
    plt.xlabel('Family', fontsize=15)
    plt.ylabel('Failure rate', fontsize=15, labelpad=-580, rotation=270)
    plt.tight_layout()
    plt.savefig(f'./sr_exp_{plot_name}.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()

if __name__ == '__main__':

    # Standard args
    passed_args = sys.argv[1:]
    codenames = list()
    exps = list()
    plot_name = passed_args[0]
    for a, arg in enumerate(passed_args[1:]):
        if a % 2 == 0:
            exps.append(arg)
        else:
            codenames.append(arg)
    assert len(codenames) == len(exps) == 6
    run(exps, codenames, plot_name)
