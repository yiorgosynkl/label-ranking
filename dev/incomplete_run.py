import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('incomplete_data.csv')

datasets =  ['authorship', 'bodyfat', 'calhousing', 'cpu-small', \
                'glass', 'housing', 'iris', 'stock',  \
                'vehicle', 'vowel', 'wine', 'wisconsin']
models = ['RFC100', 'SVR', 'RFR100']

PUT_YLIM = True

for csv_name in datasets:
    tmp_df = df[df['dataset'] == (' ' + csv_name)]
    for m in models:
        y_list = tmp_df[m]
        y = [float(i.strip().split('±')[0]) for i in y_list]
        x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

        fig, ax = plt.subplots()    
        if PUT_YLIM == True:
            plt.ylim([0, 1])
        plt.plot(x, y)
        ax.set(title=f'accuracy on incomplete data\n(dataset: {csv_name}, model: {m})', ylabel='kendall tau coefficient', xlabel='probability of missing data')
        if PUT_YLIM == False:
            plt.savefig(f'incomplete_analysis/dataset_{csv_name}_model_{m}.png')
        else:
            plt.savefig(f'incomplete_analysis/dataset_{csv_name}_model_{m}_ylim.png')
        plt.close(fig)
