import numpy as np
import itertools
import pickle
import os
import pandas as pd
import seaborn as sns

import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from scipy.stats import pearsonr, spearmanr

#########################################################
#           Abbreviations    
#########################################################
# Model names
names = {
    'claude-3-7-sonnet-20250219': 'sonnet_3.7',
    'claude-3-haiku-20240307': 'haiku_3',
    'gpt-3.5-turbo-0125': 'gpt_3.5-turbo',
    'gpt-4.1-2025-04-14': 'gpt_4.1',
    'o4-mini-2025-04-16': 'o4-mini',
}

model_names = names.values()

palette = sns.color_palette("Set2", n_colors=len(model_names))
model_colors = dict(zip(model_names, palette))

#############################################
#           AGP Metric Prompt
#############################################

def instructions(datA,datB):
    instruction =   f'''You are an endocrinologist analyzing CGM data for patients with type 1 diabetes. Use the following definitions: Severe hypoglycemia is glucose less than 54 mg/dL, mild hypoglycemia is glucose between 54 mg/dL and 70 mg/dL, mild hyperglycemia is glucose between 180 mg/dL and 250 mg/dL, and severe hyperglycemia is glucose above 250 mg/dL. 
                I will show you data from two patients, Patient A and Patient B. Which patient has better glycemic control?
                Patient A: {datA}
                Patient B: {datB}.
                Your response must be either "A" or "B", with no explanation or additional text.'''
    return instruction

def prompt_metrics(glucose_values):
    glucose_values = np.array(glucose_values)
    total = len(glucose_values)

    if total == 0:
        return {"GRI": np.nan, "VLow": 0, "Low": 0, "High": 0, "VHigh": 0}

    VLow = np.sum(glucose_values < 54) / total
    Low = np.sum((glucose_values >= 54) & (glucose_values < 70)) / total
    tir= np.sum((glucose_values >= 70) & (glucose_values <= 180)) / total

    High = np.sum((glucose_values > 180) & (glucose_values <= 250)) / total
    VHigh = np.sum(glucose_values > 250) / total

    GRI = (3.0 * VLow + 2.4 * Low + 1.6 * VHigh + 0.8 * High) * 100 

    metrics_string =f'''Mean  {np.round(np.mean(glucose_values),2)} mg/dL, Time in range {np.round(tir*100,2)}%, CV {np.round(100*np.std(glucose_values)/np.mean(glucose_values),2)}%, Severe Hypoglycemia {np.round(VLow * 100,2)}%, Mild Hypoglycemia {np.round(Low * 100,2)}%,  Mild Hyperglycemia {np.round(High * 100,2)}%,  Severe Hyperglycemia {np.round(VHigh * 100,2)}%'''

    stats =  {
        "GRI": GRI,
        "Mean": np.mean(glucose_values),
        "CV":np.round(100*np.std(glucose_values)/np.mean(glucose_values),2),
        "TIR": tir*100,
        "VLow (%)": VLow * 100,
        "Low (%)": Low * 100,
        "High (%)": High * 100,
        "VHigh (%)": VHigh * 100
    }
    return metrics_string, stats
    

def load_data_metrics_for_prompt(dataset):
    all_prompts = {}
    for i in range(1,101):

        if i<10:
            df = pd.read_excel(f"data{dataset}_001to100.xlsx", sheet_name=f"Data_00{i}")
        elif i==100:
            df = pd.read_excel(f"data{dataset}_001to100.xlsx", sheet_name=f"Data_{i}")
        else:
            df = pd.read_excel(f"data{dataset}_001to100.xlsx", sheet_name=f"Data_0{i}")

        df = df.reset_index()
        df.index = df['index'].apply(lambda x: f"{(x*5)//60:02d}:{(x*5)%60:02d}")

        df = df.rename_axis("Time")
        df = df.drop(columns = 'index').iloc[:,:14]

        values_raw = df.values.reshape(-1)
        
       
        metrics_string, stats= prompt_metrics(values_raw)
        all_prompts[i] = metrics_string
    return all_prompts
        
def compile_prompt_pairs(directory, individual_data, ):
    info_dict={}
    prompts={}
    i=0
  
    column_pairs = list(itertools.combinations(list(range(1, len(individual_data)+1)), 2))
    print(f'There are {len(column_pairs)} pairs')
    for col1, col2 in column_pairs:
        prompt = instructions(individual_data[col1],individual_data[col2])
        
        pair_df= [col1, col2]  

        info_dict[i]={'col1':col1,'col2':col2,'prompt':prompt}
        prompts[i] = prompt
        i+=1
    pickle.dump(info_dict,open(f'{directory}/prompts_input.p', 'wb'))
    print(f"Saved to {f'{directory}/prompts_input.p'}")
    return prompts, info_dict

def compile_prompt_pairs_reversed(directory, individual_data, ):
    info_dict={}
    prompts={}
    i=0
  
    column_pairs = list(itertools.combinations(list(range(1, len(individual_data)+1)), 2))
    print(f'There are {len(column_pairs)} pairs')
    for col1, col2 in column_pairs:
        prompt = instructions(individual_data[col2],individual_data[col1]) #*** reversed here
        
        pair_df= [col2, col1]  

        info_dict[i]={'col1':col2,'col2':col1,'prompt':prompt}
        prompts[i] = prompt
        i+=1
    pickle.dump(info_dict,open(f'{directory}/prompts_input.p', 'wb'))
    print(f"Saved to {f'{directory}/prompts_input.p'}")
    return prompts, info_dict

#############################################
#          GRI
#############################################
def calculate_GRI(glucose_values):
    glucose_values = np.array(glucose_values)
    total = len(glucose_values)

    if total == 0:
        return {"GRI": np.nan, "VLow": 0, "Low": 0, "High": 0, "VHigh": 0}

    VLow = np.sum(glucose_values < 54) / total
    Low = np.sum((glucose_values >= 54) & (glucose_values <70)) / total
    High = np.sum((glucose_values > 180) & (glucose_values <= 250)) / total
    VHigh = np.sum(glucose_values > 250) / total
    TIR = np.sum((glucose_values >= 70) & (glucose_values <= 180)) / total

    GRI = (3.0 * VLow + 2.4 * Low + 1.6 * VHigh + 0.8 * High) * 100 

    return {
        "GRI": GRI,
        "Mean": np.mean(glucose_values),
        "TIR": TIR*100,
        "VLow (%)": VLow * 100,
        "Low (%)": Low * 100,
        "High (%)": High * 100,
        "VHigh (%)": VHigh * 100
    }





#############################################
#           Analyzing
#############################################

def get_percentiles_fixed(batch_dir):

    with open(f"{batch_dir}/outputs.p", "rb") as f:
            outputs = pickle.load(f)

    with open(f'{batch_dir}/prompts_input.p', 'rb') as f:
        prompts_input = pickle.load(f)
    results = {}   
    for key in outputs:
        split_keys = key.split('-')
        id = int(split_keys[1])

        col1 = prompts_input[id]['col1']
        col2 = prompts_input[id]['col2']
        title = f"{col1}_{col2}"
        
        result = outputs[key]
        result = result.replace('\n', '')
        result = result.replace('.', '')
        result = result.replace(' ', '')
        result = result.replace("'", '')
        result = result.replace('"', '')
        result = result.replace("*", '')
        result = result.replace(":", '')
        result = result.replace("Answer", '')
        result = result.lower()
        if result =='b' or   result =='patientb':
            results[title] = col2
        elif result =='a' or result =='patienta':
            results[title] = col1
        else:
            results[title] = 'None'
            print(f"Error with pair {title}: {result}")
    df = pd.DataFrame(list(results.items()), columns=['pair', 'winner'])

    win_counts = {}
    total = {}
    for _, row in df.iterrows():
        pair = row['pair']
        winner = row['winner']
        a, b = map(int, pair.split('_'))

        if a not in win_counts:
            win_counts[a] = 0
            total[a] = 0
        if b not in win_counts:
            win_counts[b] = 0
            total[b] = 0
        
        if winner == a: 
            win_counts[a] += 1
        elif winner == b: 
            win_counts[b] += 1
        else:
            print('Problem with pair:', pair, 'winner:', winner)
        total[a] += 1
        total[b] += 1
    win_df = pd.DataFrame(list(win_counts.items()), columns=['number', 'wins'])
    print("PROBLEM WITH TOTALS:", [key for key, value in total.items() if value != 99])
    win_df['percentile'] = 100-win_df['wins']
    outcomes_df = win_df.sort_values(by='percentile', ascending=True).reset_index(drop=True)
    with open(f"{batch_dir}/percentiles.p", "wb") as f:
        pickle.dump(outcomes_df, f)
            
    return outcomes_df[['number', 'percentile']],  win_df, df

#############################################
#          Stats and plotting
#############################################

def get_stats(dataset):
    gri_values = {}
    for i in range(1,101):

        if i<10:
            df = pd.read_excel(f"data{dataset}_001to100.xlsx", sheet_name=f"Data_00{i}")
        elif i==100:
            df = pd.read_excel(f"data{dataset}_001to100.xlsx", sheet_name=f"Data_{i}")

        else:
            df = pd.read_excel(f"data{dataset}_001to100.xlsx", sheet_name=f"Data_0{i}")

        df = df.reset_index()
        df.index = df['index'].apply(lambda x: f"{(x*5)//60:02d}:{(x*5)%60:02d}")

        df = df.rename_axis("Time")
        df = df.drop(columns = 'index').iloc[:,:14]

        values_raw = df.values.reshape(-1)
        _, gri= prompt_metrics(values_raw)
        gri_values[i] = gri
        stats = pd.DataFrame(gri_values).transpose()
    return stats

def plot_all_vars(model_dict, dataset, model):
    df = model_dict[dataset][model]
    variables = df.columns[:8].values

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10), sharey=True)

   
    axes = axes.flatten()

    for i, var in enumerate(variables):
        sns.regplot(x=df[var], y=df['percentile'], ax=axes[i],color =model_colors[model],  scatter_kws={'alpha': 0.5, 'color': model_colors[model]})
        axes[i].set_title(f'{var} vs Percentile')
        axes[i].set_xlabel(var)
        if i % 4 == 0:
            axes[i].set_ylabel('Percentile')
        else:
            axes[i].set_ylabel('')

    for j in range(len(variables), len(axes)):
        fig.delaxes(axes[j])

    plt.ylim([0,100])
    plt.suptitle(f'{dataset} - {model} Stats', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  
    plt.savefig(f'Figures/{dataset}_{model}_all_vars.png', dpi=300, bbox_inches='tight')  # Save the figure
    plt.show()


def plot_gri_tir_dual_axis_fixed_scale(df_dict_datasets, datasets, correlation_type='pearson'):
    
    if correlation_type not in ['pearson', 'spearman']:
        raise ValueError("correlation_type must be 'pearson' or 'spearman'")

    model_names = list(df_dict_datasets[datasets[0]].keys())
    num_models = len(model_names)
    num_datasets = len(datasets)

    

    all_gri_vals = []
    all_tir_vals = []

    for dataset in datasets:
        for model in model_names:
            df = df_dict_datasets[dataset][model]
            all_gri_vals.extend(df['GRI'].values)
            all_tir_vals.extend(df['TIR'].values)

    gri_min, gri_max = np.nanmin(all_gri_vals), np.nanmax(all_gri_vals)
    tir_min, tir_max = np.nanmin(all_tir_vals), np.nanmax(all_tir_vals)

    pad = 0.05
    gri_range = gri_max - gri_min
    tir_range = tir_max - tir_min
    gri_lim = (gri_min - pad * gri_range, gri_max + pad * gri_range)
    tir_lim = (tir_min - pad * tir_range, tir_max + pad * tir_range)

    # 
    fig, axes = plt.subplots(
        nrows=num_models,
        ncols=num_datasets,
        figsize=(5 * num_datasets, 3.5 * num_models),
        sharex=True
    )

    set_pretty_seaborn_theme()  

    for i, model_name in enumerate(model_names):
        for j, dataset in enumerate(datasets):
            df = df_dict_datasets[dataset][model_name]
            color = model_colors[model_name]

            ax = axes[i][j] if num_models > 1 else axes[j]

            sns.regplot(
                x=df['percentile'], y=df['GRI'], ax=ax,
                scatter_kws={'alpha': 0.5, 'color': color},
                line_kws={'color': color},
                label='GRI'
            )
            ax.set_ylim(gri_lim)
            if j == 0:
                ax.set_ylabel(f'{model_name}\n\nGRI', color=color)
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([])

            ax.tick_params(axis='y', labelcolor=color)

            r_gri, p_gri = (pearsonr(df['GRI'], df['percentile']) if correlation_type == 'pearson'
                            else spearmanr(df['GRI'], df['percentile']))
            p_gri_str = "p < 0.001" if p_gri < 0.001 else f"p = {p_gri:.4f}"

            # TIR plot on right y
            ax2 = ax.twinx()
            sns.regplot(
                x=df['percentile'], y=df['TIR'], ax=ax2,
                scatter_kws={'alpha': 0.5, 'color': 'gray'},
                line_kws={'color': 'gray'},
                label='TIR'
            )
            ax2.set_ylim(tir_lim)
            if j == num_datasets - 1:
                ax2.set_ylabel('TIR', color='gray')
            else:
                ax2.set_ylabel('')
                ax2.set_yticklabels([])

            ax2.tick_params(axis='y', labelcolor='gray')

            r_tir, p_tir = (pearsonr(df['TIR'], df['percentile']) if correlation_type == 'pearson'
                            else spearmanr(df['TIR'], df['percentile']))
            p_tir_str = "p < 0.001" if p_tir < 0.001 else f"p = {p_tir:.4f}"

            ax.text(0.25, 0.95,
                    f"GRI\n$r$ = {r_gri:.3f}\n{p_gri_str}",
                    transform=ax2.transAxes, fontsize=13, fontweight='bold',
                    ha='left', va='top',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),zorder=10)


            ax2.text(0.95, 0.05,
                     f"TIR\n$r$ = {r_tir:.3f}\n{p_tir_str}",
                     transform=ax2.transAxes, fontsize=13, fontweight='bold',
                     ha='right', va='bottom',
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.75),zorder=10)

            # X and titles
            if i == num_models - 1:
                ax.set_xlabel('Percentile')
            else:
                ax.set_xlabel('')
                ax.set_xticklabels([])

            if i == 0:
                ax.set_title(dataset)

    plt.suptitle(f'GRI & TIR vs Percentile (Dual Y-Axis, Consistent Scales) — {correlation_type.title()}',
                 fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if not os.path.exists('Figures'):
        os.makedirs('Figures')
    plt.savefig(f'Figures/GRI_TIR_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()



import matplotlib as mpl


########################################################
#            Regression Plots
#########################################################


def compare_regression_coefficients_sm(datasets, all_dict, true_coeffs, scaled=False):
    num_datasets = len(datasets)
    fig, axes = plt.subplots(num_datasets, 1, figsize=(12, 3 * num_datasets), sharex=True)



    for ax, dataset in zip(axes, datasets):
        dfs_dict = all_dict[dataset]
        drop_columns = ['percentile', 'GRI', 'TIR', 'CV', 'Mean']
        true_coeffs = true_coeffs
        scaled = False

        first_df = next(iter(dfs_dict.values()))
        X_sample = first_df.drop(columns=drop_columns)
        feature_names = list(X_sample.columns)

        x = np.arange(len(feature_names))
        width = 0.15
        num_models = len(dfs_dict) + 1 

        
        gt_offset = (0 - num_models / 2) * width + width / 2
        gt_positions = x + gt_offset
        ax.bar(gt_positions, true_coeffs, width, label='GRI', color='black')
        for xpos, val in zip(gt_positions, true_coeffs):
            ax.text(xpos, val, f'{val:.2f}', ha='center', va='bottom', fontsize=7)

        all_coeffs = []

        for i, (dataset_name, df) in enumerate(dfs_dict.items(), start=1):
            X = df.drop(columns=drop_columns)
            X = X.rename(columns={'VLow (%)': 'Severe Hypo', 'Low (%)': 'Mild Hypo', 'High (%)': 'Mild Hyper', 'VHigh (%)': 'Severe Hyper'})
            y = df['percentile']

            if scaled:
                X = StandardScaler().fit_transform(X)
                y = StandardScaler().fit_transform(y.values.reshape(-1, 1)).ravel()

            X_model = sm.add_constant(X)
            model = sm.OLS(y, X_model).fit()

            print(f"\n--- OLS Summary for {dataset_name} ---")
            print(model.summary())

            coeffs = model.params[1:]
            conf_int = model.conf_int().iloc[1:]

            lower_err = coeffs - conf_int[0]
            upper_err = conf_int[1] - coeffs
            yerr = np.vstack([lower_err, upper_err])

            all_coeffs.extend(coeffs)

            offset = (i - num_models / 2) * width + width / 2
            bar_positions = x + offset

            for xpos, val in zip(bar_positions, coeffs):
                ax.text(xpos, val, f'{val:.2f}', ha='center', va='bottom', fontsize=9)

        ax.set_ylim(-6, 6)

        ax.set_ylabel("Coefficient (Standardized)" if scaled else "Coefficient")
        ax.set_title(f'{dataset}')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(feature_names, fontweight='bold', rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('Figures/Regression_Coefficients_All_Datasets.png', dpi=300, bbox_inches='tight')
    plt.show()


#########################################################
#            Case Studies
#########################################################
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

def plot_two_rows(row1, row2,save_dir, model, dataset, idx1, idx2):
    columns = ['VLow (%)', 'Low (%)', 'TIR', 'High (%)', 'VHigh (%)']
    colors = [
        '#A05252',  
        '#E99696',  
        '#6BA85A',  
        '#FFB87F', 
        '#D9853B'   
    ]

    def calculate_label_positions(pcts, min_gap=2):
        centers = []
        current_bottom = 0
        for p in pcts:
            centers.append(current_bottom + p/2)
            current_bottom += p
        centers = np.array(centers)
        for i in range(1, len(centers)):
            if centers[i] - centers[i-1] < min_gap:
                shift = min_gap - (centers[i] - centers[i-1])
                centers[i:] += shift
        return centers

    fig, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.2
    x_positions = [0, .6]

    for i, row in enumerate([row1, row2]):
        values = [
            row['VLow (%)'],
            row['Low (%)'],
            row['TIR'],
            row['High (%)'],
            row['VHigh (%)']
        ]

        bottom = 0
        for val, color in zip(values, colors):
            ax.bar(x_positions[i], val, bottom=bottom, color=color, width=bar_width)
            bottom += val

        label_ys = calculate_label_positions(values)

        for y, val in zip(label_ys, values):
            label_text = f"{val:.1f}%"
            
            ax.text(
                x_positions[i] + bar_width / 2 + 0.05, 
                y, 
                label_text,
                ha='left', 
                va='center', 
                fontsize=11,       
                fontweight='bold',  
                color='black',
                linespacing=2       
            )

        gri = row.get('GRI', 'N/A')
        gri = round(gri,2)
        percentile = row.get('percentile', 'N/A')
        ax.text(x_positions[i], bottom + 5, f"GRI: {gri}\nPercentile: {percentile}",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.axis('off')

    legend_handles = [Patch(color=color, label=col) for col, color in zip(columns, colors)]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.ylim(0, 120)
    plt.xlim(-0.5, 1.5)
    plt.tight_layout()
    plt.title(f'{model} - {dataset} - Comparison of ID:{idx1} and ID:{idx2}', fontsize=16, fontweight='bold')
    if not os.path.exists('Figures/case_studies'):
        os.makedirs('Figures/case_studies')
    plt.savefig(f'Figures/case_studies/{save_dir}', bbox_inches='tight')

    plt.show()

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np



def plot_two_cases_with_votes(row1, row2, model_votes, dataset, idx1, idx2,model_dfs):
    colors = [
        '#A05252',  
        '#E99696',  
        '#6BA85A',  
        '#FFB87F', 
        '#D9853B'   
    ]

    def calculate_label_positions(pcts, min_gap=5):
        centers = []
        current_bottom = -.2
        for p in pcts:
            centers.append(current_bottom + p/2)
            current_bottom += p
        centers = np.array(centers)
        for i in range(1, len(centers)):
            if centers[i] - centers[i-1] < min_gap:
                shift = min_gap - (centers[i] - centers[i-1])
                centers[i:] += shift
        return centers

    fig, ax = plt.subplots(figsize=(5, 5))
    bar_width = 0.2
    x_positions = [0, .6]

    # Plot bars and annotate percentages
    for i, row in enumerate([row1, row2]):
        values = [
            row['VLow (%)'],
            row['Low (%)'],
            row['TIR'],
            row['High (%)'],
            row['VHigh (%)']
        ]

        bottom = 0
        for val, color in zip(values, colors):
            ax.bar(x_positions[i], val, bottom=bottom, color=color, width=bar_width)
            bottom += val

        label_ys = calculate_label_positions(values)
        for y, val, color in zip(label_ys, values, colors):
            label_text = f"{val:.1f}%"
            ax.text(
                x_positions[i] + bar_width / 2 + 0.05,
                y,
                label_text,
                ha='left',
                va='center',
                fontsize=11,
                fontweight='bold',
                color=color,
                linespacing=2
            )
        
        gri = row.get('GRI', 'N/A')
        mean = row.get('Mean', 'N/A')
        mean = round(mean,2)
        gri = round(gri, 2)

        cv = row.get('CV', 'N/A')
        cv = round(cv,2)
        ax.text(x_positions[i], bottom + 5, f"GRI: {gri}",
                ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.text(x_positions[i], bottom + 12, f"Mean: {mean}",
                ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.text(x_positions[i], bottom + 19, f"CV: {cv}",
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    y_star = -5  

    for i, idx in enumerate([idx1, idx2]):
        # Get models that voted for this idx
        models_for_idx = [m for m, vote in model_votes.items() if vote == idx]
        if models_for_idx:
            annotated_names = []
            count = 0
            for model in models_for_idx:
                display_name = names.get(model, model)
                annotated_names.append(f"{display_name}")
                count += 1
            if count <6:
                annotated_names.append(' ')

            text = '\n \n'.join(annotated_names)

            ax.text(x_positions[i], y_star, text,
                    ha='center', va='top',  
                    fontsize=12, fontweight='bold', color='black')
    ax.axis('off')
    plt.ylim(0, 125)
    plt.xlim(-0.5, 1.5)
    plt.tight_layout()

    plt.title(f'{dataset} - Comparison of ID:{idx1} and ID:{idx2}', fontsize=16, fontweight='bold')
    plt.savefig(f'Figures/case_studies{dataset}_{model}_ID{idx1}_vs_ID{idx2}.png', bbox_inches='tight')
    plt.show()