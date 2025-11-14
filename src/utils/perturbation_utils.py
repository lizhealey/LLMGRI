
import numpy as np

from utils.utils import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt


def compare_regression_coefficients_sm_perturbation_mild(all_dict, true_coeffs, scaled=False):
    fig, ax = plt.subplots(figsize=(12, 3))
   

    dfs_dict = all_dict
    drop_columns = ['output', 'GRI', ]
    true_coeffs = true_coeffs
    scaled = False

    first_df = next(iter(dfs_dict.values()))
    X_sample = first_df.drop(columns=drop_columns)
    feature_names = list(X_sample.columns[:2])

    x = np.arange(len(feature_names))
    width = 0.15
    num_models = len(dfs_dict) + 1 

    
    gt_offset = (0 - num_models / 2) * width + width / 2
    gt_positions = x + gt_offset
    ax.bar(gt_positions, true_coeffs, width, label='GRI', color='black')
    for xpos, val in zip(gt_positions, true_coeffs):
        ax.text(xpos, val, f'{val:.2f}', ha='center', va='bottom')

    all_coeffs = []

    for i, (dataset_name, df) in enumerate(dfs_dict.items(), start=1):
        X = df[['hypo', 'hyper']]
        X = X.rename(columns={'hypo': 'Mild Hypo', 'hyper': 'Mild Hyper'})
        y = df['output']

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

        bars = ax.bar(
            bar_positions, coeffs, width,
            yerr=yerr, capsize=4,
            error_kw=dict(ecolor='grey', alpha=0.7, lw=1),color = model_colors[names[dataset_name]],
            label=f'{names[dataset_name]} (R² = {model.rsquared:.2f})'
        )

        for xpos, val in zip(bar_positions, coeffs):
            ax.text(xpos, val, f'{val:.2f}', ha='center', va='bottom')

        ymin = min(all_coeffs + list(true_coeffs)) * 1.1
        ymax = 6
        #ax.set_ylim(-1, 3)

        ax.set_ylabel("Coefficient (Standardized)" if scaled else "Coefficient")
        #ax.set_title(f'{dataset}')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(['Mild Hypo','Mild Hyper'], fontweight='bold', rotation=0, ha='right')

    plt.tight_layout()
    plt.savefig('Figures/Regression_perturbatio_mild.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_regression_coefficients_sm_perturbation_severe(all_dict, true_coeffs, scaled=False):
    fig, ax = plt.subplots(figsize=(12, 3))
    

    dfs_dict = all_dict
    drop_columns = ['output', 'GRI', ]
    true_coeffs = true_coeffs
    scaled = False

    first_df = next(iter(dfs_dict.values()))
    X_sample = first_df.drop(columns=drop_columns)
    feature_names = list(X_sample.columns[:2])

    x = np.arange(len(feature_names))
    width = 0.15
    num_models = len(dfs_dict) + 1 

    
    gt_offset = (0 - num_models / 2) * width + width / 2
    gt_positions = x + gt_offset
    ax.bar(gt_positions, true_coeffs, width, label='GRI', color='black')
    for xpos, val in zip(gt_positions, true_coeffs):
        ax.text(xpos, val, f'{val:.2f}', ha='center', va='bottom')

    all_coeffs = []

    for i, (dataset_name, df) in enumerate(dfs_dict.items(), start=1):
        X = df[['hypo', 'hyper']]
        X = X.rename(columns={'hypo': 'Severe Hypo', 'hyper': 'Severe Hyper'})
        y = df['output']

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
            ax.text(xpos, val, f'{val:.2f}', ha='center', va='bottom')



        ax.set_ylabel("Coefficient (Standardized)" if scaled else "Coefficient")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(['Severe Hypo','Severe Hyper'], fontweight='bold', rotation=0, ha='right')

    plt.tight_layout()
    plt.savefig('Figures/Regression_perturbation_severe.png', dpi=300, bbox_inches='tight')
    plt.show()


############################# Perturbation Generation #############################

## Short prompt metrics for perturbations
def prompt_metrics_short(glucose_values):
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

    metrics_string =f'''Time in range {np.round(tir*100,2)}%, Severe Hypoglycemia {np.round(VLow * 100,2)}%, Mild Hypoglycemia {np.round(Low * 100,2)}%,  Mild Hyperglycemia {np.round(High * 100,2)}%,  Severe Hyperglycemia {np.round(VHigh * 100,2)}%'''

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

def load_data_metrics_for_prompt_short(dataset):
    # Loads just severe 
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
        
 
        metrics_string, stats= prompt_metrics_short(values_raw)
        all_prompts[i] = metrics_string
    return all_prompts

########### Perturbation Generation for Severe Hypoglycemia ####
def generate_tir_dict_severehypo():
    hypo_range = [0,.5,1,1.5,2,2.5,3,3.5,4]
    hyper_range = [0,.5,1,1.5,2,2.5,3,3.5,4]
    tir_dict = {}
    for severe_hypo in hypo_range:      
        for severe_hyper in hyper_range:  
            hyper = 30
            hypo = 2

            tir = 100 - hyper - hypo - severe_hypo - severe_hyper
            if tir < 0:
                continue  
            key = (severe_hypo, severe_hyper, tir)
            value = (
                f"Time in range {tir}%,  Severe Hypoglycemia {severe_hypo}%, "
                f"Mild Hypoglycemia {hypo}%, Mild Hyperglycemia {hyper}%,  "
                f"Severe Hyperglycemia {severe_hyper}%."
            )
            tir_dict[key] = value
            print(tir)
    return tir_dict

def generate_tir_dict():
    hypo_range = list(range(0, 10, 1))
    hyper_range = list(range(10, 30, 1))  

    tir_dict = {}
    for hypo in hypo_range:      
        for hyper in hyper_range:  
            tir = 100 - hyper - hypo
            severe_hypo = 0
            severe_hyper = 0
            if tir < 0:
                continue  
            key = (hypo, hyper, tir)
            value = (
                f"Time in range {tir}%,  Severe Hypoglycemia {severe_hypo}%, "
                f"Mild Hypoglycemia {hypo}%, Mild Hyperglycemia {hyper}%,  "
                f"Severe Hyperglycemia {severe_hyper}%."
            )
            tir_dict[key] = value
            print(tir)
    return tir_dict