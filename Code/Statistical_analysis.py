import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import friedmanchisquare, f_oneway, shapiro, wilcoxon, levene

class ModelStatisticalEvaluator:
    def __init__(self, csv_path):
        self.cv_df = pd.read_csv(csv_path)
        self.model_groups = self.cv_df.groupby("Model")
        self.metrics = ["Dice", "IoU", "Hausdorff"]
        self.results = {
            "Metric": [], "Test": [], "Statistic": [], "p-value": []
        }
        self.output_dir = "plots"
        os.makedirs(self.output_dir, exist_ok=True)

    def add_result(self, metric, test, stat, pval):
        self.results["Metric"].append(metric)
        self.results["Test"].append(test)
        self.results["Statistic"].append(round(stat, 4))
        self.results["p-value"].append("<0.0001" if pval < 0.0001 else round(pval, 4))

    def run_tests(self):
        for metric in self.metrics:
            model_data = [g[metric].values for _, g in self.model_groups]
            friedman_stat, friedman_p = friedmanchisquare(*model_data)
            self.add_result(metric, "Friedman", friedman_stat, friedman_p)

            anova_stat, anova_p = f_oneway(*model_data)
            self.add_result(metric, "ANOVA", anova_stat, anova_p)

            levene_stat, levene_p = levene(*model_data)
            self.add_result(metric, "Levene", levene_stat, levene_p)

        trionix = self.cv_df[self.cv_df["Model"] == "Trionix"].reset_index()
        for metric in self.metrics:
            for model in ["U-Net", "Hybrid U-Net", "Swin Transformer ViT"]:
                other = self.cv_df[self.cv_df["Model"] == model].reset_index()
                wilcoxon_stat, wilcoxon_p = wilcoxon(trionix[metric], other[metric])
                self.add_result(metric, f"Wilcoxon (Trionix vs {model})", wilcoxon_stat, wilcoxon_p)

        for metric in self.metrics:
            stat, p = shapiro(trionix[metric])
            self.add_result(metric, "Shapiro-Wilk (Trionix)", stat, p)

    def plot_visualizations(self):
        model_order = ["Trionix", "U-Net", "Hybrid U-Net", "Swin Transformer ViT"]
        colors = sns.color_palette("Set2")

        fig, axes = plt.subplots(3, 4, figsize=(22, 15))
        
        for i, metric in enumerate(self.metrics):
            # Boxplot
            sns.boxplot(ax=axes[i, 0], data=self.cv_df, x="Model", y=metric, order=model_order, palette=colors)
            axes[i, 0].set_title(f'{metric} Score - Boxplot')
            axes[i, 0].set_ylabel(metric)
            axes[i, 0].set_xlabel("Model")

            # Violin Plot
            sns.violinplot(ax=axes[i, 1], data=self.cv_df, x="Model", y=metric, order=model_order, palette=colors, inner="quart")
            axes[i, 1].set_title(f'{metric} Score - Violin Plot')
            axes[i, 1].set_ylabel(metric)
            axes[i, 1].set_xlabel("Model")

            # Q-Q Plot
            stats.probplot(self.cv_df[self.cv_df["Model"] == "Trionix"][metric], dist="norm", plot=axes[i, 2])
            axes[i, 2].set_title(f'{metric} Score - Q-Q Plot (Trionix)')
            axes[i, 2].set_xlabel('Theoretical Quantiles')
            axes[i, 2].set_ylabel('Sample Quantiles')

            # Bland–Altman Plot
            trionix_vals = self.cv_df[self.cv_df["Model"] == "Trionix"][metric].values
            other_models_avg = (
                self.cv_df[self.cv_df["Model"] != "Trionix"]
                .groupby("Fold")[metric]
                .mean()
                .values
            )
            avg = (trionix_vals + other_models_avg) / 2
            diff = trionix_vals - other_models_avg
            mean_diff = np.mean(diff)
            std_diff = np.std(diff)

            axes[i, 3].scatter(avg, diff, color=colors[i], alpha=0.7)
            axes[i, 3].axhline(mean_diff, color='red', linestyle='--', label='Mean Diff')
            axes[i, 3].axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle='--', label='±1.96 SD')
            axes[i, 3].axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle='--')
            axes[i, 3].set_title(f'{metric} - Bland-Altman (Trionix vs Others Avg)')
            axes[i, 3].set_xlabel('Average of Trionix & Others')
            axes[i, 3].set_ylabel('Difference (Trionix - Others Avg)')
            axes[i, 3].annotate(f'Mean = {mean_diff:.4f}', xy=(np.min(avg), mean_diff + 0.001), color='red')
            axes[i, 3].annotate(f'+1.96 SD = {mean_diff + 1.96*std_diff:.4f}', xy=(np.min(avg), mean_diff + 1.96*std_diff + 0.001), color='gray')
            axes[i, 3].annotate(f'-1.96 SD = {mean_diff - 1.96*std_diff:.4f}', xy=(np.min(avg), mean_diff - 1.96*std_diff - 0.001), color='gray')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(self.output_dir, "model_metric_comparison_extended.png"), dpi=300)
        plt.show()

    def get_results(self):
        return pd.DataFrame(self.results)
