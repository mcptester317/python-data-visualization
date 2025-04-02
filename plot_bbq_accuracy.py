import numpy as np
import matplotlib.pyplot as plt

# Models and their corresponding colors
models = [
    "LLaMA 3.1 8B",
    "Qwen 2.5 14B",
    "DeepSeek LLaMA 3.1 8B",
    "DeepSeek Qwen 2.5 14B",
    "Finetuned LLaMA Best Checkpoint",
    "Finetuned LLaMA Last Checkpoint",
]

colors = ["#f4a261", "#e76f51", "#d62828", "#ff66b2", "#6a0572", "#003f5c"]

# Categories
categories = ["Nationality", "Age", "Religion"]

# Data for the three types of accuracy
overall_acc = [
    [26.79, 29.48, 22.08],
    [22.82, 30.05, 18.58],
    [81.03, 59.23, 56.04],
    [86.59, 63.14, 58.96],
    [92.14, 87.17, 92.67],
    [89.51, 84.54, 90.33],
]

ambig_acc = [
    [5.78, 2.45, 5.33],
    [4.55, 2.99, 4.33],
    [73.45, 42.5, 45.74],
    [79.97, 46.07, 48.67],
    [87.21, 82.93, 90.83],
    [82.47, 75.82, 90.17],
]

disambig_acc = [
    [47.79, 56.52, 38.83],
    [41.1, 57.12, 32.83],
    [88.62, 75.96, 66.34],
    [93.31, 80.33, 69.36],
    [97.08, 91.41, 94.50],
    [96.56, 93.26, 90.50],
]

# Create figure with optimized size and increased height
fig, axes = plt.subplots(1, 3, figsize=(12, 7.5), sharey=True)
fig.suptitle("Accuracy Comparisons Across Models and BBQ Splits", fontsize=14, y=0.98)

# Plot function
def plot_bars(ax, data, title):
    bar_width = 0.12  # Width of individual bars
    x = np.arange(len(categories))  # X positions

    for i, (model, color) in enumerate(zip(models, colors)):
        ax.bar(x + i * bar_width, data[i], width=bar_width, label=model, color=color)

    ax.set_xticks(x + (len(models) / 2 - 0.5) * bar_width)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title(title, fontsize=12)
    
    # Only add y-label to the first subplot to save space
    if ax == axes[0]:
        ax.set_ylabel("Accuracy (%)", fontsize=10)
    
    ax.set_ylim(0, 100)  # Consistent y-axis scale
    
    # Add thin grid lines for readability
    ax.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Make tick labels smaller
    ax.tick_params(axis='both', which='major', labelsize=9)

# Plot each accuracy type
plot_bars(axes[0], overall_acc, "Overall Accuracy")
plot_bars(axes[1], ambig_acc, "Ambig Accuracy")
plot_bars(axes[2], disambig_acc, "Disambig Accuracy")

# Adjust the legend position to be closer to the plot
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.08), 
          ncol=3, frameon=True, title="Models", fontsize=8, title_fontsize=9)

# Adjust spacing between subplots and reduce margins
plt.subplots_adjust(wspace=0.05, bottom=0.18, left=0.05, right=0.98, top=0.90)

if __name__ == "__main__":
    # Show the chart if run directly
    plt.show()
    # Save the figure
    plt.savefig('bbq_model_comparison.png', dpi=300, bbox_inches='tight')
    print("Chart saved as bbq_model_comparison.png")