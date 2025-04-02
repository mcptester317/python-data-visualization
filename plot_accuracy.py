import matplotlib.pyplot as plt
import numpy as np

# Data
accuracy_types = ['Overall accuracy', 'Ambig accuracy', 'Disambig accuracy']
llama_3_1_8b = [29.78, 2.98, 56.63]
qwen_2_5_14b = [34.32, 3.45, 65.2]
deepseek_llama3_1_8b = [68.19, 67.27, 69.1]
deepseek_qwen_2_5_14b = [75.75, 81.62, 69.27]

# Set up the positions for the bars
x = np.arange(len(accuracy_types))
width = 0.2  # Width of the bars

# Colors for the bars
colors = ['#FFA500',  # Orange for LLaMA 3.1 8B
         '#FF6700',   # Dark Orange for Qwen 2.5 14B
         '#DC143C',   # Crimson for DeepSeek LLaMA 3.1 8B
         '#FF69B4']   # Pink for DeepSeek Qwen 2.5 14B

# Create the figure and bars with adjusted legend placement
fig, ax = plt.subplots(figsize=(10, 6))  # Standardized figure size
ax.bar(x - 1.5 * width, llama_3_1_8b, width, label="LLaMA 3.1 8B", color=colors[0])
ax.bar(x - 0.5 * width, qwen_2_5_14b, width, label="Qwen 2.5 14B", color=colors[1])
ax.bar(x + 0.5 * width, deepseek_llama3_1_8b, width, label="DeepSeek LLaMA 3.1 8B", color=colors[2])
ax.bar(x + 1.5 * width, deepseek_qwen_2_5_14b, width, label="DeepSeek Qwen 2.5 14B", color=colors[3])

# Labels and title
ax.set_xlabel("Accuracy Types", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Model Performance Comparison on SQuAD v2 Dataset", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(accuracy_types, rotation=0)

# Set y-axis range from 0 to 100 for consistency
ax.set_ylim(0, 100)

# Add grid for better readability
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Adjust legend placement below the plot
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

# Add value labels on top of each bar
def add_value_labels(ax, spacing=5):
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width()/2

        label = f"{y_value:.1f}"
        ax.annotate(label, (x_value, y_value), xytext=(0, spacing),
                   textcoords="offset points", ha='center', va='bottom')

add_value_labels(ax)

# Adjust layout to prevent label cutoff
plt.tight_layout()

if __name__ == "__main__":
    # Show the chart if run directly
    plt.show()
    # Save the figure
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("Chart saved as model_comparison.png")