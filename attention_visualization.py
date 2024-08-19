import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import os

attributes = ['predict_event', 'pleasantness', 'attention',
'other_responsblt', 'chance_control', 'social_norms']

# attributes = ['predict_event',  'attention',
# 'other_responsblt', 'chance_control', 'social_norms']


# # Get the current working directory
# cwd = os.getcwd()
# os.chdir(cwd)

# Load the dataset
df = pd.read_csv('output/cls_to_appraisal_attention_data.csv')


# Convert string representations of lists into actual lists and then into numpy arrays
df['cls_to_appraisals_avg'] = df['cls_to_appraisals_avg'].apply(ast.literal_eval).apply(np.array)
df['appraisals_to_cls_avg'] = df['appraisals_to_cls_avg'].apply(ast.literal_eval).apply(np.array)

# Define a function to plot attention, allowing either individual head visualization or aggregated
def plot_attention1(data, title, aggregate_heads=True):
    if aggregate_heads:
        # Ensure the data is two-dimensional
        if data.ndim == 1:
            data = data.reshape(1, -1)  # Reshape to make it 2D if it's 1D
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(data, annot=True, cmap='viridis', fmt=".2f", xticklabels=attributes)
        plt.title(title)
        plt.xlabel('Appraisal Attributes')
        plt.ylabel('Attention')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better fit
        plt.tight_layout()  # Adjust layout to make room for label rotation
        plt.savefig(f"output/{title.replace(' ', '_').replace('/', '_')}.png")  # Save figure
        plt.close()  # Close the figure to free up memory
    else:
        # Plot each head separately
        num_heads = data.shape[0]
        fig, axes = plt.subplots(num_heads, 1, figsize=(10, num_heads * 5))
        for i, ax in enumerate(axes.flat):
            head_data = data[i, :]
            if head_data.ndim == 1:
                head_data = head_data.reshape(1, -1)  # Ensure 2D for heatmap

            sns.heatmap(head_data, annot=True, cmap='viridis', fmt=".2f", ax=ax, xticklabels=attributes)
            ax.set_title(f"{title} - Head {i+1}")
            ax.set_xlabel('Appraisal Attributes')
            ax.set_ylabel('Attention')
        
        plt.xticks(rotation=45, ha='right')  
        plt.tight_layout()  
        plt.savefig(f"output/{title.replace(' ', '_').replace('/', '_')}.png")  # Save figure with title indicating the head number
        plt.close()  # Close the plot to free up memory



def plot_attention(data, title, aggregate_heads=True):
    if aggregate_heads:
        # Ensure the data is two-dimensional
        if data.ndim == 1:
            data = data.reshape(1, -1)  # Reshape to make it 2D if it's 1D

        # Calculate the mean across all heads (assuming heads are along axis 0)
        mean_data = data.mean(axis=0)
        data_with_mean = np.vstack([data, mean_data])  # Append mean as a new row

        # Calculate the total attention to all tokens from the CLS token
        total_attention = np.sum(data)

        # Calculate the percentage of attention to appraisal attributes
        appraisal_attention_percentage = (np.sum(data_with_mean[:, -len(attributes):], axis=1) / total_attention) * 100
        appraisal_attention_percentage = appraisal_attention_percentage.reshape(-1, 1)  # Reshape for concatenation

        # Append the percentage as a new column
        data_with_percentage = np.hstack([data_with_mean, appraisal_attention_percentage])

        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(data_with_percentage, annot=True, cmap='viridis', fmt=".2f",
                        xticklabels=attributes + ['% Total Attn'])
        ax.set_title(title)
        ax.set_xlabel('Appraisal Attributes and Percentage')
        ax.set_ylabel('Attention')
        ax.set_xticklabels(attributes + ['% Total Attn'], rotation=45, ha='right')  # Rotate x-axis labels for better fit
        # Annotate the last row as the mean
        for tick in ax.get_yticklabels():
            if tick.get_text() == str(data.shape[0]):
                tick.set_color('red')
                tick.set_fontweight('bold')
        plt.tight_layout()  # Adjust layout to make room for label rotation
        plt.savefig(f"output/{title.replace(' ', '_').replace('/', '_')}.png")  # Save figure
        plt.close()  # Close the figure to free up memory
    else:
        # Plot each head separately and include the mean as an additional subplot
        num_heads = data.shape[0]
        fig, axes = plt.subplots(num_heads + 1, 1, figsize=(10, (num_heads + 1) * 5))  # Include one extra subplot for the mean
        for i, ax in enumerate(axes[:-1]):  # Iterate through original number of heads
            head_data = data[i, :]
            if head_data.ndim == 1:
                head_data = head_data.reshape(1, -1)  # Ensure 2D for heatmap

            sns.heatmap(head_data, annot=True, cmap='viridis', fmt=".2f", ax=ax, xticklabels=attributes)
            ax.set_title(f"{title} - Head {i+1}")
            ax.set_xlabel('Appraisal Attributes')
            ax.set_ylabel('Attention')
            ax.set_xticklabels(attributes, rotation=45, ha='right')

        # Plot mean of all heads in the last subplot
        mean_data = data.mean(axis=0).reshape(1, -1)
        ax = axes[-1]
        sns.heatmap(mean_data, annot=True, cmap='viridis', fmt=".2f", ax=ax, xticklabels=attributes)
        ax.set_title(f"{title} - Mean of All Heads")
        ax.set_xlabel('Appraisal Attributes')
        ax.set_ylabel('Mean Attention')
        ax.set_xticklabels(attributes, rotation=45, ha='right')

        plt.tight_layout()  # Adjust layout to make room for label rotation
        plt.savefig(f"output/{title.replace(' ', '_').replace('/', '_')}.png")  # Save figure with title indicating the head number
        plt.close()  # Close the plot to free up memory

def save_attention_to_csv(data, title, attributes):
    if data.ndim == 1:
        data = data.reshape(1, -1)  # Reshape to make it 2D if it's 1D

    # Calculate the mean across all heads if the data includes multiple heads, and consider it as aggregate
    mean_data = data.mean(axis=0)
    data_with_mean = np.vstack([data, mean_data])  # Append mean as a new row

    # Calculate the total attention to all tokens from the CLS token
    total_attention = np.sum(data)

    # Calculate the percentage of attention to appraisal attributes
    appraisal_attention_percentage = (np.sum(data_with_mean[:, -len(attributes):], axis=1) / total_attention) * 100
    appraisal_attention_percentage = appraisal_attention_percentage.reshape(-1, 1)  # Reshape for concatenation

    # Append the percentage as a new column
    data_with_percentage = np.hstack([data_with_mean, appraisal_attention_percentage])

    # Prepare DataFrame
    columns = attributes + ['% Total Attn']
    df = pd.DataFrame(data_with_percentage, columns=columns)
    
    # Save to CSV
    csv_filename = f"output/{title.replace(' ', '_').replace('/', '_')}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")


# # # Visualize attention for a single sample
# sample_data = df.iloc[0]['cls_to_appraisals_avg']
# plot_attention(sample_data, "Attention from CLS to Appraisals - Sample 1", aggregate_heads=False)

# Aggregated visualization by emotion label
for label in df['emotion_label'].unique():
    mean_attention = np.mean(np.stack(df[df['emotion_label'] == label]['cls_to_appraisals_avg']), axis=0)
    plot_attention(mean_attention, f"Average Attention from CLS to Appraisals - Emotion {label}", aggregate_heads=True)
    save_attention_to_csv(mean_attention, f"Attention Data- Emotion {label}", attributes)
    
