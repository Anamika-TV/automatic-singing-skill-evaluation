import pandas as pd

# Input and output file paths
INPUT_FILE = "dataset/labels.csv"
OUTPUT_FILE = "dataset/labels_classification.csv"

# Load original regression labels
df = pd.read_csv(INPUT_FILE, names=["filename", "score"])

# Function to convert score to class
def score_to_class(score):
    if 8 <= score <= 10:
        return 2   # Good
    elif 4 <= score <= 7:
        return 1   # Intermediate
    elif 0 <= score <= 3:
        return 0   # Bad
    else:
        return None  # In case something unexpected appears

# Apply conversion
df["class"] = df["score"].apply(score_to_class)

# Save new classification CSV
df[["filename", "class"]].to_csv(OUTPUT_FILE, index=False, header=False)

print("Classification labels created successfully!")
print("\nClass Distribution:")
print(df["class"].value_counts())