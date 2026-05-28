import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# -------------------------
# Load features and labels
# -------------------------
X = np.load("features/engineered_features.npy")
y = np.load("features/engineered_labels.npy")

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# Reduce to 2D using t-SNE
# -------------------------
print("🔄 Reducing to 2D...")
tsne = TSNE(n_components=2, random_state=42, perplexity=20, learning_rate=200)
X_2d = tsne.fit_transform(X_scaled)

# -------------------------
# BRIGHT, VIBRANT COLORS - NO DARK SHADES
# -------------------------
colors = ['#FF6B6B', '#4ECDC4', '#FFE66D']  # Coral, Turquoise, Bright Yellow
# Alternative: ['#FF8C42', '#4C9A8A', '#FFD93D']  # Orange, Teal, Sun Yellow
# Alternative: ['#FF4D4D', '#6C5CE7', '#00B894']  # Bright Red, Purple, Mint

class_names = ['Bad (0-3)', 'Intermediate (4-7)', 'Good (8-10)']

# -------------------------
# Create professional visualization
# -------------------------
plt.figure(figsize=(14, 10))

# Plot each class with bright colors and NO dark edges
for class_id in [0, 1, 2]:
    mask = y == class_id
    plt.scatter(
        X_2d[mask, 0], 
        X_2d[mask, 1],
        c=colors[class_id],
        marker='o',
        s=250,  # Slightly larger
        alpha=0.9,  # More opaque
        edgecolors='white',  # White edges for contrast
        linewidth=1,
        label=f'{class_names[class_id]} ({np.sum(mask)} songs)',
        zorder=3
    )

# Add song indices with clean white numbers
for i, (x, y_coord) in enumerate(X_2d):
    plt.annotate(
        str(i+1),
        (x, y_coord),
        fontsize=8,
        fontweight='bold',
        ha='center',
        va='center',
        color='black',  # Black text for visibility
        bbox=None  # No background box
    )

# Light grid
plt.grid(True, alpha=0.15, linestyle='--', zorder=1, color='gray')

# Title and labels
plt.title(
    '🎤 Singing Skill Dataset - Song Distribution', 
    fontsize=20, 
    fontweight='bold',
    pad=20,
    color='#2C3E50'
)

plt.xlabel('Feature Space Dimension 1', fontsize=14, fontweight='bold', color='#2C3E50')
plt.ylabel('Feature Space Dimension 2', fontsize=14, fontweight='bold', color='#2C3E50')

# Clean legend with bright colors
legend = plt.legend(
    fontsize=13,
    title='Skill Levels',
    title_fontsize=14,
    loc='upper right',
    framealpha=1,
    edgecolor='white',
    facecolor='white'
)

# Simple insight box with white background
insight_text = (
    "🔍 Quick Insights:\n"
    "• 🟥 Bad (0-3): Top cluster\n"
    "• 🟨 Intermediate (4-7): Middle spread\n"
    "• 🟩 Good (8-10): Bottom concentration"
)

plt.text(
    0.02, 0.02, insight_text,
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='bottom',
    bbox=dict(boxstyle='round', facecolor='white', alpha=1, edgecolor='#DDDDDD')
)

# Clean background
plt.gca().set_facecolor('#FAFAFA')

# Remove ticks for cleaner look
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.savefig('singing_skill_dataset_map.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("✅ Visualization saved as 'singing_skill_dataset_map.png'")