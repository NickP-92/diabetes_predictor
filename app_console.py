import matplotlib.pyplot as plt
from ml_models import load_and_preprocess, train_models

print("Loading and preprocessing data...")
X_train, X_test, y_train, y_test = load_and_preprocess()

print("Training models...")
results = train_models(X_train, X_test, y_train, y_test)

print("\nF1 Scores:")
for model, metrics in results.items():
    print(f"{model}: {metrics['f1_score']:.2f}")

# Plot F1 Score Comparison
print("\nGenerating F1 score chart...")
fig, ax = plt.subplots()
ax.bar(results.keys(), [m["f1_score"] for m in results.values()], color=['skyblue', 'salmon'])
ax.set_title("F1 Score Comparison")
ax.set_ylim(0, 1)
ax.set_ylabel("F1 Score")
ax.grid(axis='y')
plt.tight_layout()
plt.savefig("f1_score_comparison.png")
plt.show()
