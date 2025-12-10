import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



metrics = {
    "Number of Years": [5, 10, 15, 20, 25, 30],
    "Test_size": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
    "Accuracy": [0.9842, 0.9871, 0.9824, 0.9834, 0.9803, 0.9817],
    "Precision": [0.9778, 0.9868, 0.9795, 0.9783, 0.9740, 0.9744],
    "Recall": [0.9940, 0.9904, 0.9896, 0.9936, 0.9929, 0.9950],
    "F1-Score": [0.9858, 0.9886, 0.9845, 0.9859, 0.9833, 0.9845]
}

df = pd.DataFrame(metrics)


y_axis = "Accuracy"
x_axis = "Number of Years"

plt.figure(figsize=(8,5))
sns.set(style="whitegrid", font_scale=1.2)

sns.scatterplot(
    data=df,
    x = x_axis,
    y = y_axis,
    s=150,
    hue = "Test_size"
)


sns.regplot(
    data=df,
    x= x_axis,
    y= y_axis,
    scatter=False,
    color="red",
    line_kws={"linewidth": 2},
)

plt.title("Comparing Testing Splits on Accuracy vs Learning Set Size", fontsize=16, weight="bold")
plt.xlabel("Number of Years Learned From")
plt.ylabel("Accuracy")
plt.xlim(0, 35)     # because metrics are usually 0–1
plt.tight_layout()
plt.show()