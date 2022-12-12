import matplotlib.pyplot as plt

x = ["Binary Cross Entropy", "Focal Loss"]
h = [79.33, 88.22]

ax = plt.subplot()
p1 = ax.bar(x, h)
ax.bar_label(p1, h)
ax.set_title("Accuracy Comparison")
ax.set_ylabel("Accuracy (%)")
plt.savefig("accuracy_comparison.jpg")
plt.show()
