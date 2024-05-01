import matplotlib.pyplot as plt
import numpy as np

nouns = ["noodles", "napkin", "onion", "knife", "spoon", "pot"]

accuracies = {'CLIP Zero-shot': (51.618398637137986, 68.37416481069042, 48.888888888888886, 61.146496815286625, 
                                 44.86486486486487, 42.33576642335766),
              'CLIP Fine-tuned': (76.32027257240205, 87.52783964365256, 88.61111111111111, 88.95966029723992, 
                                  75.13513513513513, 83.57664233576642)}

x = np.arange(len(nouns))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in accuracies.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_title('Binary Classifcation Performance')
ax.set_xticks(x + width, nouns)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 100)

plt.show()
