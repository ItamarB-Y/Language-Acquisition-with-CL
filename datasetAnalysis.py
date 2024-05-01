import json
import matplotlib.pyplot as plt


frames_path =  "/nfs/turbo/coe-chaijy/itamarby/ALA/processed_frames/"
js_path = "frames.json"

frames_dict = {}

with open(js_path) as js_f:
    frames_dict = json.load(js_f)



frequency_per_noun = {"noodles": 0, "onion": 0, "pot": 0, "knife": 0, "napkin": 0, "spoon": 0}
frequency_per_comb = {}

for frame, nouns in frames_dict.items():
    string_nouns = " \n ".join(nouns)
    if string_nouns in frequency_per_comb:
        frequency_per_comb[string_nouns] += 1
    else:
        frequency_per_comb[string_nouns] = 1
    for noun in nouns:
        frequency_per_noun[noun] += 1

frequency_per_noun = dict(sorted(frequency_per_noun.items(), key=lambda item: item[1], reverse=True))
frequency_per_comb = dict(sorted(frequency_per_comb.items(), key=lambda item: item[1], reverse=True))

print(frequency_per_noun)
print('\n\n\n')
print(frequency_per_comb)

for comb in frequency_per_comb:
    frequency_per_comb[comb] = 100*(frequency_per_comb[comb] / 3387)


total = 0
frequency_per_comb2 = {}
for comb in frequency_per_comb:
    if frequency_per_comb[comb] < 3:
        total += frequency_per_comb[comb]
    else:
        frequency_per_comb2[comb] = frequency_per_comb[comb]

frequency_per_comb2["Other"] = total


plt.title("Frequqency of nouns in the Dataset - 3387 Frames in Total")
plt.xticks(rotation=0)
plt.bar(frequency_per_noun.keys(), frequency_per_noun.values())
plt.show()

plt.title("Frequqency of combinations in the Dataset - 3387 Frames in Total")
plt.xticks(rotation=0)
plt.pie(frequency_per_comb2.values(), labels=frequency_per_comb2.keys(), autopct='%1.1f%%')
plt.show()

