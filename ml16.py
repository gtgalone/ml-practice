from sklearn import svm, metrics
import glob, os.path, re, json
import matplotlib.pyplot as plt
import pandas as pd

files = glob.glob("./lang/train/*.txt")
train_data = []
train_label = []
for file_name in files:
  # get label
  basename = os.path.basename(file_name)
  lang = basename.split("-")[0]
  # get text
  file = open(file_name, "r", encoding="utf-8")
  text = file.read()
  text = text.lower()
  file.close()
  # get alphabet rate
  code_a = ord("a")
  code_z = ord("z")
  count = [0 for n in range(0, 26)]
  for character in text:
    code_current = ord(character)
    if code_a <= code_current <= code_z:
      count[code_current - code_a] += 1
  # normalize
  total = sum(count)
  count = list(map(lambda n: n / total, count))

  train_label.append(lang)
  train_data.append(count)

# get graph
graph_dict = {}
for i in range(0, len(train_label)):
  label = train_label[i]
  data = train_data[i]
  if not (label in graph_dict):
    graph_dict[label] = data

asclist = [[chr(n) for n in range(97, 97 + 26)]]
print(asclist)
df = pd.DataFrame(graph_dict, index=asclist)

plt.style.use('ggplot')
df.plot(kind="bar", subplots=True, ylim=(0, 0.15))
plt.savefig("lang-plot.png")

# files = glob.glob("./lang/test/*.txt")
# test_data = []
# test_label = []
# for file_name in files:
#   # get label
#   basename = os.path.basename(file_name)
#   lang = basename.split("-")[0]
#   # get text
#   file = open(file_name, "r", encoding="utf-8")
#   text = file.read()
#   text = text.lower()
#   file.close()
#   # get alphabet rate
#   code_a = ord("a")
#   code_z = ord("z")
#   count = [0 for n in range(0, 26)]
#   for character in text:
#     code_current = ord(character)
#     if code_a <= code_current <= code_z:
#       count[code_current - code_a] += 1

#   # normalize
#   total = sum(count)
#   count = list(map(lambda n: n / total, count))
  
#   test_label.append(lang)
#   test_data.append(count)

# # learning

# clf = svm.SVC()
# clf.fit(train_data, train_label)
# predict = clf.predict(test_data)
# score = metrics.accuracy_score(test_label, predict)
# print("score=", score)
# report = metrics.classification_report(test_label, predict)
# print("--report--")
# print(report)
