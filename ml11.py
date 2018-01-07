import csv, codecs

filename = "test.csv"
file = codecs.open(filename, "r", "utf8")

reader = csv.reader(file, delimiter=",", quotechar='"')
for cells in reader:
  print(cells[1], cells[2])

# splitted = csv.split("\n")
# for item in splitted:
#   list_mushroom = item.split(",")
#   list_mushroom[0]
#   list_mushroom[1]
#   list_mushroom[-1]
#   list_mushroom[-2]

# print(list_mushroom[1:4])