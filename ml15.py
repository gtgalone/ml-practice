from sklearn import model_selection, svm, metrics
import pandas

train_csv = pandas.read_csv("./mnist/train.csv", header=None)
tk_csv = pandas.read_csv("./mnist/t10k.csv", header=None)


def test(l):
  output = []
  for i in l:
    output.append(float(i) / 256)
  return output


train_csv_data = list(map(test, train_csv.iloc[:, 1:].values))
tk_csv_data = list(map(test, tk_csv.iloc[:, 1:].values))

train_csv_label = train_csv[0].values
tk_csv_label = tk_csv[0].values


clf = svm.SVC()
clf.fit(train_csv_data, train_csv_label)
predict = clf.predict(tk_csv_data)
score = metrics.accuracy_score(tk_csv_label, predict)
print("정답률:", score)



# input = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 84 185 159 151 60 36 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 222 254 254 254 254 241 198 198 198 198 198 198 198 198 170 52 0 0 0 0 0 0 0 0 0 0 0 0 67 114 72 114 163 227 254 225 254 254 254 250 229 254 254 140 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 17 66 14 67 67 67 59 21 236 254 106 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 83 253 209 18 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 22 233 255 83 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 129 254 238 44 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 59 249 254 62 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 133 254 187 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9 205 248 58 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 126 254 182 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 75 251 240 57 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 19 221 254 166 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 203 254 219 35 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 38 254 254 77 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 31 224 254 115 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 133 254 254 52 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 61 242 254 254 52 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 121 254 254 219 40 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 121 254 207 18 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
# input = input.split(" ")

# for i in range(len(input)):
#   print("{:3}".format(input[i]), end=" ")
#   if i % 28 == 0:
#     print()


# import urllib.request as request
# import gzip, os, os.path

# savepath = "./mnist"
# baseurl = "http://yann.lecun.com/exdb/mnist"
# files = [
#   "train-images-idx3-ubyte.gz",
#   "train-labels-idx1-ubyte.gz",
#   "t10k-images-idx3-ubyte.gz",
#   "t10k-labels-idx1-ubyte.gz"
# ]

# if not os.path.exists(savepath): os.mkdir(savepath)
# for f in files:
#   url = baseurl + "/" + f
#   loc = savepath + "/" + f
#   print("download:", url)
#   if not os.path.exists(loc):
#     request.urlretrieve(url, loc)

# for f in files:
#   gz_file = savepath + "/" + f
#   raw_file = savepath + "/" + f.replace(".gz", "")
#   print("gzip:", f)
#   with gzip.open(gz_file, "rb") as fp:
#     body = fp.read()
#     with open(raw_file, "wb") as w:
#       w.write(body)
      
# print("ok")



# import struct
# def to_csv(name, maxdata):
#     # 레이블 파일과 이미지 파일 열기
#     lbl_f = open("./mnist/"+name+"-labels-idx1-ubyte", "rb")
#     img_f = open("./mnist/"+name+"-images-idx3-ubyte", "rb")
#     csv_f = open("./mnist/"+name+".csv", "w", encoding="utf-8")
#     # 헤더 정보 읽기 --- (※1)
#     mag, lbl_count = struct.unpack(">II", lbl_f.read(8))
#     mag, img_count = struct.unpack(">II", img_f.read(8))
#     rows, cols = struct.unpack(">II", img_f.read(8))
#     pixels = rows * cols
#     # 이미지 데이터를 읽고 CSV로 저장하기 --- (※2)
#     res = []
#     for idx in range(lbl_count):
#         if idx > maxdata: break
#         label = struct.unpack("B", lbl_f.read(1))[0]
#         bdata = img_f.read(pixels)
#         sdata = list(map(lambda n: str(n), bdata))
#         csv_f.write(str(label)+",")
#         csv_f.write(",".join(sdata)+"\r\n")
#         # 잘 저장됐는지 이미지 파일로 저장해서 테스트하기 -- (※3)
#         if idx < 10:
#             s = "P2 28 28 255\n"
#             s += " ".join(sdata)
#             iname = "./mnist/{0}-{1}-{2}.pgm".format(name,idx,label)
#             with open(iname, "w", encoding="utf-8") as f:
#                 f.write(s)
#     csv_f.close()
#     lbl_f.close()
#     img_f.close()
# # 결과를 파일로 출력하기 --- (※4)
# to_csv("train", 1000)
# to_csv("t10k", 500)



# from sklearn import model_selection, svm, metrics
# # CSV 파일을 읽어 들이고 가공하기 --- (※1)
# def load_csv(fname):
#     labels = []
#     images = []
#     with open(fname, "r") as f:
#         for line in f:
#             cols = line.split(",")
#             if len(cols) < 2: continue
#             labels.append(int(cols.pop(0)))
#             vals = list(map(lambda n: int(n) / 256, cols))
#             images.append(vals)
#     return {"labels":labels, "images":images}
# data = load_csv("./mnist/train.csv")
# test = load_csv("./mnist/t10k.csv")
# # 학습하기 --- (※2)
# clf = svm.SVC()
# clf.fit(data["images"], data["labels"])
# # 예측하기 --- (※3)
# predict = clf.predict(test["images"])
# # 결과 확인하기 --- (※4)
# ac_score = metrics.accuracy_score(test["labels"], predict)
# cl_report = metrics.classification_report(test["labels"], predict)
# print("정답률 =", ac_score)
# print("리포트 =")
# print(cl_report)