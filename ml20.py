from gensim.models import word2vec

data = word2vec.LineSentence("toji.wakati")
model = word2vec.Word2Vec(data, size=200, window=10, hs=1, min_count=2, sg=1)

model.save("toji.model")