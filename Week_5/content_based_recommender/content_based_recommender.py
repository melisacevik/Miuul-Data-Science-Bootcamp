#############################
# Content Based Recommendation (İçerik Temelli Tavsiye)
#############################

#############################
# Film Overview'larına Göre Tavsiye Geliştirme
#############################

# 1. TF-IDF Matrisinin Oluşturulması
# 2. Cosine Similarity Matrisinin Oluşturulması
# 3. Benzerliklere Göre Önerilerin Yapılması
# 4. Çalışma Scriptinin Hazırlanması

#################################
# 1. TF-IDF Matrisinin Oluşturulması
#################################

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# https://www.kaggle.com/rounakbanik/the-movies-dataset
df = pd.read_csv("/Users/melisacevik/Desktop/recommender_systems/datasets/the_movies_dataset/movies_metadata.csv", low_memory=False)  # DtypeWarning kapamak icin
df.head()
df.shape

df["overview"].head()


# kullanacak olduğumuz metin ing. olduğu için on,in,the,an gibi ölçüm değeri taşımayan ve yaygın kullanılan bu ifadeleri veri setimizden çıkarıyoruz.
# ne kadar kelime varsa o kadar sütun olacak o yüzden gereksizleri çıkardık
tfidf = TfidfVectorizer(stop_words='english') # şu an silmiş olmadık , nesne oluşturmuş olduk ( beklemede )

df[df["overview"].isnull()] #NA değerler geldi.

df["overview"] = df["overview"].fillna("") #NA yerine boşluk ( NA hata çıkarabilir işlemlerde)

tfidf_matrix = tfidf.fit_transform(df["overview"])
#fit_transform() => fit etme işlemi ilgili veri yapısı üzreinde işlemi yapar,
# transform => fit edilip elde edilen değerleri kalıcı olarak değiştirir.

tfidf_matrix.shape

# 45466 => doküman, yorum, açıklamadır
# 75827 => eşsiz kelimeler var

# bunların kesişimdinde TF-IDF skorları vardır.

feature_names = tfidf.get_feature_names_out()
for name in feature_names:
    print(name)
# kelimelerin isimleri
    
tfidf_matrix = tfidf_matrix.toarray() # buradaki skorlar dokümanlar ile terimlerin kesişimlerindeki skorlar


#################################
# 2. Cosine Similarity Matrisinin Oluşturulması
#################################

# benzerliğini hesaplamak istediğin matrisi ver

cosine_sim = cosine_similarity(tfidf_matrix,
                               tfidf_matrix)

cosine_sim.shape

# 45466 film var ve her biri için 45466 film ile benzerlik skorları var.
cosine_sim[1]


#################################
# 3. Benzerliklere Göre Önerilerin Yapılması
#################################

# isimleri olmadığı için seri oluşturuyoruz
indices = pd.Series(df.index, index=df['title'])

indices.index.value_counts()

# title'larda çoklama var! bunu düzeltmek için birisini tutup diğerlerini atacağız. sonuncuyu tutuyoruz.

indices = indices[~indices.index.duplicated(keep='last')] # DUPLICATE EDİLENLERİN DIŞINDAKİLERİ TUT

indices.index.value_counts() # ÇOKLAMA PROBLEMİ ÇÖZÜLDÜ

indices["Cinderella"]

indices["Sherlock Holmes"]

movie_index = indices["Sherlock Holmes"]

# cosine sim'deki Sherlock Holmes indexine filmine ait benzerlik skorları
cosine_sim[movie_index]

# okuyabilmek için dataframe'e çevirme
similarity_scores = pd.DataFrame(cosine_sim[movie_index], 
                                 columns=["score"]) #Sherlock Holmes ile diğer filmler arasındaki benzerlik skorları


movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index # kendisi hariç ilk 10 filmi getirme ( 0. index kendisi olacağı için 1 den başladık)

df['title'].iloc[movie_indices]

#################################
# 4. Çalışma Scriptinin Hazırlanması
#################################

def content_based_recommender(title, cosine_sim, dataframe):
    # index'leri olusturma
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # title'ın index'ini yakalama
    movie_index = indices[title]
    # title'a gore benzerlik skorlarını hesapalama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # kendisi haric ilk 10 filmi getirme
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]

content_based_recommender("Sherlock Holmes", cosine_sim, df)

content_based_recommender("The Matrix", cosine_sim, df)

content_based_recommender("The Godfather", cosine_sim, df)

content_based_recommender('The Dark Knight Rises', cosine_sim, df)


def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


cosine_sim = calculate_cosine_sim(df)
content_based_recommender('The Dark Knight Rises', cosine_sim, df)
# 1 [90, 12, 23, 45, 67]
# 2 [90, 12, 23, 45, 67]
# 3
