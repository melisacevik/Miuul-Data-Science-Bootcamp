############################################
# User-Based Collaborative Filtering
#############################################

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
# Adım 6: Çalışmanın Fonksiyonlaştırılması

#############################################
# Adım 1: Veri Setinin Hazırlanması
#############################################
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["count"] <= 10000].index #indexte tutuluyor
    common_movies = df[~df["title"].isin(rare_movies)] # title'larda 10.000 küçük yorum olmayanları seçmedik ve filtreledik
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating") # kesişim rating
    return user_movie_df

user_movie_df = create_user_movie_df()

import pandas as pd
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values) #random 1 kullanıcı seç, hocayla aynı olsun diye 45 aldık

#############################################
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################
random_user
user_movie_df #filtelediğimiz data frame'in pivot ( matris ) hali - en günceli -
random_user_df = user_movie_df[user_movie_df.index == random_user] # bu df'in içindeki random kullanıcımızı seçiyoruz - veriseti random kullanıcımıza göre indirgendi - satır randoma indirgendi ama sütunlarda bütün filmler duruyor
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist() # sütunlarda herhangi NA olmayan değerleri aldık yani random kullanıcının İZLEDİĞİ filmlere git

user_movie_df.loc[user_movie_df.index == random_user,
                  user_movie_df.columns == "Silence of the Lambs, The (1991)"]

# loc[ user_movie_df'in indexinde Random'u bul , user_movie_df'in sütunlarınd ise şu filmi bul]

len(movies_watched) #random'un izlediği toplam film sayısı



#############################################
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################

# Random kullanıcımızın izlediği filmleri izleyen kullanıcılara bakacağız.
# belirli bir sayının üstünde ortak filmleri izleyen kullanıcıları yakalamam lazım.

movies_watched_df = user_movie_df[movies_watched] #random kullanıcının izlediği film kadar indirge bütün filmleri

user_movie_count = movies_watched_df.T.notnull().sum() # notnull() => true false döner => bu dönüşümün trueları sum ediliyor (1)

# movies_watched_df.T => dataframein satırları sütuna geçecek

user_movie_count = user_movie_count.reset_index() # üzerinde işlem yapıcam indexte olmamalı ( 2 kolon oldu )

# her kullanıcı özelinde her bir kullanıcını toplam kaç tane ortak film izlediği bilgisi
user_movie_count.columns = ["userId", "movie_count"]

user_movie_count[user_movie_count["movie_count"] > 140].sort_values("movie_count", ascending=False) # Random kullanıcı ile en az 140 ortak film izleyen kullanıcılar

user_movie_count[user_movie_count["movie_count"] == 191].count() #random kullanıcının izlediği bütün filmleri izleyen kaç kullanıcı var


users_same_movies = user_movie_count[user_movie_count["movie_count"] > 140]["userId"] # random ile ortak 140 film izleyenlerin UserId'si


# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
# perc = len(movies_watched) * 60 / 100

#############################################
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
#############################################

# Bunun için 3 adım gerçekleştireceğiz:
# 1. Sinan ve diğer kullanıcıların verilerini bir araya getireceğiz.
# 2. Korelasyon df'ini oluşturacağız.
# 3. En benzer bullanıcıları (Top Users) bulacağız


final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])

# concat [izlenen filmlerin df'inde indexinde randomla aynı filmi izleyenleri ara ve randomun izlediği filmlerle birleştir ]

final_df = final_df.drop_duplicates() # tekrarlanan satırları kaldır
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

corr_df = pd.DataFrame(corr_df, columns=["corr"])

corr_df.index.names = ['user_id_1', 'user_id_2']

corr_df = corr_df.reset_index() # bunlar tüm kullanıcılar açısından korelasyonlar

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.35)][
    ["user_id_2", "corr"]].reset_index(drop=True)

#normalde >= 0.65 yazdık ama benim random kullanıcıda veri gelmediği için 0.35e çektim

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True) #user_id_2'nin adını değiştid

# random ile en yüksek korelasyona sahip olan kullanıcılar elimizde ama bu kullanıcıların hangi filme kaç puan
# verdiğini bilmiyorum. top users ile veri setini birleştirelim.
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user] #randomu buradan cıkardık

# Şu anda elimizde Random ile en yüksek korelasyona sahip olan kullanıcılar ve bunların çeşitli filmlere
# verdiği puanlar var
# her birinde sinanla en az 140 film izleyen kullanıcıların vermiş olan puanlar ve bu kullanıcıların korelasyonları hesaplandı.

#############################################
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
#############################################

# korelasyon ve ratingin etkisini aynı anda göz önünde bulundurmalıyız.

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

# ratingleri korelasyona göre düzeltmiş olduk
# bir kişinin korelasyonu düşükse Rating ona göre güncellendi, daha düşük gözlemlenir

top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"}) # movieId'ye göre groupby yapıp w.r 'in ortalamasını aldık

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = recommendation_df.reset_index() # movieId indexte tutulmamalı sütunda görelim

recommendation_df[recommendation_df["weighted_rating"] > 3.5]

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 1.905].sort_values("weighted_rating", ascending=False)

# bu random da çok fazla örnek olmadığı için 3.5 dan büyük diyemedik

# bu filmlerin hangi filmler olduğunu bulalım
movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
movies_to_be_recommend.merge(movie[["movieId", "title"]])



#############################################
# Adım 6: Çalışmanın Fonksiyonlaştırılması
#############################################

# 1000den az yorum yapanları filtrelediğimiz ve pivot tablo oluşturduğumuz fonks.
def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

# Random kullanıcı ile yüzde kaç benzerliğe göre seçim yapma ihtiyacımızı ifade eden bölüm

# perc = len(movies_watched) * 60 / 100
# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

# ratio => random kullanıcı ile yüzde kaç ortak filmleri izlemiş olsun
def user_based_recommender(random_user, user_movie_df, ratio=60, cor_th=0.65, score=3.5):
    import pandas as pd
    random_user_df = user_movie_df[user_movie_df.index == random_user]
    movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
    movies_watched_df = user_movie_df[movies_watched]
    user_movie_count = movies_watched_df.T.notnull().sum()
    user_movie_count = user_movie_count.reset_index()
    user_movie_count.columns = ["userId", "movie_count"]
    perc = len(movies_watched) * ratio / 100 # random 100 film izlediyse, kullanıcılar en az 60 izlemeli gibi
    users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

    final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                          random_user_df[movies_watched]])

    corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
    corr_df = pd.DataFrame(corr_df, columns=["corr"])
    corr_df.index.names = ['user_id_1', 'user_id_2']
    corr_df = corr_df.reset_index()

    # cor_th => random ile ortak filmleri izlemiş olabilir ama benzer davranışı bilemem o yüzden cor threshold değerini dışardan 0.65 belirledim.
    top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= cor_th)][
        ["user_id_2", "corr"]].reset_index(drop=True)

    top_users = top_users.sort_values(by='corr', ascending=False)
    top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
    top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating'] # korelasyon ve ratingi  aynı anda göz önünde bulunduruyorum.

    recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
    recommendation_df = recommendation_df.reset_index()

    movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > score].sort_values("weighted_rating", ascending=False)
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    return movies_to_be_recommend.merge(movie[["movieId", "title"]])



random_user = int(pd.Series(user_movie_df.index).sample(1).values) #randomu belirle
user_based_recommender(random_user, user_movie_df, cor_th=0.70, score=4) # fonk. çağır


