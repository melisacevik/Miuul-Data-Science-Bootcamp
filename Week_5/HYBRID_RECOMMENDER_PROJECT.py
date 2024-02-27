
#############################################
# PROJE: Hybrid Recommender System
#############################################

# ID'si verilen kullanıcı için item-based ve user-based recomennder yöntemlerini kullanarak tahmin yapınız.
# 5 öneri user-based modelden 5 öneri de item-based modelden ele alınız ve nihai olarak 10 öneriyi 2 modelden yapınız.

#############################################
# Görev 1: Verinin Hazırlanması
#############################################
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

# Adım 1: Movie ve Rating veri setlerini okutunuz.
# movieId, film adı ve filmin tür bilgilerini içeren veri seti
# UserID, film adı, filme verilen oy ve zaman bilgisini içeren veri seti

movie = pd.read_csv("Case-Study/datasets/movie.csv")
rating = pd.read_csv("Case-Study/datasets/rating.csv")


# Adım 2: Rating veri setine filmlerin isimlerini ve türünü movie film setini kullanrak ekleyiniz.
# Ratingdeki kullanıcıların oy kullandıkları filmlerin sadece id'si var.
# Idlere ait film isimlerini ve türünü movie veri setinden ekliyoruz.

df = rating.merge(movie, how="left", on="movieId")

# Adım 3: Herbir film için toplam kaç kişinin oy kullandığını hesaplayınız.Toplam oy kullanılma sayısı 1000'un altında olan filmleri veri setinden çıkarınız.
# Herbir film için toplam kaç kişinin oy kullanıldığını hesaplıyoruz.

comment_counts = pd.DataFrame(df["title"].value_counts())


# Toplam oy kullanılma sayısı 10000'in altında olan filmlerin isimlerini rare_movies de tutuyoruz.
# Ve veri setinden çıkartıyoruz

rare_movies = comment_counts[comment_counts["count"] <= 10000].index #10000'in altındaki izlenmeler(yorumlar)

common_movies = df[~df["title"].isin(rare_movies)] # rare movies i liste dışı bırak

# Adım 4: # index'te userID'lerin sutunlarda film isimlerinin ve değer olarakta ratinglerin bulunduğu
# dataframe için pivot table oluşturunuz.

                #filtrelenen üzerinden pivot oluşturduk
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values=["rating"])

# Adım 5: Yukarıda yapılan tüm işlemleri fonksiyonlaştıralım

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv("Case-Study/datasets/movie.csv")
    rating = pd.read_csv("Case-Study/datasets/rating.csv")
    df = rating.merge(movie, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["count"] <= 10000].index  # 1000'in altındaki izlenmeler
    common_movies = df[~df["title"].isin(rare_movies)] # rare movies i liste dışı bırak
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values=["rating"])

    return user_movie_df

user_movie_df = create_user_movie_df() # 10000'den yukarıda rate i olan filtrelenmiş pivot tablomuz

#############################################
# Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################

# Adım 1: Rastgele bir kullanıcı id'si seçiniz.

random_user = int(pd.Series(user_movie_df.index).sample(1).values) # dataframe'in indexini seçmek için Series olarak seçmeliyim


# Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz.
# random_user = 109655 ilk kullanım
# random_user = 46691
#user_movie_df

random_user_df = user_movie_df[user_movie_df.index == random_user] # random kullanıcımı seçtim

# Adım 3: Seçilen kullanıcının oy kullandığı filmleri movies_watched adında bir listeye atayınız.

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist() # sütunlarda herhangi NA olmayan değerleri aldık,
# random kullanıcının İZLEDİĞİ filmleri movies_watched'a atadık.

#############################################
# Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################

# Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve movies_watched_df adında yeni bir dataframe oluşturuyoruz.

movies_watched_df = user_movie_df[movies_watched] # user_movie_df'in sütunlarını filtreledik

# Adım 2: Herbir kullancının seçili user'in izlediği filmlerin kaçını izlediği bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturunuz.
# Ve yeni bir df oluşturuyoruz.

# kullanıcıların random ile ortak izledikleri filmlerin toplamı, 1 ortak olursa almalı mıyız?
user_movie_count = movies_watched_df.T.notnull().sum() # notnull() => true false döner => bu dönüşümün trueları sum ediliyor (1)

# T => dataframein satırları(userId) sütuna geçecek


# Adım 3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenleri benzer kullanıcılar olarak görüyoruz.
# Bu kullanıcıların id’lerinden users_same_movies adında bir liste oluşturunuz.

user_movie_count = user_movie_count.reset_index() # indexte userId olmamalı
user_movie_count.columns = ["userId", "movie_count"]

users_same_movies = user_movie_count[user_movie_count["movie_count"] > len(movies_watched) * 60 / 100]["userId"].tolist()

# random ile benzer davranışları gösteren user'ları list halinde tuttuk

#############################################
# Görev 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
#############################################

# Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz.
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])

# concat => df birleştirmek için => korelasyonlarına bakmak için birleştirdik!
# 1. izlenen filmlerin indexlerinde benzer kullanıcıları ara
# 2. random user 'ı da satır olarak birleştir
# özetle random kullanıcının pivot df i ile bütün kullanıcıların df 'ini birleştirdik


# Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.

# hesaplama yapmak için user'ları sütuna alıcaz.

final_df = final_df.drop_duplicates() # tekrar eden
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

corr_df = pd.DataFrame(corr_df, columns=["corr"]) # okunmadığı için DF ' e çevirdik
corr_df.index.names = ['user_id_1', 'user_id_2'] #bunlar index isimleri değişkene çevirmek için;
corr_df = corr_df.reset_index()  # bunu yaptık

# bunlar tüm kullanıcılar açısından korelasyonlar

# Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları filtreleyerek top_users adında yeni bir dataframe oluşturunuz.

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True) # "user_id_2", "corr" 'u seçiyorum , user_id_1 'e ihtiyacım yok

top_users = top_users.sort_values(by="corr", ascending = False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True) #user_id_2'nin adını değiştid

# burada - leri almamam gerekiyor çünkü + korelasyon => randomla aynı artma azalma davranışı gösteriyor olması demektir

# Adım 4:  top_users dataframe’ine rating veri seti ile merge ediniz

rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

# top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user] #randomu buradan cıkardık

# ilk 5 gelmezse neler yapılabilir? []


#############################################
# Görev 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
#############################################

# Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturunuz.

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating'] # bu aşama çok önemli

# ratinge göre sıralarsak random da beğenebilir ama sadece rating i alırsak korelasyonu gözden kaçırırız ( korelasyonu aynı olduğunu varsaymış oluruz )
# sadece korelasyona göre de alamayız cünkü bazı ratingler düşük
# corr ve ratingin etkisini aynı anda göz önünde bulundurmalıyız.

# Adım 2: Film id’sive her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini içeren recommendation_df adında yeni bir
# dataframe oluşturunuz.

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"}).reset_index()
# movieId'ye birçok kullanıcı puan vermiş olabilir

#randomun filmlere vereceği değerler ( düşük olanlar da var )

# Adım 3: Adım3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating’e göre sıralayınız.
# İlk 5 gözlemi movies_to_be_recommend olarak kaydediniz.

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

# Adım 4:  Tavsiye edilen 5 filmin isimlerini getiriniz.
movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
movies_to_be_recommend.merge(movie[["movieId", "title"]]).head()

#############################################
# Adım 6: Item-Based Recommendation
#############################################

# Kullanıcının en son izlediği ve en yüksek puan verdiği filmin adına göre item-based öneri yapınız.
user = 108170

user_id_data_type = user_movie_df.index.dtype
print("userId sütununun veri tipi:", user_id_data_type)

# Adım 1: movie,rating veri setlerini okutunuz.
movie = pd.read_csv("Case-Study/datasets/movie.csv")
rating = pd.read_csv("Case-Study/datasets/rating.csv")

df = rating.merge(movie, how="left", on="movieId")

# Adım 2: Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.

user = 108170
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

               ### YARDIM AL ###

# Adım 3 :User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz.
movie[movie["movieId"] == movie_id]["title"].values[0]
movie_df = user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]
movie_df

user_movie_df[('rating', 'Wild at Heart (1990)')]

user_movie_df.reset_index().head().columns[:4]
user_movie_df.columns[:4]

user_movie_df['rating', '10 Things I Hate About You (1999)']
movie_df = user_movie_df[('rating', '12 Angry Men (1957)')]

# Adım 4: Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.
user_movie_df.corrwith(movie_df).sort_values(ascending=False).head(10)

# Son iki adımı uygulayan fonksiyon
def item_based_recommender(movie_name, user_movie_df):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)

# Adım 5: Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak veriniz.
movies_from_item_based = item_based_recommender("12 Angry Men (1957)", user_movie_df)
# 1'den 6'ya kadar. 0'da filmin kendisi var. Onu dışarda bıraktık.
movies_from_item_based[1:6].index


# 'My Science Project (1985)',
# 'Mediterraneo (1991)',
# 'Old Man and the Sea,
# The (1958)',
# 'National Lampoon's Senior Trip (1995)',
# 'Clockwatchers (1997)']

