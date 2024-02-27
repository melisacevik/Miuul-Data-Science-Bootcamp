###########################################
# Item-Based Collaborative Filtering
###########################################

# Veri seti: https://grouplens.org/datasets/movielens/

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: User Movie Df'inin Oluşturulması
# Adım 3: Item-Based Film Önerilerinin Yapılması
# Adım 4: Çalışma Scriptinin Hazırlanması

######################################
# Adım 1: Veri Setinin Hazırlanması
######################################
import pandas as pd
pd.set_option('display.max_columns', 500)
movie = pd.read_csv('/Users/melisacevik/Desktop/recommender_systems/datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('/Users/melisacevik/Desktop/recommender_systems/datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId") #movie sol , rating sağ olacak şekilde birleştirildi
df.head()

# amaç bir filmi seçtiğimizde o filme benzer filmleri bulmak

######################################
# Adım 2: User Movie Df'inin Oluşturulması
######################################

df.head()
df.shape

#eşsiz film sayısı
df["title"].nunique()

# her bir filme kaç tane yorum yapılmış, userId, rating, timestamp bilgileri yorum sayısını verir. biz böylece filmlere kaç rating yapılmış onu da bulmuş oluruz.
df["title"].value_counts().head()

# value_count'u bir dataframe'e çeviriyoruz
comment_counts = pd.DataFrame(df["title"].value_counts()) # her bir film için kaç tane yorum yapılmış

rare_movies = comment_counts[comment_counts["count"] <= 15000].index  # az yorum yapılan filmleri bulduk

# az yorum yapılan filmleri df'imizden çıkarıyoruz

common_movies = df[~df["title"].isin(rare_movies)] # rare_moviesin içindekilere bak, BUNLARIN DIŞINDAKİLERİ AL

common_movies["title"].nunique()

# satırlara kullanıcı sütunlara film isimleri gelecek şekilde pivot table oluşturuyoruz
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating") #satır => user, sütun => film , kesişimde => rating

user_movie_df.shape

######################################
# Adım 3: Item-Based Film Önerilerinin Yapılması
######################################

movie_name = "Matrix, The (1999)"
movie_name = "Ocean's Twelve (2004)"
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10) #korelasyonu en yüksek olan ilk 10'u getir( en yüksek benzerlikteki filmler )


movie_name = pd.Series(user_movie_df.columns).sample(1).values[0] #rastgele bir film seç, 0.indexteki değerini al
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)


def check_film(keyword, user_movie_df):
    return [col for col in user_movie_df.columns if keyword in col] #herhangi bir keyword girildiğinde o keyword içeren filmleri getirir

check_film("Toy", user_movie_df)


######################################
# Adım 4: Çalışma Scriptinin Hazırlanması
######################################

#kütüphaneler ve df oluşturma
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

#filme benzer filmleri bulma ( korelasyonu en yüksek olan ilk 10 filmi getirme )
def item_based_recommender(movie_name, user_movie_df):
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

item_based_recommender("Matrix, The (1999)", user_movie_df) #Matrix, The (1999) filmine benzer filmleri getir

movie_name = pd.Series(user_movie_df.columns).sample(1).values[0] #rastgele bir film seç, 0.indexteki değerini al

item_based_recommender(movie_name, user_movie_df) #tekrar çalıştırıldığında farklı sonuçlar alınır(random seçildiği için)





