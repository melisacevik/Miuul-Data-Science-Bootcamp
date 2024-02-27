#############################
# Model-Based Collaborative Filtering: Matrix Factorization
#############################

# !pip install surprise
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
pd.set_option('display.max_columns', None)

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Modelleme
# Adım 3: Model Tuning
# Adım 4: Final Model ve Tahmin

#############################
# Adım 1: Veri Setinin Hazırlanması
#############################

movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()

movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]

sample_df = df[df.movieId.isin(movie_ids)] # bu filmlere göre bütün df 'i indirge
sample_df.head()

sample_df.shape

user_movie_df = sample_df.pivot_table(index=["userId"],
                                      columns=["title"],
                                      values="rating")

user_movie_df.shape

# buradaki satırlar kullanıcıları ifade ediyor. sütunlar => filmi

reader = Reader(rating_scale=(1, 5)) #ratelerin skalasını vermeliyiz.

data = Dataset.load_from_df(sample_df[['userId',
                                       'movieId',
                                       'rating']], reader) # df 'in yapısını SUPRISE kütüphanesinin kendi istediği veri formatına verimizi getirmiş olduk.

# okuyamıyoruz, çünkü özel bir veri yapısı.

##############################
# Adım 2: Modelleme
##############################

# Modelleme neyi ifade eder? model oluşturmayı ifade eder ve sınıflandırma,regresyon ve kümeleme gibi birçok görevi içerir.

#Modelleri belirli bir eğitim seti üzerinde kurup daha sonra modelin görmediği başka bir test seti üzerinde test ederiz.
# train seti ve test seti oluşturucaz, train seti üzerinden model kuracağız. modelimizi test seti üzerinde test edeceğiz.
# görmediği veri setinde doğru çalışıyor mu ( test seti )

trainset, testset = train_test_split(data, test_size=.25) # %75 ' i train , %25 'i test olacak şekilde bölüyoruz.
svd_model = SVD() # Matrix factorization yöntemini kullanacak olduğumuz fonksiyon
svd_model.fit(trainset) # trainset üzerinden öğren # p ve q ağırlıklarını burada bulduk
predictions = svd_model.test(testset) #bu p ve q ağırlıklarını kullanarak test setinin değerlerini tahmin ediyor

# predictions = elimizdeki tüm olası user id ve item idlerin yani user ve movie lerin gerçek ve tahmin edilen değerlerini verdi

#Prediction(uid=89813.0, iid=356, r_ui=5.0, est=4.183944112877576, details={'was_impossible': False}),
#89813 idli kullanıcının, 356 filmine, 5.0 verdiği gerçek puan , 4.18.. tahmin edilen


accuracy.rmse(predictions) #ortalama ne kadar hata
# rmse => hata kareler ortalaması karekökü fonksiyonu => ∑(gerçek-tahmin )^2 => karekökü

# 0.94 geldi. yani çağrı bir filme 4 verecekse ben 4.92 veya 3.06 veririm => ortalama hatam

svd_model.predict(uid=1.0, iid=541, verbose=True) # bir kullanıcı için bir tahmin edelim

svd_model.predict(uid=1.0, iid=356, verbose=True)


sample_df[sample_df["userId"] == 1] # 1 için uid değerini yazdık

##############################
# Adım 3: Model Tuning
##############################

# Modelin hiperparametrelerini ayarlama ve performansını iyileştirme işlemidir.

# Model Optimize etmek nedir ? Modelin tahmin performansını arttırmaya çalışmak demektir.
# Dışsal parametreler => iterasyon sayısı ( kullanıcı tarafından verilmesi gerekir ya da opt.edilmesi gerekir.)
#

param_grid = {'n_epochs': [5, 10, 20],
              'lr_all': [0.002, 0.005, 0.007]}


gs = GridSearchCV(SVD,
                  param_grid,
                  measures=['rmse', 'mae'],
                  cv=3,
                  n_jobs=-1,
                  joblib_verbose=True)

# model nesnen SVD ,
# griddeki bütün parametre çiftlerini tek tek dene,
# gerçek ( rmse ) değerlerle tahmin edilen ( mae ) değerlerinin farklarının kare ortalamalarını al => measures[]
# cv => 3 farklı çapraz doğrulama yap
# veri setini 3 e böl 2 parçasıyla model kur 1 iyle test et ( bunu 3 ü için de yapcak 3 kez )
#n_jobs => işlemcileri full performansıyla kullan demek
# joblib_Verbose => işlemler gerçekleştirken bana raporlama yap

gs.fit(data) # GridSearchCV 'yi modelle

gs.best_score['rmse']
gs.best_params['rmse']


##############################
# Adım 4: Final Model ve Tahmin
##############################

# Son modelin seçilmesi ve gerçek verilerle tahmin yapılmasıdır.

dir(svd_model) #bunun içinden ne alabiliriz
svd_model.n_epochs

svd_model = SVD(**gs.best_params['rmse']) #modeli yeni değerleriyle oluşturacak

data = data.build_full_trainset()
svd_model.fit(data)

svd_model.predict(uid=1.0, iid=541, verbose=True)






