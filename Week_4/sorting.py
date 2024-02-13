###################################################
# Sorting Products
###################################################

###################################################
# Uygulama: Kurs Sıralama
###################################################
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler #standartlaştırma için

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("../../Measurement-Problems/datasets/product_sorting.csv")
print(df.shape)
df.head(10)

####################
# Sorting by Rating
####################

df.sort_values(by="rating", ascending=False).head(20)

# örneğin bir veri bilimi keyword'ü ile arama yapıldığı varsayımımız varsa 2. gözlem uygun olmayabilir.

# 1. gözlemlediğim durum => bağımsız sonuçların da gelmesi / göz ardı edeceğiz.

# 2. gözlemlediğim durum => 4. gözlemde rating yüksek olmasına rağmen comment / 5_point düşük
# satın alma sayısı ve yorum sayısı ratingin altında ezilmiş!

# satın alma sayıları ve yorum sayıları gözden kaçmış gibi gözüküyor.

####################
# Sorting by Comment Count or Purchase Count
####################

df.sort_values("purchase_count", ascending=False).head(20)
#2. gözlemde düşük commentliler geldi
df.sort_values("commment_count", ascending=False).head(20)

####################
# Sorting by Rating, Comment and Purchase
####################

# bu 3 faktörünü göz önünde bulundurup skor oluşturalım. ( hepsini aynı ölçeğe getirelim )
# rating 1-5 arasında sayılardan oluştuğu için diğer 2 değişkeni de 1-5 arasında skorlayalım.

# önce satın alma sayılarını(purchase) skorla
df["purchase_count_scaled"] = (MinMaxScaler(feature_range=(1,5)). \
    fit(df[["purchase_count"]]). \
    transform(df[["purchase_count"]]))
# (MinMaxScaler => aralık ver).( fit_transform() => değişiklik yapılan metod - beklemede).( transform() => dönüştürüldüğü metod)

# comment sayılarını skorla

df["comment_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["commment_count"]]). \
    transform(df[["commment_count"]])

df.head()

#sosyal ispatı güçlendirmeliyim!
(df["comment_count_scaled"] * 32 / 100 + #ücretli/ücretsiz fark etmez
 df["purchase_count_scaled"] * 26 / 100 + #ücretsiz eğitim olabilir
 df["rating"] * 42 / 100)

# BU HESAPLADIĞIMIZ ORANLAR SKOR!! Birçok faktörün ağırlığı ile oluşturulan skorlar

#fonksiyonu
def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return (dataframe["comment_count_scaled"] * w1 / 100 +
            dataframe["purchase_count_scaled"] * w2 / 100 +
            dataframe["rating"] * w3 / 100)

df["weighted_sorting_score"] = weighted_sorting_score(df) # bu skorlar için değişken oluşturup sıralayalım!

df.sort_values(by="weighted_sorting_score",ascending=False).head(20)

# sadece veri bilimi içerenleri alalım

df[df["course_name"].str.contains("Veri Bilimi")].sort_values("weighted_sorting_score", ascending=False).head(20)

# birden fazla faktörü göz önünde bulunduracaksam önce bu faktörleri aynı standarda getiriyoruz
# daha sonra ağırlıklandırabliriz ve sıralamayı yapıyoruz.

####################
# Bayesian Average Rating Score
####################

# Sorting Products with 5 Star Rated
# Sorting Products According to Distribution of 5 Star Rating

# puan dağılımlarının üzerinden ağırlıklı olasılıksal ortalama hesabı yapar.

# import math
# import scipy.stats as st import et!

# confidence => hesaplanacak olan z tablo değerine ilişkin bir değer elde edebilmek adına girilmiş bir değerdir.
# n  => girilecek olan yıldızların ve bu yıldızlara ait gözlenme frekansı
def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score


df.head()

df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                "2_point",
                                                                "3_point",
                                                                "4_point",
                                                                "5_point"]]), axis=1)
# x => değişkenleri temsil eder. apply aracılığıyla değişkenlerin üzerinde bir işlem gerçekleştireceğiz.
# axis =1 => sütunlarda bir işlem gerçekleştirmek
# df["bar_score"] = df.apply( lambda x: , axis=1) yaz önce , daha sonra x: 'den sonrasını yaz
# yazmış olduğum bayesian fonk. kullanıcam ve bu lambdanın görevi sütunlarda seçme işlemini gerçekleştirsin

#weighted fonksiyonu => 3 değişkeni göz önünde bulundurarak skor oluşturulup ağırlık verildi, tekrar çağıralım.
df.sort_values("weighted_sorting_score", ascending=False).head(20)
# şimdi yeni fonk. bakalım
df.sort_values("bar_score", ascending=False).head(20)

# bar_score bize sadece ratinglere odaklanarak bir sıralama sağladı. ama bu seferde social proof gözden kaçtı.
# tek odağımız verilen puanlar olsaydı ve buna göre sıralama yapacak olsaydık bar_score tek başına kullanılabilir.
# diğer faktörler göz önünde bulundurulacaksa bu geçerli değil.

# kurs isimlerinden indexe göre 5. ve 1. indexteki kursları getirme
df[df["course_name"].index.isin([5, 1])].sort_values("bar_score", ascending=False)
# burada 1. kurs neden 2. kurstan daha yukarda? çünkü puanlara göre sıralama yaptı ve düşük puan
# miktarları diğer kursa göre daha az olmasından dolayı yani;
# daha yüksek puanlara sahio olan dağılım açısından kurslar için hesapladığı skor daha yüksek
# bar_score => veri seti içinde yeni olsa da potansiyel vaad edenleri de yukarı taşır







####################
# Hybrid Sorting: BAR Score + Diğer Faktorler
####################

# Rating Products
# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating
# - Bayesian Average Rating Score

# Sorting Products
# - Sorting by Rating
# - Sorting by Comment Count or Purchase Count
# - Sorting by Rating, Comment and Purchase
# - Sorting by Bayesian Average Rating Score (Sorting Products with 5 Star Rated)
# - Hybrid Sorting: BAR Score + Diğer Faktorler


# wss => 3 faktörü ağırlandırarak bir araya getirdik.
# bu fonks. amacı hem bar score hem de 3 değişkeni göz önünde bulundurmak
# wss = %40 , bar = %60 ağırlık
def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                     "2_point",
                                                                     "3_point",
                                                                     "4_point",
                                                                     "5_point"]]), axis=1)
    wss_score = weighted_sorting_score(dataframe)

    return bar_score*bar_w/100 + wss_score*wss_w/100

df["hybrid_sorting_score"] = hybrid_sorting_score(df)

df.sort_values("hybrid_sorting_score", ascending=False).head(20)

# sadece veri bilimi içersin
df[df["course_name"].str.contains("Veri Bilimi")].sort_values("hybrid_sorting_score", ascending=False).head(20)






############################################
# Uygulama: IMDB Movie Scoring & Sorting
############################################

import pandas as pd
import math
import scipy.stats as st
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("/Users/melisacevik/PycharmProjects/Measurement-Problems/datasets/movies_metadata.csv",
                 low_memory=False)  # DtypeWarning kapamak icin

df = df[["title", "vote_average", "vote_count"]]

df.head()
df.shape

########################
# Vote Average'a Göre Sıralama
########################

df.sort_values("vote_average", ascending=False).head(20)

# az oylananları istemediğimiz için filtreleyelim
# çeyrek değerlerine bakalım

df["vote_count"].describe([0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]).T
# 1 tane oy alan filmde var %10'u
# medyan(%50) => 10
# oy ortalaması 109
# oylama sayısını ortalamanın biraz üstünde tutuyorum ve 400 üstü diyorum.  ( 45k film var - daha azına bakıcaz )

df[df["vote_count"] > 400].sort_values("vote_average" , ascending=False).head(20)

# 600 gibi değerler geldi, daha yukarı mı çekmeliydim?

# birlikte kullansak daha iyi gibi.
#  yapmak istediğim şey vote_count değişkenini standartlaştırmak

from sklearn.preprocessing import MinMaxScaler  #

df["vote_count_score"] = MinMaxScaler(feature_range=(1, 10)). \
    fit(df[["vote_count"]]). \
    transform(df[["vote_count"]])

# vote_count'u skorlaştırdım ama vote_count ile vote_average arasında bir etkileşim yok.


########################
# vote_average * vote_count
########################

df["average_count_score"] = df["vote_average"] * df["vote_count_score"]
df.sort_values("average_count_score", ascending=False).head(20)



########################
# IMDB Weighted Rating
########################


# weighted_rating = (v/(v+M) * r) + (M/(v+M) * C)

# r = vote average
# v = vote count
# M = minimum votes required to be listed in the Top 250
# C = the mean vote across the whole report (currently 7.0)

# burada bütün kitlenin ortalaması ve minimum gereken oy sayısı, her bir filmin kendi içindeki oy sayısı ve
# kendi puanına göre bir işleme tabii tutulup ağırlıklı ortalama hesaplanıyor.

# ilgili filmin oy sayısı ( v )  / fimin oy sayısı (v)  + gereken min oy sayısı(m) * ilgili filmin puanı ( R )
# +
#  gereken minimum oy sayısı ( m ) / filmin kendi oy sayısı ( v ) + ( gereken min oy sayısı ) * genel kitlenin ortalaması ( C )


# değerlendirme:
# sol + sağ
# sol
# aynı puanda, aynı gerekli oy sayısı old.halde eğer bir film daha yüksek sayıda oy aldıysa => +'nın sol tarafındaki formülde bu yakalandı
# sol formülün görevi, puanlara bakarak gereken min oy sayısı ile işlem yapmaktır, düzeltmek yapmak

#sağ
#alınan oy sayısı yüksekse dezavantaja dönüşebilir. ortalama değerde göz önünde bulundurulacak.


# Film 1:
# r = 8
# M = 500
# v = 1000

# (1000 / (1000+500))*8 = 5.33


# Film 2:
# r = 8
# M = 500
# v = 3000

# (3000 / (3000+500))*8 = 6.85

# (1000 / (1000+500))*9.5

# Film 1:
# r = 8
# M = 500
# v = 1000

# Birinci bölüm:
# (1000 / (1000+500))*8 = 5.33

# İkinci bölüm:
# 500/(1000+500) * 7 = 2.33

# Toplam = 5.33 + 2.33 = 7.66


# Film 2:
# r = 8
# M = 500
# v = 3000

# Birinci bölüm:
# (3000 / (3000+500))*8 = 6.85

# İkinci bölüm:
# 500/(3000+500) * 7 = 1

# Toplam = 7.85

# kitle için olş. parametreler: (genel)
M = 2500 #ortalama min oy sayısı
C = df["vote_average"].mean()

def weighted_rating(r, v, M, C):
    return (v / (v + M) * r) + (M / (v + M) * C)

df.sort_values("average_count_score", ascending=False).head(10)

# r = vote average
# v = vote count
# M = minimum votes required to be listed in the Top 250
# C = the mean vote across the whole report (currently 7.0)

#deapoolu inceleyelim.
weighted_rating(7.40000, 11444.00000, M, C)
# daha önceki skor 7.40 idi şimdi 7.08 geldi.

#inception
weighted_rating(8.10000, 14075.00000, M, C)

# daha önce 8.10 du 7.72 geldi

# The Shawshank Redemption

weighted_rating(8.50000, 8358.00000, M, C)
# daha önce 8.5 idi 7.83 geldi

df["weighted_rating"] = weighted_rating(df["vote_average"],
                                        df["vote_count"], M, C)

df.sort_values("weighted_rating", ascending=False).head(10)


####################
# Bayesian Average Rating Score
####################

# 12481                                    The Dark Knight
# 314                             The Shawshank Redemption
# 2843                                          Fight Club
# 15480                                          Inception
# 292                                         Pulp Fiction

def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score


# bu fonk.da 1,2,3,4,5,6,7,8,9,10 yıldız alan kaç değerlendirme var ([1yıldız, 2yıldız, ...])
#esaretin bedeli
bayesian_average_rating([34733, 4355, 4704, 6561, 13515, 26183, 87368, 273082, 600260, 1295351])
# godfather
bayesian_average_rating([37128, 5879, 6268, 8419, 16603, 30016, 78538, 199430, 402518, 837905])

# puan dağılımları bu dataset 1,2,3,4,..,10 yıldızın aldığı puanları içeriyor
df = pd.read_csv("/Users/melisacevik/PycharmProjects/Measurement-Problems/datasets/imdb_ratings.csv")
df = df.iloc[0:, 1:] #problemli bazı satırlardan kurtulduk


df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["one", "two", "three", "four", "five",
                                                                "six", "seven", "eight", "nine", "ten"]]), axis=1)
df.sort_values("bar_score", ascending=False).head(20)