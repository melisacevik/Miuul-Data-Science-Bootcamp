###################################################
# Rating Products
###################################################

# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating


############################################
# Uygulama: Kullanıcı ve Zaman Ağırlıklı Kurs Puanı Hesaplama
############################################

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# (50+ Saat) Python A-Z™: Veri Bilimi ve Machine Learning
# Puan: 4.8 (4.764925)
# Toplam Puan: 4611
# Puan Yüzdeleri: 75, 20, 4, 1, <1
# Yaklaşık Sayısal Karşılıkları: 3458, 922, 184, 46, 6

df = pd.read_csv("../../Measurement-Problems/datasets/course_reviews.csv")
df.head()
df.shape

#odağımız satırdaki puanlar ve yapılacak olan hesaplama işlemi

# Rating Dağılımı
df["Rating"].value_counts()
df["Questions Asked"].value_counts()

# 2 soru soran kişinin verdiği puanı öğrenmek için,

df.groupby("Questions Asked").agg({"Questions Asked": "count", #value_count bi daha alındı
                                   "Rating": "mean"})

df.head()

####################
# Average
####################

# Ortalama Puan
df["Rating"].mean()

# Ne yaparsak güncel trendi ortalamaya daha iyi bir şekilde yansıtabiliriz?
####################
# Time-Based Weighted Average
####################
# Puan Zamanlarına Göre Ağırlıklı Ortalama ( Time- Based Weighted Average )

df.info()
# zamana göre işlem yapacağız ama object türünde olduğu için zaman değişkenine çevircez

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# yapılan yorumları gün cinsinden ifade et
# 4 => 4 gün önce yorum yapılmış

#eski veri seti olduğu için tarih belirleyelim.
current_date = pd.to_datetime('2021-02-10 0:0:0') #stringi tarihe çevir

df["days"] = (current_date - df["Timestamp"]).dt.days

# bu veri setinde son 30 günde yapılan yorumlar

df[df["days"] <= 30].count()
# bu veri setinde son 30 günde yapılan yorumların ortalaması

# 30dan küçük olanları day'e göre getir ve bunun içinden Ratingi getir
df.loc[df["days"] <= 30 ,"Rating" ].mean()

#30'dan büyük ( en az 31 ) 90'dan küçük

df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean()

# 90'dan büyük 180'den küçük
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean()

# 180'den büyük

df.loc[(df["days"] > 180), "Rating"].mean()

#zamanı göre ağırlıklı ortalamayı hesaplayalım . son 1 ay içindeyse %28, 1-3 ay arası ise %26 , 3-6 ay arası ise %24, 6'dan fazla ise %22 ağırlık olsun.
# güncel yoruma ağırlıklı oran vericem

df.loc[df["days"] <= 30 ,"Rating" ].mean() * 28/100 + \
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26/100 + \
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24/100 + \
df.loc[(df["days"] > 180), "Rating"].mean() * 22 /100

# fonksiyon

def time_based_weighted_average(dataframe, w1 = 28, w2=26, w3 =24, w4=22):
    return  df.loc[df["days"] <= 30 ,"Rating" ].mean() * w1/100 + \
           df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * w2/100 + \
           df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * w3/100 + \
           df.loc[(df["days"] > 180), "Rating"].mean() * w4/100

time_based_weighted_average(df)
time_based_weighted_average(df,30,26,22,22)

# herkesin verdiği puanın ağırlığı aynı mı olmalı?

####################
# User-Based Weighted Average
####################

# kursun %1 ini izleyenler %100'ünü izleyen kişinin ağırlığı aynı mı olmalı?

df.head()
# kursu izleme oranlarına göre daha farklı bir ağırlık mı olmalı ?

df.groupby("Progress").agg({"Rating":"mean"})

# gözlem üzerine %1 ilerleme kaydeden katılımcılar 4.6 verirken %100 tamamlayanlar 4.86 vermiş!
# az izlediyse %22

df.loc[df["Progress"] <= 10, "Rating"].mean() * 22 / 100 + \
    df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 24 / 100 + \
    df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 26 / 100 + \
    df.loc[(df["Progress"] > 75), "Rating"].mean() * 28 / 100

def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return dataframe.loc[dataframe["Progress"] <= 10, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 45) & (dataframe["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 75), "Rating"].mean() * w4 / 100


user_based_weighted_average(df, 20, 24, 26, 30)

# kalite metriği belirledik, zaman bazlı, kullanıcı bazlı.
# zamana göre ve izlenmeye göre ağırlıklı ortalama hesabı yaptık. bunların bir ortalamasını alsak?

####################
# Weighted Rating
####################

def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return time_based_weighted_average(dataframe) * time_w/100 + user_based_weighted_average(dataframe)*user_w/100

course_weighted_rating(df)

course_weighted_rating(df, time_w=40, user_w=60)