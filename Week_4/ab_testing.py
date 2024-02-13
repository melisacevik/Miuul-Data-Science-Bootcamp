######################################################
# Temel İstatistik Kavramları
######################################################

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


############################
# Sampling (Örnekleme)
############################

# popülasyon ve örneklem arasındaki ilişki nasıl yansıyor?

populasyon = np.random.randint(0,80,10000) #0-80 arası 10000 sayı üreteceğiz. ör => yaş

populasyon.mean()

# burada hedeflediğimiz popülasyon; bir ilçedeki kişilerin bilgisi.
# 10000 kişi var, bu kişilere ulaşıp, yaş bilgisini öğrenip bu ilçedeki insanların yaş ort. erişmek istiyorum.
# tek tek sormak yerine örnek teorisi ; 10000 kişiyi temsil eden iyi bir alt küme seçtiğimizde bu kişilere tek tek sormadan
#GENELLEME şansı verir!

np.random.seed(115)
orneklem = np.random.choice(a=populasyon,size=100)

orneklem.mean()

# örneklem => daha az veriyle genellemeler yapabilmek! bize zaman, para,işgücüde kolaylık sağlar


np.random.seed(10)
orneklem1 = np.random.choice(a=populasyon, size=100)
orneklem2 = np.random.choice(a=populasyon, size=100)
orneklem3 = np.random.choice(a=populasyon, size=100)
orneklem4 = np.random.choice(a=populasyon, size=100)
orneklem5 = np.random.choice(a=populasyon, size=100)
orneklem6 = np.random.choice(a=populasyon, size=100)
orneklem7 = np.random.choice(a=populasyon, size=100)
orneklem8 = np.random.choice(a=populasyon, size=100)
orneklem9 = np.random.choice(a=populasyon, size=100)
orneklem10 = np.random.choice(a=populasyon, size=100)

(orneklem1.mean() + orneklem2.mean() + orneklem3.mean() + orneklem4.mean() + orneklem5.mean()
 + orneklem6.mean() + orneklem7.mean() + orneklem8.mean() + orneklem9.mean() + orneklem10.mean()) / 10


# ÖRNEKLEM SAYISI ARTTIĞINDA BU ÖRNEKLEM DAĞILIMINA İLİŞKİN ORTALAMA DA POPÜLASYONA YAKINSIYOR OLACAKTIR.
# Veri arttığında bir yakınsama söz konusudur.


############################
# Descriptive Statistics (Betimsel İstatistikler)
############################

df = sns.load_dataset("tips")
df.describe().T

# describe metodu => sayısal değişkenleri seçerek onları betimler.
# değişken içinda kaç gözlem olduğu (count ) , değişken ortalaması (mean), ss , min , max , çeyrek değerleriyle ilgili bilgi verir.

# medyan => non parametrik  | mean => parametrik ortalama
# küçükten büyüğe sıralanır ve tam ortadaki değer => %50 => medyan
# örneğin bir değişkende aykırı değer var mı, dağılımı nasıl, çarpık mı? diye bakacağımızda mean ve %50 arasına bakarım. ( yakın olmalı )

# 12 aylık satış verilerim var;
# ilk 9 ay 50.000 satışım var son 3 ay 150 bin satışım var. Ortalamayı mı medyanı mı tercih ederiz?
# ortalama yanıtlıcı olur.
# elimde aykırı, çarpık değerler varsa ortalamayı değil medyanı seçmeliyim

############################
# Confidence Intervals (Güven Aralıkları)
############################

# güven aralığı olmasa ; 180'nden 40 çıkarıp 40 eklersem ; 140-220 saniye arası zaman geçiriyorlar derdim
# güven aralığı bize ortalamadan bir şeyler ekleyip çıkarma işlemini bilimsel bir formla yapar
# bir kitleden alınabilecek olan örneklemin ortalaması var. Başka örneklemler de alınabilir.
# başka örneklemler de alındığı için, olası alabileceğin 100 örneklemden 95'inin ortalaması bu aralıkta olacaktır.
# ne sağlar? => olası bütün senaryoları %5 hata payı ile almak karar noktası sağlar


df = sns.load_dataset("tips")
df.describe().T

# elimde bir ortalama var ve daha kapsayıcı bir yorum gerekiyor!
# kötü senaryoda ne kazanırım( maaş belirleme, ödeme zamanlarını planlamak istiyor olabilirim. )

# güven aralığı işlemi
sms.DescrStatsW(df["total_bill"]).tconfint_mean()
# sms : statmodels içerisindeki bir API, bu sms içindeki DescrStatW'nin içinde tconfint_mean() metodu var. bu metodla güven aralığı
# hesabını gerçekleştiricem.
# 100 defa örneklem çeksem, 100'ünün ortalamasını alsan, bunların %95'inde ortalama bu aralıkta çıkar
# Restorantımın müşterilerinin ödediği hesap ortalamaları istatistiki olarak %95 güven ile 18.66 ve 20.90 değerleri arasındadır.
# en kötü senaryom => 18.66 en iyi senaryom => 20.90

# bahşiş

sms.DescrStatsW(df["tip"]).tconfint_mean()
# bana gelen müşterilerimin bırakacak oldukları bahşişler ortalama olarak bu 2 değer arasında yer alıyor.

# Titanic Veri Setindeki Sayısal Değişkenler için Güven Aralığı Hesabı
df = sns.load_dataset("titanic")
df.describe().T
# yaş değişkeninin içindeki eksik değerleri siliyoruz(hata vermemesi için)
sms.DescrStatsW(df["age"].dropna()).tconfint_mean()

sms.DescrStatsW(df["fare"].dropna()).tconfint_mean()

######################################################
# Correlation (Korelasyon)
######################################################


# Bahşiş veri seti:
# total_bill: yemeğin toplam fiyatı (bahşiş ve vergi dahil)
# tip: bahşiş
# sex: ücreti ödeyen kişinin cinsiyeti (0=male, 1=female)
# smoker: grupta sigara içen var mı? (0=No, 1=Yes)
# day: gün (3=Thur, 4=Fri, 5=Sat, 6=Sun)
# time: ne zaman? (0=Day, 1=Night)
# size: grupta kaç kişi var?

df = sns.load_dataset('tips')
df.head()

#bahşişler ile ödenen hesap arasında bir korelasyon var mı?

#yemeğin toplam fiyatının içinde bahşiş de dahil old. için çıkarıyoruz

df["total_bill"] = df["total_bill"] - df["tip"]

# scatter plot ( saçılım grafiği ) oluşturalım.

df.plot.scatter("tip", "total_bill")
plt.show()

#grafiğin matematiksel gösterimi - corr() tip le total bill arasındaki korelasyonu verir

# corr() metodu iki değişken arasındaki korelasyonu gözlemlemek istediğimizde kullanabileceğimiz bir metottur.
df["tip"].corr(df["total_bill"]) # düşük pozitif korelasyonun biraz üzerinde

# toplam hesap ile ödenen bahşiş arasında pozitif yönlü orta şiddetli bir ilişki vardır.
# Dolayısıyla ödenen hesap miktarı arttıkça bahşiş de artıyor.


######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı
#   - 2. Varyans Homojenliği
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direk 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.


############################
# Uygulama 1: Sigara İçenler ile İçmeyenlerin Hesap Ortalamaları Arasında İst Ol An Fark var mı?
############################


df = sns.load_dataset("tips")
df.head()

# sigara içip içmeme durumuna göre toplam hesabın ort al
df.groupby("smoker").agg({"total_bill": "mean"})

#fark var ama şans eseri ortaya çıkmış olabilir

############################
# 1. Hipotezi Kur
############################

# H0: M1 = M2
# H1: M1 != M2

############################
# 2. Varsayım Kontrolü
############################

# Normallik Varsayımı
# Varyans Homojenliği

############################
# Normallik Varsayımı
############################

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.

# Normallik varsayımını test edeceğim. => shapiro testi ile
# shapiro testi bir değişkenin dağılımının normal olup olmadığını test eder.

                                # 1. gruba ilgili grup, ilgili değişkeni ver
test_stat, pvalue = shapiro(df.loc[df["smoker"] == "Yes", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

# burada p'den küçük çıktı => H0'ı reddettim. => normallik varsayımı sağlanmamaktadır.


test_stat, pvalue = shapiro(df.loc[df["smoker"] == "No", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# H0 red. p < 0.05

#bu varsayım için normal dağılım varsayımı sağlanmamaktadır. Şimdi diğer varsayıma bakalım

############################
# Varyans Homojenligi Varsayımı
############################

# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

# varyans homojenliği varsayımını incelemek için levene() testi kullanırız.
                            # iki farklı grubu gönder

test_stat, pvalue = levene(df.loc[df["smoker"] == "Yes", "total_bill"],
                           df.loc[df["smoker"] == "No", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.


# H0 red, 2 varsayımda sağlanmıyor => non parametrik!!

############################
# 3 ve 4. Hipotezin Uygulanması
############################

# 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
# 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)


# bizim örneğimizde varsayım sağlanmıyor ama bunu da görelim
############################
# 1.1 Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
############################

#ttest metodu => 2sinden biri ve 2 varsayım da sağlanıyorsa da kullanılabilir
# varyans homojenliği varsayımı sağlanmıyorsa equal_var = False gir

test_stat, pvalue = ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],
                              df.loc[df["smoker"] == "No", "total_bill"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < 0.05 'ten ise HO RED.
# p-value < 0.05 değilse H0 REDDEDILEMEZ.

# 0.18 cıktı. H0 reddedilemez.

############################
# 1.2 Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
############################

# mannwhitneyu testi : nonparametrik ortalama kıyaslama, medyan kıyaslama testidir.


test_stat, pvalue = mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                                 df.loc[df["smoker"] == "No", "total_bill"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# H0 reddedilemedi ! İkisinin ortalamaları arasında istatistiki olarak anlamlı bir fark yoktur!
# ortalamaları arasında fark yoktur!
# h0 'ı ya reddederiz ya reddedemeyiz. h1 'i kabul etmek gibi bir yorum yok












############################
# Uygulama 2: Titanic Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anl. Fark. var mıdır?
############################

df = sns.load_dataset("titanic")
df.head()

df.groupby("sex").agg({"age": "mean"})

# 1. Hipotezleri kur:
# H0: M1  = M2 (Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anl. Fark. Yoktur)
# H1: M1! = M2 (... vardır)


# 2. Varsayımları İncele

# Normallik varsayımı ( shapiro )
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır

test_stat, pvalue = shapiro(df.loc[df["sex"] == "female", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) #kadınlar için H0 RED

test_stat, pvalue = shapiro(df.loc[df["sex"] == "male", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) # erkekler için de H0 RED

# iki grup için de varsayım sağlanmamaktadır. => non parametriğe git. ama yine de parametriğe bakalım. ( görmek için )

# Varyans homojenliği ( levene )
# H0: Varyanslar Homojendir ( varyanslar benzer )
# H1: Varyanslar Homojen Değildir

# test_stat, pvalue = levene(df.loc[df["sex"] == "female", "age"].dropna(),
#                           df.loc[df["sex"] == "male", "age"].dropna())

#print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#bu kısım örnek olması içindi

# varsayımlar sağlanmadığı için nonparametrik

test_stat, pvalue = mannwhitneyu(df.loc[df["sex"] == "female", "age"].dropna(),
                                 df.loc[df["sex"] == "male", "age"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# H0 RED => yaş ortalaması arasında gözlemlediğimiz fark istatistiki olarak da vardır.










############################
# Uygulama 3: Diyabet Hastası Olan ve Olmayanların Yaşları Ort. Arasında İst. Ol. Anl. Fark var mıdır?
############################

df = pd.read_csv("/Users/melisacevik/PycharmProjects/Measurement-Problems/datasets/diabetes.csv")
df.head()

df.groupby("Outcome").agg({"Age": "mean"})

# 1. Hipotezleri kur
# H0: M1 = M2 (Diyabet Hastası Olan ve Olmayanların Yaşları Ort. Arasında İst. Ol. Anl. Fark Yoktur )
# H1: M1 != M2 (.... vardır. )

# 2. Varsayımları İncele

# Normallik Varsayımı (H0: Normal dağılım varsayımı sağlanmaktadır.)
test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) # H0 red , normallik varsayımı sağlanmamaktadır

test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) # H0 red, normallik varsayımı sağlanmamaktadır


# Normallik varsayımı sağlanmadığı için nonparametrik.

# Hipotez (H0: M1 = M2)
test_stat, pvalue = mannwhitneyu(df.loc[df["Outcome"] == 1, "Age"].dropna(),
                                 df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))  # H0 red,

# diyabet olanların ve olmayanların yaş ort. arasında istatistiki olarak anlamlı bir farklılık vardır.







###################################################
# İş Problemi: Kursun Büyük Çoğunluğunu İzleyenler ile İzlemeyenlerin Puanları Birbirinden Farklı mı?
###################################################

# H0: M1 = M2 (... iki grup ortalamaları arasında ist ol.anl.fark yoktur.)
# H1: M1 != M2 (...vardır)

df = pd.read_csv("/Users/melisacevik/PycharmProjects/Measurement-Problems/datasets/course_reviews.csv")
df.head()

# ilerlemesi %75'den yüksek olanların Ratinglerinin ortalaması
df[(df["Progress"] > 75)]["Rating"].mean()

df[(df["Progress"] < 25)]["Rating"].mean()

# normallik varsayımı
test_stat, pvalue = shapiro(df[(df["Progress"] > 75)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# 1. grup için sağlanamıyor.


test_stat, pvalue = shapiro(df[(df["Progress"] < 25)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#2. grup için de sağlanamıyor.

# Normallik testinin H0'ı normallik varsayımı sağlanıyor idi. RED => nonparametriğe git


test_stat, pvalue = mannwhitneyu(df[(df["Progress"] > 75)]["Rating"],
                                 df[(df["Progress"] < 25)]["Rating"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) # H0 RED

# kursu daha fazla izleyenlerin daha az izleyenlere göre verdikleri puan ort. daha yüksek


######################################################
# AB Testing (İki Örneklem Oran Testi)
######################################################

# H0: p1 = p2
# Yeni Tasarımın Dönüşüm Oranı ile Eski Tasarımın Dönüşüm Oranı Arasında İst. Ol. Anlamlı Farklılık Yoktur.
# H1: p1 != p2
# ... vardır

# başarı sayıları ve gözlem sayıları ayrı ayrı arraylere koy.

basari_sayisi = np.array([300, 250])
gozlem_sayilari = np.array([1000, 1100])

proportions_ztest(count=basari_sayisi, nobs=gozlem_sayilari)

# p value değeri 0.05'ten küçük H0 red. => fark var

basari_sayisi / gozlem_sayilari

# 0.3 daha başarılı

############################
# Uygulama: Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Farklılık var mıdır?
############################

# H0: p1 = p2    ( bu p1 - p2 = 0 olarak da karşımıza çıkabilir! )
# Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Fark yoktur

# H1: p1 != p2
# .. vardır

df = sns.load_dataset("titanic")
df.head()

# kadınların hayatta kalma oranı

df.loc[df["sex"] == "female", "survived"].mean()

# erkeklerin hayatta kalma oranı

df.loc[df["sex"] == "male", "survived"].mean()

#proportions_ztest'in beklentisini yerine getir

female_succ_count = df.loc[df["sex"] == "female", "survived"].sum()
male_succ_count = df.loc[df["sex"] == "male", "survived"].sum()

# 1. argümanda her bir sınıfın başarı oranı metriği gerekiyor. count=[female_succ_count, male_succ_count]
# 2. argümanda gözlem sayısını istiyor ( örnek sayısı )

test_stat, pvalue = proportions_ztest(count=[female_succ_count, male_succ_count],
                                      nobs=[df.loc[df["sex"] == "female", "survived"].shape[0],
                                            df.loc[df["sex"] == "male", "survived"].shape[0]])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))




######################################################
# ANOVA (Analysis of Variance)
######################################################

# İkiden fazla grup ortalamasını karşılaştırmak için kullanılır.

df = sns.load_dataset("tips")
df.head()

df.groupby("day")["total_bill"].mean()

# 1. Hipotezleri kur

# HO: m1 = m2 = m3 = m4 ( 4 ortalama var )
# Grup ortalamaları arasında fark yoktur.

# H1: .. fark vardır

# 2. Varsayım kontrolü

# Normallik varsayımı
# Varyans homojenliği varsayımı

# Varsayım sağlanıyorsa one way anova
# Varsayım sağlanmıyorsa kruskal

# H0: Normal dağılım varsayımı sağlanmaktadır.
# veri setinde day değişkenini burdaki günlere göre filtrelemek için;
# day değişkenindeki unique değerlerini liste yap ve içinde gez list(df["day"].unique())

# bir kategorik değişkeninin sınıflarını iteratif bir nesneye çevirmek için listeye çevirdik


for group in list(df["day"].unique()):
    pvalue = shapiro(df.loc[df["day"] == group, "total_bill"])[1]
    print(group, 'p-value: %.4f' % pvalue)


# h0 red => non parametrik

#örnek açısından varyans homojenliği varsayımına bakalım

# H0: Varyans homojenliği varsayımı sağlanmaktadır.

test_stat, pvalue = levene(df.loc[df["day"] == "Sun", "total_bill"],
                           df.loc[df["day"] == "Sat", "total_bill"],
                           df.loc[df["day"] == "Thur", "total_bill"],
                           df.loc[df["day"] == "Fri", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# 3. Hipotez testi ve p-value yorumu

# Hiç biri sağlamıyor.
df.groupby("day").agg({"total_bill": ["mean", "median"]})


# ortalamalar arasında fark var mı yok mu sorusunu genelden sormak ile /
# ikiden fazla grup testiyle bu gruplar arasında fark var mı yok mu sorusunu test etmek
# farklı şeyler

# varsayım sağlanıyorsa => tek yönlü parametrik testi
# varsayım sağlanmıyorsa => kruskal()


# HO: Grup ortalamaları arasında ist ol anl fark yoktur. varsayım sağlanmıyor. non parametrik bir test kullanmamız lazım



#varsayımın sağlandığını varsayalım
# parametrik anova testi:

f_oneway(df.loc[df["day"] == "Thur", "total_bill"],
         df.loc[df["day"] == "Fri", "total_bill"],
         df.loc[df["day"] == "Sat", "total_bill"],
         df.loc[df["day"] == "Sun", "total_bill"])

# h0 red, fark var

kruskal(df.loc[df["day"] == "Thur", "total_bill"],
        df.loc[df["day"] == "Fri", "total_bill"],
        df.loc[df["day"] == "Sat", "total_bill"],
        df.loc[df["day"] == "Sun", "total_bill"])

# h0 reddedildi , bu grupların arasında istatistiki olarak anlamlı bir fark var


# Diğerlerinden farklı olarak bir problemimiz daha var.
# Farklılık hangisinden kaynaklanıyor ?

# statsmodels içerisindeki çoklu karşılaştırma metodunu kullanacağız. TUKEY testini kullanacağız.

from statsmodels.stats.multicomp import MultiComparison
comparison = MultiComparison(df['total_bill'], df['day'])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())