#####################################################
# AB Testi ile BiddingYöntemlerinin Dönüşümünün Karşılaştırılması
#####################################################

#####################################################
# İş Problemi
#####################################################

# Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme türüne alternatif
# olarak yeni bir teklif türü olan "average bidding"’i tanıttı. Müşterilerimizden biri olan bombabomba.com,
# bu yeni özelliği test etmeye karar verdi ve average bidding'in maximum bidding'den daha fazla dönüşüm
# getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.A/B testi 1 aydır devam ediyor ve
# bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.Bombabomba.com için
# nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler için Purchase metriğine odaklanılmalıdır.




#####################################################
# Veri Seti Hikayesi
#####################################################

# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları
# reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır.Kontrol ve Test
# grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleriab_testing.xlsx excel’inin ayrı sayfalarında yer
# almaktadır. Kontrol grubuna Maximum Bidding, test grubuna Average Bidding uygulanmıştır.

# impression: Reklam görüntüleme sayısı
# Click: Görüntülenen reklama tıklama sayısı
# Purchase: Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning: Satın alınan ürünler sonrası elde edilen kazanç



#####################################################
# Proje Görevleri
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı (shapiro)
#   - 2. Varyans Homojenliği (levene)
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direkt 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.




#####################################################
# Görev 1:  Veriyi Hazırlama ve Analiz Etme
#####################################################

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)




# Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.

# Kontrol grubu:
df_control = pd.read_excel("/Users/melisacevik/PycharmProjects/AB-Testing/ab_testing.xlsx", sheet_name="Control Group")

# Test grubu:
df_test = pd.read_excel("/Users/melisacevik/PycharmProjects/AB-Testing/ab_testing.xlsx", sheet_name="Test Group")


# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.



df_control["Earning"].sum() # toplam kazanç

def check_df(dataframe, head=5):
    print("####shape####")
    print(dataframe.shape)
    print("####dtype####")
    print(dataframe.dtypes)
    print("####ilk 5 ####")
    print(dataframe.head(head))
    print("####son 5 ####")
    print(dataframe.tail(head))
    print("#### boş değer var mı ####")
    print(dataframe.isnull().sum())
    print("yüzdelik")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def check_sum_df(dataframe, col):
    print(f"{col} gözleminin toplamı")
    print(dataframe[col].sum())

# Kontrol grubu analizi: ( Maximum Bidding )
check_df(df_control)
check_sum_df(df_control, "Impression") # toplam görüntülenme
check_sum_df(df_control, "Click") # toplam tıklanma
check_sum_df(df_control, "Purchase") #toplam satın alma
check_sum_df(df_control, "Earning") #toplam kazanç


# Test grubu analizi: ( Average Bidding )
check_df(df_test)
check_sum_df(df_test, "Impression") # toplam görüntülenme
check_sum_df(df_test, "Click") # toplam tıklanma
check_sum_df(df_test, "Purchase") #toplam satın alma
check_sum_df(df_test, "Earning") #toplam kazanç


# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.

df = pd.concat([df_control, df_test], axis=1) # axis = 1 : her iki grup arasındaki karşılaştırmaları daha kolay yapabilmek için

#####################################################
# Görev 2:  A/B Testinin Hipotezinin Tanımlanması
#####################################################

# Adım 1: Hipotezi tanımlayınız.

# H0 = M1 = M2 ( Maximum Bidding ile Average Bidding arasında istatistiki olarak anlamlı bir fark yoktur! )
# H0 = M! != M2 ( Maximum Bidding ile Average Bidding arasında istatistiki olarak anlamlı bir fark vardır! )

# Adım 2: Kontrol ve test grubu için purchase(kazanç) ortalamalarını analiz ediniz

df_control["Purchase"].mean()
df_test["Purchase"].mean()

#####################################################
# GÖREV 3: Hipotez Testinin Gerçekleştirilmesi
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################


# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.Bunlar Normallik Varsayımı ve Varyans Homojenliğidir.

# Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz

# Normallik varsayımı

test_stat, pvalue = shapiro(df_control["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) # H0 Reddedilemez! Normallik Varsayımı sağlanıyor

test_stat, pvalue = shapiro(df_test["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) # H0 Reddedilemez! Normallik Varsayımı sağlanıyor

# Varyans Homojenliği Varsayımı

test_stat, pvalue = levene(df_control["Purchase"], df_test["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) # H0 Reddedilemez! Varyans Homojenliği Varsayımı sağlanıyor



# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz

# Kontrol grubu için uygun test => ttesttir.

test_stat, pvalue = ttest_ind(df_control["Purchase"], df_test["Purchase"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) # H0 Reddedilemez!

# Test grubu için uygun test => ttesttir.

test_stat, pvalue = ttest_ind(df_control["Purchase"], df_test["Purchase"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) # H0 Reddedilemez!

# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma
# ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.

# Kontrol ve test grubu satın alma ortalamaları arasında istatistiki olarak anlamlı bir fark yoktur!

##############################################################
# GÖREV 4 : Sonuçların Analizi
##############################################################

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.

# Kontrol ve Test Grubu için ayrı ayrı,
# Başta, Normallik varsayımını inceledim ve p_value değeri 0.05'ten yüksek çıktığı için Varyans Homojenliği varsayımını inceledim.
# İki varsayımda da p_value değeri 0.05'ten yüksek çıktı, 2 varsayımda da normal dağılım sağlanıyor. bu yüzden => ttest kullandım.


# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.

# Maximum Bidding ve Average Bidding arasında bir fark olmadığı gözlemlenmiştir. Yani, iki teklif verme yöntemi arasında satın alma
# ( Purchase ) açısından istatistiksel olarak fark yoktur.

# Bu sonuçlar ile Avg. Bidding'e geçişin gerekli olmadığını düşünebilirsiniz.