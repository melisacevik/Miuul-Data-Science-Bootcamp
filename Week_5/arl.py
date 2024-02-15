############################################
# ASSOCIATION RULE LEARNING (BİRLİKTELİK KURALI ÖĞRENİMİ)
############################################

# 1. Veri Ön İşleme
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
# 3. Birliktelik Kurallarının Çıkarılması
# 4. Çalışmanın Scriptini Hazırlama
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak

############################################
# 1. Veri Ön İşleme
############################################

# !pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

df_ = pd.read_excel("/Users/melisacevik/PycharmProjects/Miuul-Data-Science-Bootcamp/datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.head() #fatura id'leri çoklanmış

# pip install openpyxl
# df_ = pd.read_excel("datasets/online_retail_II.xlsx",
#                     sheet_name="Year 2010-2011", engine="openpyxl")

# bu projenin zorluğu veri setini özel veri yapısına dönüştürmek

df.describe().T #price, quantityde negatif değerler var, max değeri çok yüksek ( aykırı değerler var demek )
df.isnull().sum() # veri setinde eksik değerler var mı?
df.shape

# 2 fonksiyon yazacağız.
# 1.si iade faturaları çıkar, NA'leri çıkar, Quantity ve Price ı 0'dan büyük yap
# 2.si Quantity ve Price değişkenleri için 0'dan büyük olanları al, bu değişkenler için aykırı değerleri bul , eşik değerlere baskıla

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True) #null'lardan kurtul
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)] # invoice'da C barındırmayanları getir dediğimiz için tilda #iadeden kurtul
    dataframe = dataframe[dataframe["Quantity"] > 0] #0'dan küçüklerden kurtul
    dataframe = dataframe[dataframe["Price"] > 0]
    return dataframe


# na = True olduğunda => içerisinde herhangi Nan değerlerinin olduğu satırları da dahil eder. C ve NaN olanları birlikte filtreler
df = retail_data_prep(df)

# sırada aykırı değerleri temizle
# fonksiyon eşik değerler belirleme => outlier fonk

# AYKIRI DEĞERLERİ TEMİZLEME
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01) #çeyrek değer
    quartile3 = dataframe[variable].quantile(0.99) #%75'lik değer
    interquantile_range = quartile3 - quartile1 # IQR = Q3 - Q1
    up_limit = quartile3 + 1.5 * interquantile_range # 99'luk çeyrek değerden 1.5 IQR uzaktaki nokta benim üst limitim
    low_limit = quartile1 - 1.5 * interquantile_range # 1'lik çeyrek değerden - 1.5 IQR uzaktaki nokta benim alt limitim
    return low_limit, up_limit

#baskılama yapma
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit # low limitten aşağıda olanları low_limitle değiştir yani baskıla
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit # up limitten yüksek olanları


# iki fonksiyonla işlem yapalım

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)
df.isnull().sum()
df.describe().T

############################################
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
############################################

df.head()
df["Descr"]

# satırda invoice sütunda product olsun istiyoruz.
# sütunda belirli bir üründen kaç tane olduğu bilgisi olacak.


# Description   NINE DRAWER OFFICE TIDY   SET 2 TEA TOWELS I LOVE LONDON    SPACEBOY BABY GIFT SET
# Invoice
# 536370                              0                                 1                       0
# 536852                              1                                 0                       1
# 536974                              0                                 0                       0
# 537065                              1                                 0                       0
# 537463                              0                                 0                       1

# invoice'lar sepet(transaction) olacak
# örn. index = 0 'da nine drawer yok => 0

# veri setini belirli bir ülkeye indirgeyerek ilerle

df_fr = df[df["Country"] == "France"]

#Fransa müşterilerinin birliktelik kurallarını türetmiş olacağım.

# Daha önce Almanya'dakilere satış yapmamışsam Almanyayla benzer alışkanlıklar sergilemesini beklediğim bir ülkeyi belirlersem Fransa'yı kullabilirim.

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20) # bir faturada aynı üründen kaç tane var bilgisi

# descriptionlara sütuna geçirmek için pivot() işlemi yapacağız.
# unstack() => buradaki isimlendirmeleri değişken isimlendirmelerine çeviriyorum.
# iloc => index based seçimi yap => satırlardan ve sütunlardan 5'er adet getir.

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]

# boş olan yerlerde 0 , dolu olan yerlerde 1 yazsın.

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]

# 0'dan büyük değerler görürse 1 yazacak fonksiyon
df_fr.groupby(['Invoice', 'StockCode']). \
    agg({"Quantity": "sum"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]

# apply = satır ya da sütun bilgisi verilir, bir fonksiyonu bu satır ya da sütunları döngüsüz otomatik olarak uygular
# applymap () = bütün gözlemleri gezer.

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

fr_inv_pro_df = create_invoice_product_df(df_fr)

fr_inv_pro_df = create_invoice_product_df(df_fr, id=True)

# buradaki id'lerin hangi ürün olduğunu öğrenmek için
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

check_id(df_fr, 10120)

############################################
# 3. Birliktelik Kurallarının Çıkarılması
############################################

# apriori () => olası tüm birlikteliklerin Support değerlerini ( olasılıklarını ) bulur!
# bu fonksiyona 1 ) df, 2) eşik değerini ver (min. Support değeri) , 3) değişkenlerin ismini kullanmak istiyorsan use_colnames = True ver

frequent_itemsets = apriori(fr_inv_pro_df, min_support=0.01, use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False)

# bu yukarıdaki her bir ürünün olasılığıdır.

# bizim ihtiyacımız olan birliktelik kurallarıdır.

#association_rules() => bu hesaplamaları yapmak için metrik ve threshold ver

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)

# antecedents =  X ürün
# consequents =  Y ürün
# antecedent support =  X ürünün tek başına gözlenme olasılığı = Support değeri
# consequents support = Y ürünün tek başına gözlenme olasılığı = Support değeri
#support   = X ve Ynin birlikte gözlenme olasılığı (antecedents                                        consequents )
# confidence = X ürünü alındığında Y'nin alınması olasılığı
# lift = X ürünü satın alındığında Y ürününün satın alınma olasılığı 17 kat artar #genele göre 17 kat daha fazla olasılık YANSIZ METRİK
# leverage => Supportu yüksek olan değerlere öncelik eğilimindedir. YANLI METRİK
# conviction => y olmadan X ürününün beklenen değeridir- x ürünü olmadan y ürününün beklenen frekansı

rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]
# birlikte görülme ols 0.05'ten büyük, x alındığında diğerinin alınma olasılığı 0.1 'den büyük, lift en az 5 kat artsın.

check_id(df_fr, 21080) # herhangi bir ürünün ismini görüntüleme

# biz kullanıcıya bir ürünü aldığında bi ürün önermek istersek,
# confidence göre sırala.

rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]. \
sort_values("confidence", ascending=False)

# bir kişi sepetine bunu eklediğinde ( antecedents ) => consequents'deki ürünü önericez !

############################################
# 4. Çalışmanın Scriptini Hazırlama
############################################

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


# son yaptığımız işlemleri bu fonk. yerleştirdik. create_invoice_product_df fonksiyonunda 2 parametre olduğu için buraya id parametresini ekledik
def create_rules(dataframe, id=True, country="France"):
    dataframe = dataframe[dataframe['Country'] == country] # bu ülkeye göre veriyi indirge
    dataframe = create_invoice_product_df(dataframe, id) # stok kodlarına(True ise) ve invoice a göre matris oluşturma
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

df = df_.copy()

df = retail_data_prep(df)
rules = create_rules(df)

rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]. \
sort_values("confidence", ascending=False)

############################################
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak
############################################

# Örnek:
# Kullanıcı örnek ürün id: 22492

# bir ürün sepete eklendiyse hangi ürünleri önermesi gerektiğinin bilgisi SQL tablolarında tutuluyor olur
# biz burada bir öneriyi üretme işlemini yapacağız.

product_id = 22492 # kullnıcı sepetine bir ürün ekledi
check_id(df, product_id)

sorted_rules = rules.sort_values("lift", ascending=False) #örneğin lift olsun , confidence de olabilirdi

# ürüne karşılık gelen indexteki önereceğim ürünü yakalamak için fonksiyon yazalım.
recommendation_list = [] # birden fazla ürün olma olasılığı olduğu için list oluştur.
# ürün ikilemesi ile ilgilenmiyorum!!! 2. ürünle ilgileniyorum!
# kolaylık açısından, bu ürünü nerede görürsem önereceğim ürünü yakalayacağım.

# sorted_rules = belirli filtelerden geçmiş ve lifte göre sıralanmış yeni df

for i, product in enumerate(sorted_rules["antecedents"]):# product satırları tek tek gezecek ( antecedents'deki ) gezerken index bilgilerini istiyorum.  indexi gezecek => i
    for j in list(product): # yakaladığın değerleri listeye çevir ve o listenin de içinde gezinmem lazım => j ile
        if j == product_id: # product_id 'yi yakalarsan
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0]) #listeye ekle


# enumerate sayesinde => bir koşul yakaladığımda , o noktadaki index bilgisiyle önereceğim ürünü seçeceğim !
# ( 21080, 21094 ) => bir set yapısıdır ! orada bir işlem yapabilmem için listeye çevirmem lazım

recommendation_list[0] # ilk öneriyi iste
recommendation_list[0:2] # 2 ürün öner
recommendation_list[0:3] #3 ürün öner

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


arl_recommender(rules, 22492, 1)
arl_recommender(rules, 22492, 2)
arl_recommender(rules, 22492, 3)