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


