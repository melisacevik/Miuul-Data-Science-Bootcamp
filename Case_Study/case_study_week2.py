# Let's Get Started ######


# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = sns.load_dataset("titanic")
df.head()

# Görev 2: Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.

df["sex"].value_counts()

# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.

unique_cols = [(col, df[col].nunique()) for col in df.columns]

# Görev 4: pclass değişkeninin unique değerlerinin sayısını bulunuz.

df["pclass"].nunique()

# Görev 5: pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.

df[["pclass", "parch"]].nunique()

# Görev 6: embarked değişkeninin tipini kontrol ediniz.
# Tipini category olarak değiştiriniz ve tekrar kontrol ediniz.

df["embarked"].info()
df["embarked"] = df["embarked"].astype("category")
df["embarked"].info()

# Görev 7: embarked değeri C olanların tüm bilgilerini gösteriniz

# Series'e geri çevirme => embarked_series = pd.Series(embarked_category)

c_values = df[df["embarked"] == "C"]

df.loc[df["embarked"] == "C"]  # bu da iyi

# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.

not_s_values = df[df["embarked"] != "S"]
df.loc[df["embarked"] != "S"]
# Görev 9: Yaşı 30'dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.

df.loc[(df["age"] < 30) & (df["sex"] == "female")]

# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.

df.loc[(df["fare"] > 500) | (df["age"] > 70)]

# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.

df.isnull().sum()

# Görev 12: who değişkenini dataframe’den çıkarınız.

df.drop("who", axis=1, inplace=True).head()  # inplace = True kalıcı

# Görev 13: deck değişkenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.

deck_mode_result = df['deck'].mode().iloc[0]
df['deck'].fillna(deck_mode_result, inplace=True)
# fillna fonksiyonunu kullanarak, NaN değerlere belirli bir değeri atayabilirsiniz.
df['deck'].isnull()

# alternatif

df_filled_deck = df["deck"].fillna(df["deck"].mode().iloc[0])

# Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurunuz.

df["age"].isnull().values.any()

age_median_result = df["age"].median()
df["age"].fillna(age_median_result, inplace=True)

df["age"].isnull().values.any()

# Görev 15: survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.

survived_pclass_sex = df.pivot_table("survived", "pclass", "sex", aggfunc=["sum", "count", "mean"])
survived_pclass_sex.reset_index()  # daha düzgün çıktı

# Görev 16: 30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazın.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz.
# (apply ve lambda yapılarını kullanınız)

df['age_flag'] = df.loc[:, df.columns[df.columns.str.contains("age")]].apply(
    lambda x: x.apply(lambda y: 1 if y < 30 else 0))

# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.

df = sns.load_dataset("Tips")
df.head()
df.info()

# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre
# total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.

result = df.groupby("time")["total_bill"].agg(["sum", "min", "max", "mean"])

# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.

day_and_time = df.groupby(["day", "time"])["total_bill"].agg(["sum", "min", "max", "mean"])
day_and_time.reset_index()  # daha düzgün çıktı

# Görev 20: Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin
# day'e göre toplamını, min, max ve ortalamasını bulunuz.

lunch_time_female_data = df[(df["time"] == "Lunch") & (df["sex"] == "Female")]

result = lunch_time_female_data.groupby("day")[["total_bill", "tip"]].agg(["sum", "min", "max", "mean"])
result.reset_index()

# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir? (loc kullanınız)

average_values = df.loc[(df["size"] < 3) & (df["total_bill"] > 10)].mean()

# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturunuz.
# Her bir müşterinin ödediği totalbill ve tip in toplamını versin.

df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]

# Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.

thirty_df = df.sort_values(by="total_bill_tip_sum", ascending=False).head(30)

#### end #######
