#################
# Görev 1) List Comprehension yapısı kullanarak car_crashes verisindeki
# numeric değişkenlerin isimlerini
# büyük harfe çeviriniz ve başına NUM ekleyiniz.
#################

import seaborn as sns

df = sns.load_dataset("car_crashes")
df.columns

["NUM_" + col.upper() if df[col].dtype != 'O' else col.upper() for col in df.columns]

from pandas.api.types import is_numeric_dtype

[f"NUM_{col.upper()}" if is_numeric_dtype(df[col]) else col.upper() for col in df.columns]

#################
# Görev 2) List Comprehensions yapısı kullanarak car_crashers
# verisinde isminde "no" barındırmayan
# değişkenlerin sonuna FLAG ekleyiniz.
#################

import seaborn as sns

df = sns.load_dataset("car_crashes")

df.columns = [col.upper() + "_FLAG" if 'no' not in df[col] else col.upper() for col in df.columns]

# df.head(5)

df.columns = [col.upper() if 'no' in col else col.upper() + '_FLAG' for col in df.columns]

#################
# Görev 3) List Comprehension yapısı kullanarak aşağıda verilen değişken isimlerinden FARKLI olan
# değişkenlerin isimlerini seçiniz ve yeni bir data frame oluşturun.
#################

import seaborn as sns

df = sns.load_dataset("car_crashes")

# "abbrev", "no_previous" olmayanları

og_list = ["abbrev", "no_previous"]
new_cols = [col for col in df.columns if col not in og_list]

# 2. çözüm

new_cols = [col for col in df.columns if "abbrev" not in col and "no_previous" not in col]
new_df = df[new_cols]
new_df.head()

# 3. çözüm
og_list = ["abbrev", "no_previous"]

new_cols = [col for col in df.columns if col not in og_list]
print(new_cols)
new_df = df[new_cols]
