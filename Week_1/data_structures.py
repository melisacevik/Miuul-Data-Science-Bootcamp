####################
# VERİ YAPILARI ( DATA STRUCTURES)
#####################
# Giriş
# Sayılar(Numbers) : int, float , complex
# Karakter Dizileri ( Strings ) = str
# Boolean ( TRUE-FALSE ) = bool
# Liste (List)
# Sözlük (Dictionary)
# Demet (Tuple)
# Set


###################
# Sayılar : int
###################

x = 46
type(46)

###################
# Sayılar : float
###################

y = 3.14
type(3.14)

###################
# Sayılar : complex
###################

x = 2j + 1
type(x)

###################
# Karakter : str
###################

x = "Hello ai era"
type(x)

###################
# Boolean : true/ false
###################

True
False
type(True)
5 == 4

###################
# Liste
###################

x = [ "btc" , "eth" , "xrp"]
type(x)

###################
# Dictionary ( Sözlük )  {key,value}
###################

x = { "name" : "Melisa" , "Age ": 25}
type(x)

###################
# Tuple
###################

x = ("python" ,"ml" , "ds")
type(x)

###################
# Set
###################

x = {"python", "ml","ds"}
type(x)

###################
## Liste, set,tuple ve dictionary veri yapıları Python Collections ( Arrays ) olarak geçmektedir.
###################

a = 5
b = 10.5

a * 3
a / 7
a ** 2

###################
## Tipleri Değiştirme
###################

int(b)
float(a)

###################
## Karakter Dizileri ( Strings )
###################

print("Melisa")
name = "Çağrı"

###################
## Çok sayıda karakter dizileri
###################

""" sklsfjnfsdlknlksnflkdn 
sddsfdsf
dsfdfsfds
fsdfsfddffddsf
dsffsdfdsfdsfdsfd
"""

###################
# Karakter dizilerinin elemanlarına erişmek
###################

name[0]

###################
# Karakter dizilerinde slice işlemi
###################
name[0:3]

###################
# Karakter dizilerinde eleman sorgulama
###################

###################
# String Metods : Çeşitli görevleri yerine getiren fonksiyon benzeri yapılardır.
# Class'lar içerisinde tanımlanan fonksiyonlardır.
###################

# bu fonksiyonlara böyle ulaşırız => dir(data_type)
dir(str)
dir(int)

###################
## len = stringlerde boyut bilgisine erişmek için kullanılır. python'ın içerisinde gömülü olarak gelir.
###################

name2 = "Melisa"
type(name2)
len(name2)
len("Melisa Çevik")

##kullanmış olduğumuz bir yapının metod mu yoksa fonksiyon mu olduğunu nasıl ayırt ederiz?
## Eğer bir fonksiyon class yapısı içerisinde tanımlandıysa = metod denir .
## Eğer bir class yapısı içerisinde tanımlanmadıysa = fonksiyon denir.
## command' bas => builtins diye bir script açıldı. örneğin len, bir classın içinde değil => fonksiyon
## metod ve fonksiyon aynı şeylerdir fakat farkı şu, fonksiyonların bağımsız, metodların classlar içinde tanımlanmış olmasıdır.

#####################
# upper() & lower(): büyük-küçük dönüşümleri
#####################

"miuul".upper()
"MIUUL".lower()

#####################
# replace : karakter değiştirme
#####################

hi = "Hello AI Era"
hi.replace("l","p")

#####################
# split : böler
#####################

"Hello AI Era".split()

#####################
# strip : kırpar
#####################

" ofofo ".strip()
"ofofo".strip("o")

#####################
# capitalize : ilk harfi büyütür
#####################

"foo".capitalize()

#####################
# Liste :
# - Değiştirilebilir ,
# - sıralı, => veri yapısının elemanlarına erişebiliyor anlamına gelir.
# - kapsayıcı [ içerisinde birden fazla veri yapısını aynı anda tutabilir. ]
# bir veri yapısıdır.
#####################

notes = [1 , 2 , 3, 4]

not_nam = [1,2,3,"a",True, [6,7,8,"a"]]

not_nam[0]
not_nam[5][1]

type(not_nam[5])
type(not_nam[5][1])

notes[0] = 99

#####################
# List Methods
# append() ( en önemli )
#####################

dir(notes)

#####################
# len : builtin python fonksiyon, boyut bilgisi.
#####################

len(notes)

#####################
# append : eleman ekler
#####################

notes.append(100)

#####################
# pop : indexe göre siler
#####################

notes.pop(0)

#####################
# insert : indexe ekler  insert(kacinciindex, deger)
#####################

notes.insert(2, 98)

#####################
# Sözlük (Dictionary) : key-value ile veri tutma imkanı sağlayan, listelere göre daha az yaygın bir şekilde kullanılan bir veri yapısıdır.
#Değiştirilebilir
#Sırasız
#Kapsayıcı
#####################

# key-value

dictionary = { "REG" : "Regression" ,
               "LOG" : "Logistik Regression",
               "CART": "Classification and Reg"}

dictionary["REG"]

dictionary = { "REG" : ["RMSE" , 10] ,
               "LOG" : ["MSE",20],
               "CART": ["SME", 30]}

dictionary["REG"][1]
#  bu şekilde de erişiriz dictionary.get("REG") de aynı

"YSA" in dictionary

## Value değiştirme

dictionary["REG" ] = ["YSA", 10]

## Tüm Key'lere erişme

dictionary.keys()

## Tüm value'lara erişme
dictionary.values()

##Tüm çiftleri tuple halinde listeye çevirme
dictionary.items()

## Key-Value Değerini Güncellemek : Varsa günceller , key'i bulamazsa dictionarynin sonuna ekler. (2.örnek)

dictionary.update({"REG" : 11})
dictionary.update({"RF" : 44})

#####################
##TUPLE  : Liste'lerin değişime kapanmış halidir.
#Değiştirilemez.
#Sıralıdır.
#Kapsayıcıdır.
#####################

t = ( "john" , "mark", 1,2)
type(t)

t[0]
t[0:3]

t[0] = 99 ##errorrrrrr. DEĞİŞTİREMEZSİNNNN!
## değiştirmenin bir yolu var elbette. LİSTE'ye çevirerek elemanlarını değiştirebilirsin. sonra tuple'a tekrar çevirebilirsin. list'e çevir > tuple'a çevir

t = list(t)
t[0] = 99
t = tuple(t)

#####################
# SET (Küme)
# Değiştirilebilir
# Sırasız - Eşsizdir
# Kapsayıcıdır
#####################

#####################
# difference() : İki kümenin farkı
#####################

set1 = set([1,3,5])
set2 = set([1,2,3])

# set1'de olup set2'de olmayanlar / set2'de olup set1'de olmayanlar da yapılır aynı sekilde
set1.difference(set2)

# symmetric_difference(): İki kümede de birbirlerine göre olmayanlar

set1.symmetric_difference(set2)
set2.symmetric_difference(set1) ##yukarıdaki ile aynı sonuc

#####################
# intersection(): İki kümenin kesişimi
#####################

set1.intersection(set2)

set1 & set2 ##küme kesişimi
set1 - set2 ##küme farkı

#####################
# union(): İki kümenin birleşimidir
#####################

set1.union(set2)

#####################
# isdisjoint(): İki kümenin birleşimi boş mu?
#####################

set1.isdisjoint(set2)

#####################
# issubset(): Bir küme diğerinin alt kümesi mi?
#####################

set1.issubset(set2)

#####################
# issuperset(): Bir küme diğer kümeyi kapsıyor mu?
#####################

set1.issuperset(set2)