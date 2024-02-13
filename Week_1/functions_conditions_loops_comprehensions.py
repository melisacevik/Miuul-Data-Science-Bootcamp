##################
# FONKSİYONLAR, KOŞULLAR, DÖNGÜLER, COMPREHENSIONS
##################
# Functions
# Koşullar (Conditions)
# Döngüler (Loops)
# Comprehesions

#################
# Functions
#################

#################
# Fonksiyon Okuryazarlığı
#################

print("a", "b")
print("a","b", sep="__")

#################
# Fonksiyon Tanımlama
#################


def calculate(x):
    print(x*2)

calculate(4)

## İki argümanlı/parametreli bir fonksiyon tanımlayalım.
# sırası önemli

def summer(arg1,arg2):
    print(arg1 + arg2)

summer(4,5)

def summer(arg1,arg2):
    """
    sum of 2 numbers

    :param arg1: int,float
    :param arg2: int,float
    :return: int,float
    """
    print(arg1 + arg2)

summer()

#######################
# Fonksiyonların Statement/Body Bölümü
#######################

# def function_name(parameters/arguments):
#       statements (function body)

def say_hi():   ##argümansız da olur
    print("meloş")
    print("hi")
    print("hello")

say_hi("Melisa")


def multiplication(a,b):

   c = (a * b)
   print(c)

multiplication(4,5)

## girilen 2 sayıyı çarpsın ve sonucu bir liste içinde saklayacak fonksiyon

list_store = []               # global scope

def add_element(a,b):
    c = a * b                  # local scope
    list_store.append(c)
    print(list_store)

add_element(1,2)
add_element(3,4)
add_element(5,7)

#######################
# Ön Tanımlı Argümanlar/Parametreler (Default Arguments/Parameters):
# Teknik olarak parametre, fonksiyon tanımlanması esnasında kullanılan ifadelerdir.
#######################

def divide(a,b):
    print(a / b)

def divide(a,b=1):
    print(a / b)

divide(80)

#######################
# Ne zaman fonksiyon yazma ihtiyacımız olur?
#######################

# Don't repeat yourself

def calculate(varm, moisture, charge):
    print((varm + moisture) / charge)

calculate(96,12,78)
calculate(33,44,55)

#######################
# Return : Fonksiyon Çıktılarını Girdi Olarak Kullanmak
#######################

def calculate(varm, moisture, charge):
    print((varm + moisture) / charge)

# calculate(96,12,78) * 10

def calculate(varm, moisture, charge):
    return (varm + moisture) / charge

calculate(96,12,78) * 10

##birden fazla işlem girdiğinde tuple olarak çıktı veriyor!!

#######################
# Fonksiyon İçerisinden Fonksiyon Çağırmak
#######################

def calculate(varm, moisture, charge):
    return int((varm + moisture) / charge)

calculate(90,12,12) * 10

def standardization(a, p):
    return a * 10 / 100 * p * p

standardization(45,1)

def all_calculation(varm, moisture,charge, p):
    a = calculate(varm, moisture, charge)
    b = standardization(a, p)
    print(b * 10)

all_calculation(1,3,5,12)

############################
# Local & Global Variables
############################

list_store = [1,2]

def add_element(a,b):
    c = a * b                  # local scope
    list_store.append(c)
    print(list_store)

############################
# KOŞULLAR (CONDITIONS)
############################

1 == 1
1 == 2

#####################
#if
#####################

if 1 == 1:
    print("smth")

number = 11

if number == 10:
    print("number is 10")

def number_check(number):
    if number == 10:
        print("number is 10")

number_check(12)

#####################
#else
#####################

def number_check(number):
    if number == 10:
        print("number is 10")
    else:
        print("print is not 10")

number_check(12)

#####################
#elif
#####################

def number_check(number):
    if number > 10:
        print("number is greater than 10 ")
    elif number < 10:
        print("number is less than 10")
    else:
        print("invalid number")

number_check(11)

#####################
# LOOPS (Döngüler)
#####################
#for loop

students = ["portakal", "çitos", "osman", "süslü"]

## her öğrenciyi gezmek için for i in students diyoruz
for student in students:
    print(student)

## isimlerini büyütmek istersek

for student in students:
    print(student.upper())

salaries = [ 1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    print(salary)

# herkese %20 zam

for salary in salaries:
    print(int(salary * (20/100) + salary))

## maaşları %20 , %30, %50 olarak tek tek yazmam kendimi tekrar etmeme yol açar o yüzden fonksiyon yazıyorum.

def new_salary(salary, rate):
    return int(salary * rate/100 + salary)

new_salary(1500,10)
new_salary(2000,20)

for salary in salaries:
    print(new_salary(salary, 10))

#maaşı 3000'den yukarısı olanlara farklı zam uygulasın

for salary in salaries:
    if salary >= 3000:
        print(new_salary(salary,10))
    else:
        print(new_salary(salary,20))

## maaşı tek tek gez
## her bir maaş için 3000den büyük mü kontrol et, new_salary fonks. çalıştır
## er bir maaş için 3000den büyük değilse %20 zam yap.

#############################
# Alternating ( MÜLAKAT SORUSU )
#############################

# Amaç aşağıdaki string değiştiren fonksiyon yazmak istiyoruz.

# before : "hi my name is john and i am learning python"
# after : "Hi mY NaMe iS JoHn aNd i aM LeArNiNg pYtHoN"

def alternating(string):
    new_string =""
                        # stringin indexlerinde gezme => range()
    for string_index in range(len(string)):
        #index çift ise büyük harfe çevir
        if string_index % 2 == 0:
            new_string += string[string_index].upper()
        # index tek ise küçük harfe çevir
        else:
            new_string += string[string_index].lower()
    print(new_string)

alternating("hi my name is john and i am learning python")

#########################
# break & continue & while
#########################
# break : akışı kesmek, durdurmak için
# continue : ilgili şart gözlemlendiğinde o şartı atlamak için ( pas geçmek için )
# while(-dığı sürece) : bir koşul sağlandığı sürece çalışmayı sürdürür.

salaries = [ 1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    if salary == 3000:
        break
    print(salary)

for salary in salaries:
    if salary == 3000:
        continue
    print(salary)

number = 1

while number < 5:
    print(number)
    number += 1

################
# Enumerate : Otomatik Counter/Indexer ile for loop
# Bir interatif nesne içinde elemanları gezip aynı zamanda index bilgisini de takip etmemizi sağlar.
################

## örnek => çift indexlileri A listesine , tek indexlileri B listesine alma

students = ["portakal", "çitos", "osman", "süslü"]

for student in students:
    print(student)     #burada index bilgisini output olarak alamadık


for index,student in enumerate(students):
    print(index,student)  # burada index bilgisini de alabildik

A = []
B = []
for index,student in enumerate(students):
    if index % 2 == 0:
        A.append(student)
    else:
        B.append(student)
print(students)
print(A)
print(B)

################
# Uygulama - Mülakat Sorusu
################
#divide_students fonksiyonu yazınız.
#Çift indexte yer alan öğrencileri bir listeye alınız.
#Tek indexte yer alan öğrencileri bir listeye alınız.
#Fakat bu iki liste tek bir liste olarak return olsun.

students = ['John', 'Mark', 'Vanessa', 'Mariam']

def divide_students(students):
    groups = [[],[]]
    for index,student in enumerate(students):
        if index % 2 == 0:
            groups[0].append(student)
        else:
            groups[1].append(student)
    print(groups)
    return groups

st = divide_students(students)
st[0]
st[1]

#################
# Alternating Fonksiyonunun Enumerate ile Yazılması
#################

def alternating_with_enumerate(string):
    new_string = ""
    for i,letter in enumerate(string):
        if i % 2 == 0:
            new_string += letter.upper()
        else:
            new_string += letter.lower()
    print(new_string)

alternating_with_enumerate("hi mY naMe is MeliSa")

#################
# Zip : Birbirinden farklı şekilde olan listeleri bir arada değerlendirme imkanı sağlar.
#################

players = ["Kobe Bryant" , "Michael Jordan" , "Alperen Şengün"]

teams = ["Los Angeles Lakers","Chicago Bulls","Houston Rockets"]

heights = [1.98 , 1.98, 2.11]

list(zip(players,teams,heights))

#################
# Lambda & Map & Filter & Reduce
#################

# Proto : Lambda : İsimsiz fonksiyonlardır.
# Bir fonksiyonlara parametre olarak gönderirken kullanıyoruz. Filter, map gibi.

denklem = lambda x: x * 2
sonuc = denklem(2)
print(sonuc)

# soyisminin ilk harfine göre sıralama yapsın.

isim_soyisim = ['Melisa Cevik', "Cagri Sabanci", "Portakal Qabanci", "Citos Rabanci"]
isim_soyisim.sort(key=lambda x: x.split(' ')[-1].lower())

print(isim_soyisim)
# name = "Melisa Cevik".split()[-1].lower()
# print(name)

##--------

#Miuul
#Lambda => fonksiyon tanımlama şeklidir. atama yapılmaksızın kullanılabilir.
#

def summer(a,b):
    return a + b

new_summer = lambda a,b: a+b; # şu an kullan- at özelliğine bakmıyoruz nasıl tanımlanır diye öğrendik

new_summer(4,5)

#
# Map => Döngü yazmaktan kurtarır.
#

salaries = [1000,2000,3000,4000]

def new_salary(x):
    return x * 20 / 100 + x   # lambda + map ile buna da ihtiyaç kalmadı.

new_salary(1000)

for salary in salaries:
    print(new_salary(salary))    # bu uzun yol

        #bu fonksiyonu, bu nesneye mapledi
list(map(new_salary, salaries))  # bu kısa yol

#
# lambda - map ilişkisi
#

# del new_summer

list(map(lambda x: x * 20 / 100 + x, salaries)) #en kısa yol

# Map - Proto : filter fonksiyonu ile benzer fakat map ile listenin içindeki elemanlara müdahale edebiliyoruz.

liste = [10,8,7,5,3,2]

yeni_liste = list(map(lambda x: x *2, liste))

print(yeni_liste)

############
# Filter
############

list_store = [1,2,3,4,5,6,7,8,9] #çift varsa filtreleyip listeleyeceğiz.

list(filter(lambda x: x % 2 == 0, list_store))

# Filter : Prototürk

number_list = [10,9,3,4,5,1,2,3,8,7]
new_list = list(filter(lambda x:(x % 2 == 0) ,number_list))
print(new_list)

# filter 2 parametre ile çalışır. 1.function 2.iteration list. function => lambda
# geriye obje döndürdüğü için list ile sarmalıyoruz

############
# Reduce : İndirgemek
############
from functools import reduce
list_store = [1,2,3,4]
reduce(lambda a,b: a+b, list_store) # peş peşe elemanları ekler


################
# COMPREHENSIONS
################

#
# 1) List Comprehension : Bu yapı ile tek satırda, hem bir döngü, hem bir if else, hem de belirli bir fonksiyonu , döngü elemanlarına uygulama işlemi sağlar.
#
def new_salary(x):
    return x * 20 / 100 + x

salaries = [1000, 2000, 3000, 4000]

null_list = []

for salary in salaries:
    null_list.append(new_salary(salary))

null_list = []

for salary in salaries:
    if salary > 3000:
        null_list.append(new_salary(salary))
    else:
        null_list.append(new_salary(salary * 2))


## bu işlemlerin yerine aşağıdaki çözümde aynı sonucu verecektir.

[new_salary(salary * 2) if salary < 3000 else new_salary(salary) for salary in salaries]

# list comprehensions nasıl oluşturulur ?
salaries = [1000, 2000, 3000, 4000]
[salary * 2 for salary in salaries]  #maaşları 2 ile çarpmak

[salary * 2 for salary in salaries if salary < 3000 ]  # maaşı 3000'den az olanları 2 ile çarp

## DİKKAT! if tek başına else olmadan çalışacaksa en sağa yazılır.
## else ile kullanacaksak for bloğu sağ tarafta kalır.

[salary * 2 if salary < 3000 else salary * 0 for salary in salaries]  # maaşı 3000'den büyük olanlar için else

[new_salary(salary * 2) if salary < 3000 else new_salary(salary * 0.2) for salary in salaries]  #var olan bir fonksiyonumu kullanmak istersem? (en başa )

## if tek başına olursa => [ işlem ; for ; if ]
## if-else olursa => [ işlem ; if-else ; for]

# ÖRNEK

students = ["John", "Mark", "Venessa", "Mariam"]

students_no = ["John", "Vanessa"]

#istemediğim öğrencilerin isimlerini küçük(student_no'dakiler istemediğim) , diğerlerini büyük yazsın.

[student.lower() if student in students_no else student.upper() for student in students]

#
# Dict Comprehension
#

dictionary = {'a':1,
              'b':2,
              'c':3,
              'd':4}

dictionary.keys()
dictionary.values()
dictionary.items()

# Örnek => her value'nun karesini almak istiyoruz.

{k: v ** 2 for (k,v) in dictionary.items()}

# key'leri büyütmek istersem;

{k.upper(): v for (k,v) in dictionary.items()}

# aynı anda ikisini de yapabilirim;

{k.upper(): v **2 for (k,v) in dictionary.items()}

#
# Dict Comprehensions - Mülakat Sorusu -
#

# Amaç : Çift sayıların karesi alınarak bir sözlüğe eklenmek istemektedir
# Key'ler orijinal değerler, valuelar ise değiştirilmiş değerler olacak.

numbers = range(10)

new_dict = {}

for n in numbers:
    if n % 2 == 0:
        new_dict[n] = n ** 2

# dict ile de çözülür

{n : n **2 for n in numbers if n % 2 == 0}

#######################
# List - Dict Comprehensions -Uygulama 1-
#######################

########
# Bir Veri Setindeki Değişken İsimlerini Değiştirmek
########

#before:
#['total', 'speeding', 'alcohol','not_distracted', 'no_previous','ins_premium', 'ins_losses', 'abbrev']

#after:
#['TOTAL', 'SPEEDING', 'ALCOHOL','NOT_DISTRACTED', 'NO_PREVIOUS','INS_PREMIUM', 'INS_LOSSES', 'ABBREV']

# DataFrame(Veri Çerçevesi) Nedir ? Pandas modülünün, paketinin, kütüphanesinin bir veri yapısıdır.
# Birbirinden farklı veri tiplerini tutma imkanı sağlar.
# Excel tarzında veri tutmak şeklinde düşünülebilir.

import seaborn as sns                  #import ettik
df = sns.load_dataset("car_crashes")   #seaborn kütüphanesi içerisinden "car_crashes" veri setini getir ve df şeklinde kaydet.
df.columns                             #ilgili dataframe'in değişkenlerinin ismi gelir

A = []

for col in df.columns:
    print(A.append(col.upper()))

df.columns = A

# List comprehension ile çözümü

df = sns.load_dataset("car_crashes")

df.columns = [col.upper() for col in df.columns]

#######################
# List - Dict Comprehensions -Uygulama 2-
#######################

########
# İsminde "INS" olan değişkenlerin başına FLAG diğerlerine NO_FLAG eklemek istiyoruz.
########

#before:
# [
# 'TOTAL',
# 'SPEEDING',
# 'ALCOHOL',
# 'NOT_DISTRACTED',
# 'NO_PREVIOUS',
# 'INS_PREMIUM',
# 'INS_LOSSES',
# 'ABBREV']

#after:
# [
# 'NO_FLAG_TOTAL',
# 'NO_FLAG_SPEEDING',
# 'NO_FLAG_ALCOHOL',
# 'NO_FLAG_NOT_DISTRACTED',
# 'NO_FLAG_NO_PREVIOUS',
# 'FLAG_INS_PREMIUM',
# 'FLAG_INS_LOSSES',
# 'NO_FLAG_ABBREV']

[col for col in df.columns if "INS" in col]  #içinde INS olan ifadeleri getir

["FLAG_" + col for col in df.columns if "INS" in col]  # INS olan değişkenlerin başına FLAG_ ekledik

["FLAG_" + col if "INS" in col else "NO_FLAG_" for col in df.columns] #INS olmayanlara NO_FLAG_ ekledik

df.columns = ["FLAG_" + col if "INS" in col else "NO_FLAG_" for col in df.columns]


#######################
# List - Dict Comprehensions -Uygulama 3-
#######################

########
# Amaç key'i string, value'su aşağıdaki gibi bir liste olan sözlük oluşturmak.
# Sadece sayısal değişkenler için yapacağız. ( Sayısal değişkenleri seçeceğiz. )
########

# Output:
# {'total' : ['mean,'min', 'max', 'var']
#'speeding' : ['mean,'min', 'max', 'var']
#'alcohol' : ['mean,'min', 'max', 'var']
#'not_distracted' : ['mean,'min', 'max', 'var']
#'no_previous' : ['mean,'min', 'max', 'var']
#'ins_premium' : ['mean,'min', 'max', 'var']
#'ins_loses' : ['mean,'min', 'max', 'var']
# }

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

num_cols = [col for col in df.columns if df[col].dtype != 'O'] #obje olmayanları aldık
soz = {}
agg_list = ["mean","min","max","sum"]

new_dict = {col: agg_list for col in num_cols}

df[num_cols].head()

df[num_cols].agg(new_dict)

##########
# Proto : Recursive Fonksiyonlar : Çalışma esnasından tekrar kendini çağıran fonksiyonlara denir.
##########

def factorial(x):
    if x == 1 or x == 0:
        return 1                     # 1
    else:
        return x * factorial(x - 1)  # 4 ,3 ,2

result = factorial(4)
print(result)

##########
# Proto # *args, **kwargs
##########

def toplam(*args,**kwargs):
    print(args)
    print(kwargs)

toplam(1,2,3,isim = 'Melisa', soyisim = 'Cevik')

