#################
# Görev 1) Verilen değerlerin veri yapılarını inceleyiniz.
#################
from typing import List, Any

x = 8
type(x)
y = 3.2
type(y)
z = 8j + 18
type(z)
a = "Hello World"
type(a)
b = True
type(b)
c = 23 < 22  # false
type(c)
l = [1, 2, 3, 4]
type(l)
d = {"Name": "Jake",
     "Age": 27,
     "Adress": "Downtown"}
type(d)

t = ("Machine Learning", "Data Science")
type(t)

s = {"Python", "Machine Learning", "Data Science"}
type(s)  # sırasız, unique
# s[1] error

####################
# Görev 2) Verilen string ifadenin tüm harflerini büyük harfe çeviriniz.
# Virgül ve nokta yerine space koyunuz, kelime kelime ayırınız.
####################

text = "The goal is to turn data into information, and information into insight."

# 1. çözüm
new_text = text.upper().replace(",", " ").replace(".", " ").split()

# 2.çözüm regEx ile de çözülür.

import re

text = text.upper()
list_of_words = re.sub(r"[.,]", " ", text.upper()).split()

####################
# Görev 3) Verilen listeye aşağıdaki adımları uygulayınız
####################

lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]

# Adım 1: Verilen listenin eleman sayısına bakınız.
len(lst)

# Adım 2: 0. ve 10.indexteki elemanları çağırınız.
lst[0]
lst[10]

# Adım 3: Verilen liste üzerinden "D","A","T","A" listesi oluşturunuz.

data_list = lst[0:4]  # 0,1,2,3
# data_list = lst[0:3]

# Adım 4: 8.indexteki elemanı siliniz.

# lst.remove(lst[8])
lst.pop(8)
# del lst[8]

# Adım 5: Yeni bir eleman ekleyiniz.
lst.append("M")

# Adım 6: 8.indexe "N" elemanını tekrar ekle.
lst.insert(8, "N")

####################
# Görev 4 : Verilen sözlük yapısına aşağıdaki adımları uygulayınız.
####################

dict = {"Christian": ["America", 18],
        "Daisy": ["England", 12],
        "Antonio": ["Spain", 22],
        "Dante": ["Italy", 25]
        }

# Adım 1: Key değerlerine erişiniz.
dict.keys()
# Adım 2: Value değerlerine erişiniz.
dict.values()
# Adım 3: Daisy key'ine ait 12 değerini 13 olarak güncelle.
dict["Daisy"][1] = 13
# dict['Daisy'] = ['England', 13]
# Adım 4: Key değeri = Ahmet , Value = ["Turkey", 24] olan yeni bir değer ekleyiniz.
dict["Ahmet"] = ["Turkey", 24]
# update ile de olur.
# Adım 5: Antonio'yu listeden siliniz.
dict.pop("Antonio")


####################
# Görev 5 : Argüman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları ayrı listelere atayan ve bu listeleri return eden fonksiyon yazınız
####################

# 1. çözüm

def my_list(input_list):
    even_list = []
    odd_list = []

    for i in range(len(input_list)):
        if i % 2 == 0:
            even_list.append(input_list[i])
        else:
            odd_list.append(input_list[i])
    return even_list, odd_list


even_list, odd_list = my_list([2, 13, 18, 93, 22])
print("Even List: ", even_list)
print("Odd List:", odd_list)

# 2. çözüm

l = [2, 13, 18, 93, 22]


def odds_evens(numbers: list[int]) -> (list[int], list[int]):
    """Takes in a list of numbers and returns two lists => odds, evens

    Args:
        numbers: list of int numbers

    Returns:
        odds: list of odd numbers
        evens: list of even numbers

    """
    # odds = [number for number in numbers if number % 2 != 0]
    # evens = [number for number in numbers if number % 2 == 0]
    odds = []
    evens = []
    [odds.append(number) if number % 2 != 0 else evens.append(number) for number in numbers]
    return odds, evens


odds_evens(l)
####################
# Görev 6 : Aşağıda verilen listede
# mühendislik ve tıp fakülterinde dereceye giren öğrencilerin isimleri bulunmaktadır.
# Sırasıyla ilk üç öğrenci mühendislik fakültesinin başarı sırasını temsil ederken
# Son üç öğrencide tıp fakültesi öğrenci sırasına aittir.
# Enumarate kullanarak öğrenci derecelerini fakülte özelinde yazdırınız.
####################

students = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]
students[0:3]
students[:3]
students[3:]


def divide_students(students):
    eng_students = students[:3]  # 0,1,2
    medic_students = students[3:6]  # 3,4,5

    for index, student in enumerate(eng_students, 1):
        print("Mühendislik Fakültesi " + str(index) + ". öğrenci : " + student)

    for index, student in enumerate(medic_students, 1):
        print("Tıp Fakültesi " + str(index) + ". öğrenci : " + student)


divide_students(students)

ogrenciler = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]

for i, ogrenci in enumerate(ogrenciler):  # ogrenci refers to student
    if i < 3:
        print(f"{ogrenci}, Mühendislik Fakültesi")  # "Mühendislik Fakültesi" refers to "Engineering Faculty"
    else:
        print(f"{ogrenci}, Tıp Fakültesi")  # "Tıp Fakültesi" refers to "Medical Faculty"

####################
# Görev 7 : Aşağıda 3 adet liste verilmiştir. Listelerde sırası ile bir dersin kodu, kredisi
# ve kontenjan bilgileri yer almaktadır. Zip kullanarak ders bilgilerini bastırınız.
####################

ders_kodu = ["CMP1005", "PSY1001", "HUK1005", "SEN2204"]
kredi = [3, 4, 2, 4]
kontenjan = [30, 75, 150, 25]

dersler = list(zip(ders_kodu, kredi, kontenjan))
print(dersler)

for i, ders in enumerate(dersler):
    print("Kredisi", dersler[i][1], "olan", dersler[i][0], "kodlu dersin kontenjanı", dersler[i][2], "kişidir.")

# [print("Kredisi {} olan {} kodlu dersin kontenjanı {} kişidir.".format(a,b,c)) for a,b,c in zip(kredi,ders_kodu,kontenjan)]

####################
# Görev 8: Aşağıda 2 adet set verilmiştir. Sizden istenilen eğer
# 1.küme 2.kümeyi kapsıyor ise ortak elemanlarını,
# kapsamıyor ise 2.kümenin 1.kümeden farkını yazdıracak
# fonksiyonu tanımlamanız beklenmektedir.
####################

kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])


def kume(kume1, kume2):
    if kume1.issuperset(kume2):
        print("ortak elemanlar:", kume1.intersection(kume2))
    else:
        print("küme farkı: ", kume2.difference(kume1))


kume(kume1, kume2)
