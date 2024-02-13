############################################
# SORTING REVIEWS
############################################

import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

###################################################
# Up-Down Diff Score = (up ratings) − (down ratings)
###################################################

# Review 1: 600 up 400 down total 1000
# Review 2: 5500 up 4500 down total 10000

def score_up_down_diff(up, down):
    return up - down

score_up_down_diff(600, 400) # %60

score_up_down_diff(5500, 4500) # %55

###################################################
# Score = Average rating = (up ratings) / (all ratings)
###################################################
# faydalı oranı gibi düşünülebilir.

def score_average_rating(up, down):
    if up + down == 0:
        return 0 #payda için excep. koyduk
    return up / (up + down)

score_average_rating(600, 400)
score_average_rating(5500, 4500)

# Review 1: 2 up 0 down total 2
# Review 2: 100 up 1 down total 101

score_average_rating(2, 0)
score_average_rating(100, 1)

# frekans yüksekliğini göz önünde bulunduramadı.!!! hatalı !! 2. review daha önde olmalıydı!

###################################################
# Wilson Lower Bound Score
###################################################

# 600-400
# 0.6
# 0.5 0.7
# 0.5

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

# p istatistiğinin güven aralığı formülü

wilson_lower_bound(600, 400)
wilson_lower_bound(5500, 4500)

wilson_lower_bound(2, 0)
wilson_lower_bound(100, 1)

# elimdeki örnekleme ilişkin olarak up rate oranının istatistiksel olarak %95 ve %5 hata payı ile hangi aralıkta olacağını biliyorum.


wilson_lower_bound(2, 0)
wilson_lower_bound(100, 1)


###################################################
# Case Study
###################################################

up = [15, 70, 14, 4, 2, 5, 8, 37, 21, 52, 28, 147, 61, 30, 23, 40, 37, 61, 54, 18, 12, 68]
down = [0, 2, 2, 2, 15, 2, 6, 5, 23, 8, 12, 2, 1, 1, 5, 1, 2, 6, 2, 0, 2, 2]
comments = pd.DataFrame({"up": up, "down": down})


# score_pos_neg_diff
comments["score_pos_neg_diff"] = comments.apply(lambda x: score_up_down_diff(x["up"],
                                                                             x["down"]), axis=1)

# score_average_rating
comments["score_average_rating"] = comments.apply(lambda x: score_average_rating(x["up"], x["down"]), axis=1)

# wilson_lower_bound
comments["wilson_lower_bound"] = comments.apply(lambda x: wilson_lower_bound(x["up"], x["down"]), axis=1)

# burada wlb 'ye göre sıralama yapmalıyız!

comments.sort_values("wilson_lower_bound", ascending=False)

# Rating
# önümüzde bir puan hesabı işi olduğunda:
# avg alabilirim ama bunu daha da hassaslaştırabilirim. zamana dayalı, userQuality'e dayalı hassaslaştırabilirim.
# elimde 5 yıldızlı bir rating olduğunda Bayesian Avg. Rating ile hesaplayabilirim.

# Sorting
# puanladıktan sonra sıralıyoruz.

# Yorum sıralama
# wlb score yöntemi ile çözdük!