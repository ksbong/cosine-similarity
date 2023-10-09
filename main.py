from konlpy.tag import Okt
from numpy import dot
from numpy.linalg import norm
import numpy as np


def cos_sim(a, b):
    # 두 벡터에 대하여 둘의 내적한 값을
    # 두 벡터의 크기의 곱으로 나눈다.
    # dot() 행렬 곱 norm() 벡터의 노름(벡터의 크기)
    return dot(a,b)/(norm(a)*norm(b))

def make_matrix(feats, list_data):
    freq_list = []
    for feat in feats:
        freq = 0
        for word in list_data:
            if feat == word:
                freq += 1

        freq_list.append(freq)
    return freq_list


okt = Okt()

text1 = '물가 상승으로 인해 최근에 식료품 가격이 상당히 올랐다.'
text2 = '최근 들어서 스마트폰의 기능이 점점 발전하여, 사람들의 일상생활에서 이용되는 빈도가 더욱 높아졌다.'


# 형태소 분류
v1 = okt.nouns(text1)
v2 = okt.nouns(text2)

# 두 문장을 합쳐서 단어 중복 제거
v3 = v1 + v2
feats = set(v3)
'''
{'진천', '재학', '철수', '광혜원', '고등학교', '영희', '저'}
'''

# [진천 재학 철수 광혜원 고등학교 영희 저]
# 생성되는 벡터의 각 요소는 위 단어들을 뜻한다.

# 각 단어의 빈도수를 담은 벡터 생성
v1_arr = np.array(make_matrix(feats, v1))
# [0 0 1 1 1 1 1]

v2_arr = np.array(make_matrix(feats, v2))

# [0 1 1 1 1 0 1]


# 코사인 유사도 계산 0~1
cs = cos_sim(v1_arr, v2_arr)

print(cs*100)