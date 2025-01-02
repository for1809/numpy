import numpy as np
# NumPy 상수 연산
# 1. 덧셈, 곱셈
array = np.random.randint(1, 10, size = 4).reshape(2, 2)
print(array)
print(array + 3)
print(array * 3)

# 서로 다른 형태의 NumPy 연산
# 서로 다른 형태의 배열을 연산할 때는 행 우선 수행(broadcasting)
array1 = np.arange(4).reshape(2, 2)
array2 = np.arange(2)
array3 = array1 + array2
array4 = np.array([[0, 1],
                   [0, 1]])
print(array3)
print(array1 + array4) # 결과가 같음(아래 행으로 확장하여 덧셈)

array1 = np.arange(8).reshape(2, 4)
array2 = np.arange(8).reshape(2, 4)
array3 = np.concatenate([array1, array2], axis = 0) # 세로로 결합
array4 = np.arange(4).reshape(4, 1)

print(array3 + array4)

# 마스킹 연산
array1 = np.arange(16).reshape(4, 4)
array2 = array1 < 10
print(array2)
array1[array2] = 100 # True 위치의 원소들에 대해서만 실행
print(array1)

# 집계 함수
array = np.arange(16).reshape(4, 4)
print('최대값: ', np.max(array))
print('최소값: ', np.min(array))
print('합계: ', np.sum(array))
print('평균: ', np.mean(array))
print('열 합: ', np.sum(array, axis = 0))
print('행 합: ', np.sum(array, axis = 1))