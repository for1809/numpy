import numpy as np
# Numpy 저장과 불러오기

# 단일 객체 저장 및 불러오기
array = np.arange(10)
np.save('saved.npy', array)

result = np.load('saved.npy')
print(result)

# 복수 객체 저장 및 불러오기
array1 = np.arange(10)
array2 = np.arange(10, 20)
np.savez('saved.npz', array1 = array1, array2 = array2)

data = np.load('saved.npz')
result1 = data['array1']
result2 = data['array2']
print(result1)
print(result2)

# Numpy원소의 정렬

# 오름차순 정렬
array = np.array([5, 9, 10, 3, 1])
array.sort() # 오름차순
print(array)
# 내림차순
print(array[::-1])

# 열 기준 정렬
array = np.array([[5, 9, 10, 3, 1],
                  [8, 3, 4, 2, 5]])
array.sort(axis = 0)
print(array)

# 균일한 간격으로 데이터 생성
array = np.linspace(0, 10, 5)
print(array)

# 난수의 재연(실행마다 결과 동일하게(
np.random.seed(1)
print(np.random.randint(0, 10, (2, 3)))

# 배열 객체 복사
array1 =np.arange(10)
array2 = array1
array2[0] = 99
print(array1) # [99  1  2  3  4  5  6  7  8  9] array2 를 수정하였으나 두 배열이 같은 주소를 가르켜 array1까지 수정됨
array2 = array1.copy()
array2[1] = 88
print(array1) # [99  1  2  3  4  5  6  7  8  9] 유지

# 중복된 원소 제거
array = np.array([1, 1, 3, 2, 3, 2])
print(np.unique(array)) # 중복된 원소 제거