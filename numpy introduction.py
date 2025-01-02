# 출처: geeksforgeeks NumPy Introduction https://www.geeksforgeeks.org/introduction-to-numpy/?ref=gcse_outind, 유튜브 나동빈
'''
다차원 배열을 나타냄
Numpy에서 차원은 axes라고 불리며 차원 수는 rank라고 칭함
NumPy의 배열 클래스는 ndarray라 불림
다차원 배열 (Multidimensional Array): 1차원, 2차원, 3차원 이상의 배열을 모두 지원합니다.
동일한 데이터 타입: 하나의 ndarray는 같은 데이터 타입의 요소만 저장할 수 있습니다.
연속된 메모리 블록: 배열 요소들이 메모리에 연속적으로 저장되어 있어 접근 속도가 빠릅니다.
벡터화 연산: 반복문 없이 배열 단위로 연산이 가능하여 속도가 빠릅니다.
'''
# ex1
import numpy as np
# 배열 생성
arr = np.array([[1, 2, 3], [4, 2, 5], [4, 2, 5]])
# 배열의 타입 출력
print(type(arr)) # <class 'numpy.ndarray'>
# 배열의 차원 수 출력
print(arr.ndim) # 3
# 배열의 shape 출력
print(arr.shape) # (3, 3) 3행 3열
# 배열의 사이즈(원소의 총 개수) 출력
print(arr.size)
# 원소의 타입 출력
print(arr.dtype) # int64

# 1. 배열 생성
# 리스트, 튜플
# a. 실수 타입의 리스트로 배열 생성
a = np.array([[1, 2, 4], [5, 8, 7]], dtype = float)
# 튜플로 배열 생성
b = np.array((1, 3, 2))

# b. 고정된 사이즈의 배열 생성
# 3x4 영배열 생성
c = np.zeros((3, 4))
print(c)
'''
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
'''
# c. 특정 원소로만 이루어진 3x3 행렬 생성
d = np.full((3, 3), 'a', dtype = 'str')
print(d)
'''
[['a' 'a' 'a']
 ['a' 'a' 'a']
 ['a' 'a' 'a']]'''
# 무작위 실수로 이루어진 행렬 생성
e = np.random.random((2, 2))
print(e)
'''
[[0.82865734 0.30178536] 
[0.63319157 0.05515122]]
'''
print(np.random.randint(0, 10, (3, 4)))
print(np.random.normal(0, 1, (4, 4)))
# e. arrange()함수 사용
# 등차수열로 이루어진 배열의 행 생성
f = np.arange(0, -5, -1)
print(f) # [ 0 -1 -2 -3 -4]

# f. linspace() 함수 사용
# 등간격으로 interval을 쪼개는 배열
g = np.linspace(0, 5, 10)
print(g)
'''
[0.         0.55555556 1.11111111 1.66666667 2.22222222 2.77777778 3.33333333 3.88888889 4.44444444 5.        ]'''

# g. 형태 변환
# 3x4 > 2x2x3
arr = np.array([[1, 2, 3, 4], [5, 2, 4, 2], [1, 2, 0, 1]])
newarr = arr.reshape(2, 2, 3)
print(arr)
print(newarr)
'''
[[1 2 3 4]
 [5 2 4 2]
 [1 2 0 1]]
[[[1 2 3]
  [4 5 2]]
  
 [[4 2 1]
  [2 0 1]]]
  '''
# h. 한 행으로 병합
arr = np.array([[1, 2, 3], [4, 5, 6]])
flat_arr = arr.flatten()

print(arr)
print(flat_arr)
'''
[[1 2 3]
 [4 5 6]]
[1 2 3 4 5 6]
'''

# 가로축으로 합치기
a = np.array([1, 2, 3])
b = np. array([4, 5, 6])
c = np.concatenate([a, b])
# 세로축으로 합치기
a = np.arange(4).reshape(1, 4)
b = np.arange(8).reshape(2, 4)
d = np.concatenate([a, b], axis = 0)
print(c)
print(d)
# 배열 나누기
array = np.arange(6).reshape(2, 3)
left, right = np.split(array, [2], axis = 1)
print(left)
print(right)
left, right = np.split(array, [2], axis = 0)
print(left)
print(right)
# 2. Array 인덱싱
'''
슬라이싱
정수 인덱싱
조건 인덱싱
'''

arr = np.array([[-1, 2, 0, 4], [4, -0.5, 6, 0], [2.6, 0, 7, 8], [3, -7, 4, 2.0]])
# 슬라이싱
temp = arr[:2, ::2]
print(temp) # 2행까지, 2칸씩 점프(1열 3열) > 2행 1열3열 출력
arr2 = np.array([[[1, 0, 2], [3, 2, 4], [5, 0, 1]], [[1, 0, 1], [3, 0, 2], [4, 1, 5]]])
temp2 = arr2[:2, :2, ::2]
print(temp2)
# 정수 인덱싱
temp = arr[[0, 1, 2, 3], [3, 2, 1, 0]] #첫번째 차원의 0번째 원소, 두번째 차원의 3번째 원소
print(temp)
temp2 = arr2[[0, 1], [2, 1], [0, 2]]
print(temp2)
# 조건 인덱싱
cond = arr > 0
print(cond)
'''
[[False  True False  True]
 [ True False  True False]
 [ True False  True  True]
 [ True False  True  True]]
'''
temp = arr[cond]
print(temp) # 조건이 참인 원소만 출력



# 수학의 행렬과 array 비교
'''
특징	        NumPy 배열 (ndarray)	        수학적 행렬 (Matrix)
차원  	    다차원 배열 지원           	2차원 행렬만 존재
원소 접근	인덱싱 및 슬라이싱이 자유롭다	수학적 위치로 접근 (i, j)
연산      	원소별 연산 이기본	            수학적 행렬 연산 (행렬곱 등)
행렬 곱셈	np.dot() 또는 @ 사용	        A * B (수학적 행렬 곱셈)
Transpose	.T 또는 transpose() 사용	    전치(transpose)는 수학적 연산
브로드캐스팅	서로 다른 shape도 연산 가능	브로드캐스팅 불가
다차원 처리	고차원 데이터 처리 가능	        2차원만 가능
'''

