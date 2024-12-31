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

# NumPy 기본 연산
# 1. 1차원 배열 함수 연산
a = np.array([1, 2, 5, 3])
# 모든 원소에 1 더하기
print(a + 1)
# 모든 원소에 3 빼기
print(a - 3)
# 모든 원소에 10 곱하기
print(a * 10)
# 모든 원소 제곱
print(a ** 2)
# transpose
print(a.T) #변화가 없는 걸 보니 수학의 행렬과 numpy의 배열은 약간의 차이가 있는 것 같다.
arr = np.arange(24).reshape(2, 3, 4)
print(arr)

# 축 재배열
transposed = arr.transpose()
print(transposed)

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

# 2. 단항 연산
arr = np.array([[1, 5, 6],
                [4, 7, 2],
                [3, 1, 9]])
# 배열 중 가장 큰 원소
print(arr.max()) # default: axis = None -> 전체 배열에서 최대값 찾음
print(arr.max(axis = 1)) # 축 1 행 내부 비교
# 배열 중 가장 작은 원소
print(arr.min())
print(arr.min(axis = 0)) # 축 0 열 내부 비교
# 원소들의 합
print(arr.sum())
# 누적 합
print(arr.cumsum(axis = 0))

# 3. 이항연산
a = np.array([[1, 2],
              [3, 4]])
b = np.array([[4, 3],
              [2, 1]])
print(a + b) # 원소별 덧셈
print(a * b) # 원소별 곱셈
print(a @ b) # 행렬 곱셈
print(a.dot(b)) # 내적

# 4. 비트연산
a = np.array([2, 8, 125])
b = np.array([3, 3, 115])
print(np.bitwise_and(a, b)) # 비트가 둘 다 1이면 1 a = [10, 1000, 11111100) b = [11, 0011, 11110010]
print(np.bitwise_or(a, b)) # 비트가 하나라도 1이면 1 a and b = [10, 0000, 11110000] = [2, 0, 113] a or b = [10, 1011, 111111110] = [3, 11, 127]
# XOR: 두 비트가 달라야 1
# Invert: 비트 전환
print(np.invert(a)) #invert(a) = [11111110, 11111000, 00000100] = [-3, -9, -126]

# Bit shift
a = [2, 8, 15] # [00010, 00001000, 0000001111]
bit_shift = [3, 4, 5]
output = np.left_shift(a, bit_shift)
print(output) # [10000, 10000000, 1111000000] = [16, 128 ,480]
print(np.right_shift(a, bit_shift)) # [0, 0, 0]

# 이진수
in_num = [10, 16]
out_num = np.binary_repr(in_num[0])
print(out_num)
print(np.binary_repr(in_num[1], width = 5)) # 비트수 5로 설정

# 이진수 패킹
a = np.array([[[1, 0, 1],
            [0, 1, 0]],
            [[1, 1, 0],
            [0, 0, 1]]])
for i in range(3):
    print(np.packbits(a, axis = i))

print(np.packbits(a, axis = 0))