def origin(num):
    cal = 0
    for i in range(len(function)):
    	cal += num**(i) * function[len(function)-i-1]
    return cal

def diff1(num):
	return (origin(num+0.0001)-origin(num))/0.0001

def diff2(num):
	return (origin(num+0.0001) -2*origin(num)+origin(num-0.0001))/(0.0001**2)
	
def approximation(num):
	old_x = num
	while True:
		new_x = old_x - (diff1(old_x) / diff2(old_x))

		if abs((new_x - old_x) / new_x) < 0.0001:
			print(new_x)
			break

		old_x = new_x

if __name__ == "__main__":
	function = list(map(float, input().split())) #함수의 계수를 공백 기준으로 입력 받아 다양한 함수 적용 가능하게 함 
	num = float(input())
	approximation(num)
