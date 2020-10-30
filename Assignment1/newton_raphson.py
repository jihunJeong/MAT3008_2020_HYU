def origin_calculate(num):
    cal = 0
    for i in range(len(function)):
    	cal += num**(i) * function[len(function)-i-1]
    return cal

def diff_calculate(num):
	cal = 0
	for i in range(len(function)-1):
		cal += num**(i) * function[len(function)-i-2] * (i+1)
	return cal

def newton_raphson(num):
	old_x = num
	while True:
		new_x = old_x - (origin_calculate(old_x) / diff_calculate(old_x))

		if abs((new_x - old_x) / new_x) < 0.0001:
			print(new_x)
			break

		old_x = new_x

if __name__ == "__main__":
	function = list(map(float, input().split())) #함수의 계수를 공백 기준으로 입력 받아 다양한 함수 적용 가능하게 함 
	num = float(input())
	newton_raphson(num)
