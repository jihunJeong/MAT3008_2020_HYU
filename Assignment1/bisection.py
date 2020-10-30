def calculate(num):
    cal = 0
    for i in range(len(function)):
    	cal += num**(i) * function[len(function)-i-1]
    return cal

def bracket(num1, num2):
	if calculate(num1) * calculate(num2) < 0:
		return True
	else :
		return False


def bisection(num1, num2):
	left, right = num1, num2
	old_mid = left
	while True: 
		new_mid = (left + right) / 2
		if calculate(new_mid) * calculate(left) > 0:
			left = new_mid
		else:
			right = new_mid

		if abs((new_mid - old_mid) / new_mid) < 0.0001:
			print(new_mid)
			break
			
		old_mid = new_mid
    

if __name__ == "__main__":
	function = list(map(float, input().split())) #함수의 계수를 공백 기준으로 입력 받아 다양한 함수 적용 가능하게 함 
	
	left, right = -10.0, -9.999
	
	while right < 10:
	    if bracket(left, right):
	    	bisection(left, right)

	    left += 0.001
	    right += 0.001
