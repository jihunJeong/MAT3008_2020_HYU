import numpy as np
import matplotlib.pyplot as plt
import random

def curve_fitting():
	matrix_a = np.array(list1)
	matrix_b = np.array(list2)

	atrans = np.transpose(matrix_a)
	inner = np.dot(atrans, matrix_a)
	pseudoinverse_a = np.dot(np.linalg.inv(inner),atrans)
	sol = np.dot(pseudoinverse_a, matrix_b)
	print("Input X : ", end="")
	print(input_x)
	print("")
	print("a : {}, b : {}, c : {}".format(sol[0][0], sol[1][0], sol[2][0]))

	plt.plot(input_x, input_y,'b')

	x = np.arange(-3, 4, 0.1)
	y = sol[0][0] * (x*x) + sol[1][0] * x + sol[2][0]
	plt.plot(x, y,'r')
	plt.show()

if __name__ == "__main__":
	list1 = [[0.0 for x in range(3)] for y in range(6)]
	list2 = [[0.0] for y in range(6)]
	index = 0

	input_x = []
	input_y = []
	
	random_list = random.sample(range(0, 8), 2)
	input_cnt = 0
	while True:
		x, y = map(float, input().split())
		if x == 0 and y == 0:
			break

		if input_cnt not in random_list:
			list1[index][0] = x*x
			list1[index][1] = x
			list1[index][2] = 1
			list2[index][0] = y

			input_x.append(x)
			input_y.append(y)
			index += 1

		input_cnt += 1

	curve_fitting()
