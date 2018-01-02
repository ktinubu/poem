import poetry_gen

if __name__ == '__main__':
	gen = poetry_gen.FullModel()
	print(gen.predict("poop"))