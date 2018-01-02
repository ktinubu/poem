# cleanPoetry.txt
# Khaled Tinubu
# This file perfors simple cleaning of for a particalar text file for use as
# a source of data
# khaledtinubu@gmail.com

# All Python resources by 

#
def containsSlash(word):
	for letter in word:
		if letter == '/':
			return True
	return False

def cleanLine(input):
	if len(input) > 80:
		return ""
	if containsSlash(input):
		return ""
	input = input.replace('*', '')
	input = input.replace('-', '')
	input = input.replace('_', '')
	nput = input.replace('~', '')
	return input.strip() + '\n'



if __name__ == '__main__':
	new_filename = "redditPoems.txt"
	f = open("preCleanRedditPoems.txt")
	txt_lines = f.readlines()

	new_lines = [cleanLine(line) for line in txt_lines]
	new_file = open(new_filename, "w")
	new_file.writelines(new_lines)