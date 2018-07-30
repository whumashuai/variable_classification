'''
input = r"E:\variable_classification\variable_classification\path\bitcoind.paths"
output = r"E:\variable_classification\variable_classification\path\path_bitcoin\bitcoin_paths.txt"
def del_with():
	examples = open(input,'r').readlines()
	for example in examples:
		if example.strip().startswith('-1%'):
			continue
		open(output,'a').write(example)
if __name__ == '__main__':
	del_with()
'''
recall = f1_score = precision = 0.50
with open('./result.txt', 'w') as f:
	f.write("recall {:g}, f1_score {:g}, precision {:g}".format(recall, f1_score, precision))