# _*_ coding:utf-8 _*_
import os
import extractText


def main():
	rootdir = '/Users/suhang/Documents/GitHub/COMP6237-Data-Mining/cw2-understanding-data/gap-html'
	texts_list = []
	folders = os.listdir(rootdir)
	folders.sort()
	print(folders)
	# 依次打开文件并读取解析，放入list中
	for folder in folders:
		if folder != '.DS_Store':
			folder_path = os.path.join(rootdir, folder)
			print(folder_path)
			files = os.listdir(folder_path)
			files.sort()
			texts = []
			for file in files:
				if file != '.ipynb_checkpoints':
					file_path = os.path.join(folder_path, file)
					print(file_path)
					if os.path.isfile(file_path):
						with open(file_path, 'r', encoding='utf-8') as f:
							this_text = extractText.extract_text(f)
							texts.append(this_text)
			texts_list.append(texts)
	# 写入txt文件
	for i in range(0, len(texts_list)):
		raw_text = ''
		for j in range(0, len(texts_list[i])):
			raw_text += texts_list[i][j]
		file_name = '/Users/suhang/Documents/GitHub/COMP6237-Data-Mining/cw2-understanding-data/raw_text_2/' + folders[
			i+1] + '.txt'
		with open(file_name, 'w') as f:
			f.write(raw_text)


if __name__ == '__main__':
	main()

