# _*_ coding:utf-8 _*_
from autocorrect import spell


# 检查行尾是否为正确单词
def spell_check(word):
	c = spell(word)

	return c


if __name__ == '__main__':
	print(spell_check('StijUdha.'))

