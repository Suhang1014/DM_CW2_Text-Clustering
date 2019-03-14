# _*_ coding:utf-8 _*_
from bs4 import BeautifulSoup
import spellChecker


# GET SPAN TEXT
def extract_text(doc_text):
    text = ''
    soup = BeautifulSoup(doc_text, 'lxml')
    try:
        for script in soup(["script", "style"]):
            script.extract()
    except Exception as error:
        print(error)
        pass
    else:
        try:
            # get text
            title = soup.title.string
        except Exception as error:
            print(error)
            pass
        temp1 = ''
        temp2 = ''
        isInsert1 = False  # 判断是否要删除句尾连字符-
        isInsert2 = False  # 判断行尾是否是一个完整单词（只能判断一部分）
        text = ''
        line_spans = soup.find_all('span', {'class': 'ocr_line'})
        for line_span in line_spans:
            word_list = [word_spans.get_text() for word_spans in line_span.find_all('span')]
            check = word_list[-1][-1]
            if isInsert1:
                word_list[0] = temp1 + word_list[0]
                temp1 = ''
            if check == '-' or check == '~':
                temp1 = word_list.pop(-1)[0:-1]
                isInsert1 = True
            if word_list:
                correct_word = spellChecker.spell_check(word_list[-1])
                if isInsert2:
                    word_list[0] = temp2 + word_list[0]
                if type(word_list[-1]) == 'str':
                    if word_list[-1] != correct_word:
                        temp2 = word_list.pop(-1)
                        isInsert2 = True
            word_list.append(' ')
            text += ' '.join(word_list)

    return text


if __name__ == '__main__':
    test_path = '/Users/suhang/Documents/GitHub/COMP6237-Data-Mining/cw2-understanding-data/gap-html/gap_-C0BAAAAQAAJ/' \
            '00000136.html'
    with open(test_path, 'r', encoding='utf-8') as f:
        test_text = extract_text(f)
        print(test_text)



