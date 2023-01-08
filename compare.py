import ast, re, pathlib, shutil, pathlib
import numpy as np
import argparse

from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def remove_comments(source):  # Ф-ция удаляющая комментарии
    string = re.sub(re.compile("'''.*?'''", re.DOTALL), "", source)
    string = re.sub(re.compile('""".*?"""', re.DOTALL), "", source)
    return string


def make_clean_ast(pyfile_path1, pyfile_path2):
    """
    Аргументы: Абсолютный путь до первого и второго Python кода соотвественно
    Возвращает: Отформатированное абстрактное синтаксическое дерево, для дальнейшего сравнения
    """
    signs = [']', '[', '(', ')', ',', "'"]
    nwn1 = pyfile_path1[:-3] + 'copy1' + '.py'
    nwn2 = pyfile_path2[:-3] + 'copy2' + '.py'

    shutil.copyfile(pyfile_path1, nwn1)
    shutil.copyfile(pyfile_path2, nwn2)

    nwn1 = pathlib.Path(nwn1)
    nwn2 = pathlib.Path(nwn2)

    nwp = str([part + '/' for part in list(str(nwn1).split('/')[:-1])]).replace(']', '').replace('[',
                                                                                                           '').replace(
        ',', '').replace("'", '').replace(' ', '')
    nwn1.rename(nwp + 'copy1' + '.txt')
    copy1 = nwp + 'copy1' + '.txt'
    nwp = str([part + '/' for part in list(str(nwn2).split('/')[:-1])]).replace(']', '').replace('[',
                                                                                                           '').replace(
        ',', '').replace("'", '').replace(' ', '')
    nwn2.rename(nwp + 'copy2' + '.txt')
    copy2 = nwp + 'copy2' + '.txt'


    f = open(copy1)
    code1 = ""
    for _ in f:
        code1 += _
    code1 = remove_comments(code1)  # Удаление комментариев
    tree1 = ast.parse(source=code1)  # Построение абстрактного синтаксического дерева с помощью библиотеки AST
    ast_code1 = ast.dump(tree1,
                         annotate_fields=False) 
    for _ in signs:
        ast_code1 = ast_code1.replace(_, ' ')
    ast_code1 = re.sub(" +", " ", ast_code1)
    f = open(copy2)
    code2 = ""
    for _ in f:
        code2 += _
    code2 = remove_comments(code2)
    tree2 = ast.parse(source=code2)
    ast_code2 = ast.dump(tree2, annotate_fields=False)
    for _ in signs:
        ast_code2 = ast_code2.replace(_, ' ')
    ast_code2 = re.sub(" +", " ", ast_code2)

    return ast_code1, ast_code2

# Консольный ввод
parser = argparse.ArgumentParser()
parser.add_argument('indir', type=str)
parser.add_argument('outdir', type=str)
args = parser.parse_args()
input_path = args.indir
output_path = args.outdir
# Открытие файлов для записи и чтения
source = open(input_path, 'r')
result = open(output_path, 'w')

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
