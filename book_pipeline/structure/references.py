"""Helpers used by the current PDF Reforge workflow."""
import re

LABELS = {'公式':'equation','式':'equation','图':'figure','表':'table','算法':'algorithm',
          '例题':'example','例':'example','定理':'theorem','引理':'lemma','推论':'corollary',
          '命题':'proposition','定义':'definition','习题':'exercise','练习':'exercise','记号':'notation'}


def normalized(number):
    value = re.sub(r'\s+','',str(number)).strip('[]()（）').replace('．','.').replace('′',"'").replace('’',"'")
    if value.startswith('附录'):return '附录'+chinese_number(value[2:])
    return chinese_number(value) if re.fullmatch(r'[零〇一二三四五六七八九十百]+', value) else value


def chinese_number(text):
    digits={'零':0,'〇':0,'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
    total,current=0,0
    for c in text:
        if c in digits:
            current=digits[c]
        elif c in {'十','百'}:
            total+=(current or 1)*(10 if c=='十' else 100);current=0
        else:
            return text
    return str(total+current)
