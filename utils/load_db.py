# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 9/19/2022

from openpyxl import load_workbook


def str2list(face_feature):
    str = face_feature.strip(']')[1:]
    return [float(i.strip('\n')) for i in str.split()]


def load_data(db='src/db.xlsx'):
    wb = load_workbook(db)
    sheet = wb.active
    id_codes = sheet["A"]
    names = sheet['B']
    feats = sheet['C']
    data = []
    for code, name, feat in zip(id_codes, names, feats):
        if code.value is None or not code.value.isdigit():
            continue
        feat = str2list(feat.value)
        info = {'id': code.value, 'name': name.value, 'feat': feat}
        data.append(info)
    return data