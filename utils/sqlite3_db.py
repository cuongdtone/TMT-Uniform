# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 9/21/2022

import sqlite3


def connect_database():
    conn = sqlite3.connect('src/db.db')
    return conn


def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        if col[0] == "face_embed":
            d[col[0]] = str2list(row[idx])
            continue
        d[col[0]] = row[idx]
    return d


def execute(db, query, values=None):
    try:
        if db == None:
            db = connect_database()
        cur = db.cursor()
        cur.execute(query, values)
        db.commit()

        cur.close()
        return True
    except Exception as ex:
        print(ex)
        db.rollback()
        return False


def insert_employee(db, code: str, fullname: str, feat):
    if db is None:
        db = connect_database()
    query = f"""INSERT INTO employees ( id, fullname, face_embed) VALUES (?, ?, ?);"""
    return execute(db, query, (code, fullname, feat))


def insert_check(db, code, fullname, card, shift, image):
    if db is None:
        db = connect_database()
    query = f"""INSERT INTO check_uniform ( id, full_name, id_card, shirt, image) VALUES (?, ?, ?, ?, ?);"""
    return execute(db, query, (code, fullname, card, shift, image))


def load_data(db=None):
    try:
        query = "SELECT id, fullname, face_embed  FROM employees;"
        if db is None:
            db = connect_database()
        cur = db.cursor()
        result = cur.execute(query)
        result_data = result.fetchall()
        employee = []
        for row in result_data:
            employee.append(dict_factory(cur, row))
        cur.close()
        return list(employee)
    except Exception as ex:
        print(ex)
        return None


def str2list(face_feature):
    str = face_feature.strip(']')[1:]
    return [float(i.strip('\n')) for i in str.split()]
