import sqlite3
import pandas as pd
import os


class DataBase:

    def __init__(self, country):
        db_name = 'Covid_Data.db'
        direct_path = os.getcwd()
        path='\\'.join([direct_path,db_name])

        self.con = sqlite3.connect(path)
        self.cursor = self.con.cursor()
        self.country = country

        self.cursor.execute(
                """SELECT count(name) FROM sqlite_master WHERE
                type='table' AND name='""" + self.country + "' ")

        if self.cursor.fetchone()[0] != 1:
            self.cursor.execute("CREATE TABLE " + self.country +
                                """(date TEXT, new_cases INT,
                                total_cases INT, new_deaths INT,
                                total_deaths INT, total_recovered INT,
                                active_cases INT, tot_1M REAL,
                                fatality_ratio REAL, total_tests INT)""")

        else:
            print("successful database connection.\n")

    def __del__(self):
        self.cursor.close()
        self.con.close()

    def commit(self, S):
        try:
            self.con.commit()
            print(f"Data was successfully committed ({S.date}).\n")
        except Exception:
            print("Data coudn't be commit.\n")

    def get_last_record_date(self):
        self.cursor.execute("SELECT date FROM " + self.country)
        try:
            last_date = self.cursor.fetchall()[-1][0]
        except Exception:
            last_date = None
        return last_date

    def insert(self, S):
        if self.get_last_record_date() != S.date:
            self.cursor.execute("INSERT INTO " + self.country +
                                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                                (S.date, S.new_cases, S.total_cases,
                                 S.new_deaths, S.total_deaths, S.total_rec,
                                 S.active_cases, S.tot, S.fatality_ratio,
                                 S.total_tests))
            #file.write('{} {} {} {} {} {} {} {} {} {} {} \n'.format(
            #    self.country, S.date, S.new_cases, S.total_cases,
            #    S.new_deaths, S.total_deaths, S.total_rec, S.active_cases,
            #    S.tot, S.fatality_ratio, S.total_tests))
            self.commit(S)
        else:
            print(f"That record is already in data base ({S.date}).\n")

    def update(self, S):
        if self.get_last_record_date() == S.date:
            sql_query = "UPDATE " + self.country + """ SET new_cases = ?,
                        total_cases = ?, new_deaths = ?, total_deaths = ?,
                        total_recovered = ?, active_cases = ?, tot_1M = ?,
                        fatality_ratio = ?, total_tests = ?
                        WHERE date = ? """
            data = [S.new_cases, S.total_cases, S.new_deaths, S.total_deaths,
                    S.total_rec, S.active_cases, S.tot, S.fatality_ratio,
                    S.total_tests, S.date]
            try:
                self.cursor.execute(sql_query, data)
                self.commit(S)
                print("Update executed\n")
            except sqlite3.Error as e:
                print(f'Something was wrong!. Error num: {e}\n')

    # def remove_record(self,day):
