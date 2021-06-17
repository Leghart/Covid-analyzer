from sqlalchemy import (create_engine, Column, Integer, String, Sequence,
                        Float, Text, PrimaryKeyConstraint, ForeignKey)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, backref
from sqlalchemy.orm import scoped_session
from sqlalchemy import create_engine
import os
from setup import Country
import types


# Prepare path to database where data will be saved.
db_name = 'Covid_Data.db'
direct_path = os.path.dirname(os.path.abspath(__file__))
path = '\\'.join([direct_path, db_name])
sql = 'sqlite:///'

# Create connection with DB
engine = create_engine(sql + path, echo=False)
db_session = scoped_session(sessionmaker(autocommit=False,
                                         autoflush=False,
                                         bind=engine))
Base = declarative_base()
Base.query = db_session.query_property()


def init_db():
    Base.metadata.create_all(bind=engine)


# A table storing data about all registered pointers.
class MainBase(Base):
    __tablename__ = Country
    date = Column(Text, primary_key=True)
    new_cases = Column(Integer)
    new_deaths = Column(Integer)
    total_cases = Column(Integer)
    total_deaths = Column(Integer)
    total_recovered = Column(Integer)
    active_cases = Column(Integer)
    tot_1M = Column(Float)
    fatality_ratio = Column(Float)
    total_tests = Column(Integer)

    # Insert new pointers into the instance fields, using specially prepared
    # capsule data from scrap.py
    def __init__(self, **kwargs):
        for key, var in kwargs.items():
            self.__dict__[key] = var


# A storing data on all predicted cases of infection and deaths
class PredBase(Base):
    __tablename__ = Country + '_pred'
    date = Column(Text, primary_key=True)
    cases_pred = Column(Integer)
    deaths_pred = Column(Integer)

    # Insert new prediction pointers into the instance fields, using specially
    # prepared capsule data from processing.py
    def __init__(self, **kwargs):
        for key, var in kwargs.items():
            self.__dict__[key] = var


# Retrieves the last record from the database on default invocation. If
# get_date changes to True, get latest date from database (useful when checking
# for the record is already in the database).
def get_last_record(cls, get_date=False):
    last_rec = [(key, val) for key, val in cls.query.all()[-1]
                .__dict__.items()]
    if last_rec[0][0] == '_sa_instance_state':
        del last_rec[0]
    if get_date:
        return [val for key, val in last_rec if key == 'date'][0]
    else:
        return last_rec


# Insert dictionary as kwargs into selected table.
def insert(cls, **kwargs):
    db_session.add(cls(**kwargs))
    db_session.commit()


# Delete record from selected table by given id as date.
def remove(cls, id):
    handler = cls.query.filter_by(date=id).first()
    db_session.delete(handler)
    db_session.commit()


# Get all data from selected table
def get_data(cls):
    data = []
    for row in cls.query.all():
        del row.__dict__['_sa_instance_state']
        data.append(row.__dict__)
    return data


# Add methods to each class
clas = ['PredBase', 'MainBase']
method = ['insert', 'remove', 'get_data', 'get_last_record']
tuples = [(a, b) for a in clas for b in method]

for (cl, meth) in tuples:
    instr = f'{cl}.{meth} = types.MethodType({meth}, {cl})'
    exec(instr)
