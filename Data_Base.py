
from sqlalchemy import (create_engine, Column, Integer, String, Sequence, Float,
                        Text, PrimaryKeyConstraint, ForeignKey)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, backref
from sqlalchemy.orm import scoped_session
from sqlalchemy import create_engine
import os


db_name = 'Covid_Data.db'
direct_path = os.path.dirname(os.path.abspath(__file__))
path='\\'.join([direct_path,db_name])
sql = 'sqlite:///'

Country = 'Poland'

engine = create_engine(sql + path, echo =False)
db_session = scoped_session(sessionmaker(autocommit=False,
                                         autoflush=False,
                                         bind=engine))
Base = declarative_base()
Base.query = db_session.query_property()

def init_db():
    Base.metadata.create_all(bind=engine)


class MainBase(Base):
    __tablename__ = Country
    date = Column(Text, primary_key = True)
    new_cases = Column(Integer )
    new_deaths = Column(Integer )
    total_cases = Column(Integer )
    total_deaths = Column(Integer )
    total_recovered = Column(Integer )
    active_cases = Column(Integer )
    tot_1M = Column(Float )
    fatality_ratio = Column(Float )
    total_tests = Column(Integer)

    def __init__(self, **kwargs):
        for key, var in kwargs.items():
            self.__dict__[key] = var

class PredBase(Base):
    __tablename__ = Country + '_pred'
    date = Column(Text, primary_key = True)
    cases_pred = Column(Integer )
    deaths_pred = Column(Integer )

    def __init__(self, **kwargs):
        for key, var in kwargs.items():
            self.__dict__[key] = var


def get_last_record(cls, get_date = False):
    last_rec = [(key, val) for key,val in cls.query.all()[-1].__dict__.items()]
    if last_rec[0][0] == '_sa_instance_state':
        del last_rec[0]
    if get_date:
        return [val for key, val in last_rec if key=='date'][0]
    else:
        return last_rec

def insert(cls, **kwargs):
    db_session.add(cls(**kwargs))
    db_session.commit()

def delete(cls, id):
    handler = cls.query.filter_by(date = id).first()
    db_session.delete(handler)
    db_session.commit()

def get_data(cls):
    data = []
    for row in cls.query.all():
        del row.__dict__['_sa_instance_state']
        data.append(row.__dict__)
    return data
