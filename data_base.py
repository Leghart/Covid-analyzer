import types  # noqa: F401

from sqlalchemy import Column, Float, Integer, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker

from settings import COUNTRY, DB_NAME, MAIN_PATH, OS_CON

# Prepare path to database where data will be saved.
path = OS_CON.join([MAIN_PATH, DB_NAME])

sql = "sqlite:///"

# Create connection with DB
engine = create_engine(sql + path, echo=False)
db_session = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=engine)
)
Base = declarative_base()
Base.query = db_session.query_property()


def init_db():
    Base.metadata.create_all(bind=engine)


class MainBase(Base):
    """A table storing data about all registered pointers."""

    __tablename__ = COUNTRY
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

    def __init__(self, **kwargs):
        """
        Insert new pointers into the instance fields, using
        specially prepared capsule data from scrap.py.

        Parameters:
        - kwargs (dict) - capsule of data containing data for each columns
        """
        for key, var in kwargs.items():
            self.__dict__[key] = var


class PredBase(Base):
    """A storing data on all predicted cases of infection and deaths."""

    __tablename__ = COUNTRY + "_pred"
    date = Column(Text, primary_key=True)
    cases_pred = Column(Integer)
    deaths_pred = Column(Integer)

    def __init__(self, **kwargs):
        """
        Insert new prediction pointers into the instance fields, using
        specially prepared capsule data from processing.py.

        Parameters:
        - kwargs (dict) - capsule of data containing data for each columns
        """
        for key, var in kwargs.items():
            self.__dict__[key] = var


def get_last_record(cls, get_date=False):
    """
    Retrieves the last record from the database on default invocation. If
    get_date changes to True, get latest date from database
    (useful when checking for the record is already in the database).

    Parameters:
    - get_date (bool) - flag to get only a date from last record

    Returns:
    - (list) - if get_date statement is True, returnin only last date, if
    get_date is False, returning last row from database
    """
    last_rec = [(key, val) for key, val in cls.query.all()[-1].__dict__.items()]
    if last_rec[0][0] == "_sa_instance_state":
        del last_rec[0]
    if get_date:
        return [val for key, val in last_rec if key == "date"][0]
    else:
        return last_rec


def insert(cls, **kwargs):
    """
    Insert dictionary as kwargs into selected table.

    Parameters:
    - kwargs (dict) - capsule of data to insert as a new row in database
    """
    db_session.add(cls(**kwargs))
    db_session.commit()


def remove(cls, id):
    """
    Delete record from selected table by given id as date.

    Parameters:
    - id (string) - ID of record to delete from database
    """
    handler = cls.query.filter_by(date=id).first()
    db_session.delete(handler)
    db_session.commit()


def get_data(cls):
    """
    Get all data from selected table.

    Returns:
    - data (list) - list where each record is a dictionary. Containing all data
    from database
    """
    data = []
    for row in cls.query.all():
        del row.__dict__["_sa_instance_state"]
        data.append(row.__dict__)
    return data


# Add methods to each class
clas = ["PredBase", "MainBase"]
method = ["insert", "remove", "get_data", "get_last_record"]
tuples = [(a, b) for a in clas for b in method]

for (cl, meth) in tuples:
    instr = f"{cl}.{meth} = types.MethodType({meth}, {cl})"
    exec(instr)
