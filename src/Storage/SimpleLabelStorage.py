#!/usr/bin/env python3

import logging
logger = logging.getLogger(__name__)

from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import Table
from sqlalchemy import String, Integer, Float
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import select


## Table definitions
class Base(DeclarativeBase):
    pass


class SimpleEvalSplit(Base):
    __tablename__ = "simple_lable_split"
    id = mapped_column(Integer, primary_key=True)
    source_fullpath = mapped_column(String())
    dest_fullpath = mapped_column(String())
    x_min = mapped_column(Integer)
    y_min = mapped_column(Integer)
    x_max = mapped_column(Integer)
    y_max = mapped_column(Integer)

    def __repr__(self):
        return f"Mapping: {self.source_fullpath} -> {self.dest_fullpath}"

## End Table definitions


class SimpleLabelStorage:
    def __init__(self, file):
        self.file = file
        self.engine = create_engine(self.file)
        Base.metadata.create_all(self.engine)

    def store(self, src_file, dest_file, x_min, y_min, x_max, y_max):
        with Session(self.engine) as session:
            detection = SimpleEvalSplit(
                source_fullpath = src_file,
                dest_fullpath   = dest_file,
                x_min = x_min,
                y_min = y_min,
                x_max = x_max,
                y_max = y_max,
            )

            session.add(detection)
            session.commit()

    def getImages(self):
        ret = []
        with Session(self.engine) as session:
            for i in session.execute(select(SimpleEvalSplit)).all():
                ret.append({
                    "source_fullpath" : i[0].source_fullpath,
                    "dest_fullpath" : i[0].dest_fullpath,
                    "x_min" : i[0].x_min,
                    "y_min" : i[0].y_min,
                    "x_max" : i[0].x_max,
                    "y_max" : i[0].y_max
                })

        return ret


    def get_all_detections(self):
        with Session(self.engine) as session:
            return session.execute(select(SimpleEvalSplit)).all()


if __name__ == "__main__":
    storage = SimpleEvalSplit("sqlite:///test.sqlite")
