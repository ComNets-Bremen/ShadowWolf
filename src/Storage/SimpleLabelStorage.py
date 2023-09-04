#!/usr/bin/env python3
import json
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
    __tablename__ = "simple_label_split"
    id = mapped_column(Integer, primary_key=True)
    source_class = mapped_column(String())
    source_getter = mapped_column(String())
    source_fullpath = mapped_column(String())
    dest_fullpath = mapped_column(String())
    relative_votings = mapped_column(String())

    def __repr__(self):
        return f"Mapping: {self.source_fullpath} -> {self.dest_fullpath}"

## End Table definitions


class SimpleLabelStorage:
    def __init__(self, file):
        self.file = file
        self.engine = create_engine(self.file)
        Base.metadata.create_all(self.engine)

    def store(self, source_class, source_getter, src_file, dest_file):
        with Session(self.engine) as session:
            detection = SimpleEvalSplit(
                source_class = source_class,
                source_getter = source_getter,
                source_fullpath = src_file,
                dest_fullpath   = dest_file,
            )

            session.add(detection)
            session.commit()

    def get_images(self):
        ret = []
        with Session(self.engine) as session:
            for i in session.execute(select(SimpleEvalSplit)).all():
                ret.append({
                    "source_fullpath" : i[0].source_fullpath,
                    "dest_fullpath" : i[0].dest_fullpath,
                    "source_class" : i[0].source_class,
                    "source_getter" : i[0].source_getter,
                    "votings" : i[0].relative_votings,
                })

        return ret

    def set_relative_voting(self, image_name, votings):
        if type(votings) is not str:
            votings = json.dumps(votings)
        with Session(self.engine) as session:
            dataset = session.execute(select(SimpleEvalSplit).filter_by(dest_fullpath=image_name)).scalar_one()
            dataset.relative_votings = votings
            session.commit()

    def get_relative_voting(self, image_name):
        with Session(self.engine) as session:
            return json.loads(session.execute(select(SimpleEvalSplit).filter_by(dest_fullpath=image_name)).scalar_one().relative_votings)

    def get_all_detections(self):
        with Session(self.engine) as session:
            return session.execute(select(SimpleEvalSplit)).all()


if __name__ == "__main__":
    storage = SimpleEvalSplit("sqlite:///test.sqlite")
