#!/usr/bin/env python3

import logging

logger = logging.getLogger(__name__)

from sqlalchemy import String, Integer
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import mapped_column
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import select

import json


## Table definitions
class Base(DeclarativeBase):
    pass


class Backmapping(Base):
    __tablename__ = "Backmappings"
    id = mapped_column(Integer, primary_key=True)
    image_fullpath = mapped_column(String())
    image_name = mapped_column(String())
    source = mapped_column(String())
    votings = mapped_column(String())
    x_min = mapped_column(Integer)
    y_min = mapped_column(Integer)
    x_max = mapped_column(Integer)
    y_max = mapped_column(Integer)

    def __repr__(self):
        return f"Mapping for image {self.image_fullpath}"


## End Table definitions


class BackmappingStorage:
    def __init__(self, file):
        self.file = file
        self.engine = create_engine(self.file)
        Base.metadata.create_all(self.engine)

    @staticmethod
    def get_class():
        return Backmapping

    def store(self,
              image_fullpath,
              image_name,
              source,
              votings,
              x_min,
              x_max,
              y_min,
              y_max
              ):
        with Session(self.engine) as session:
            if session.query(Backmapping).filter_by(
                    image_fullpath=image_fullpath,
                    image_name=image_name,
                    source=source,
                    votings=json.dumps(votings),
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max
            ).count() == 0:
                backmapping = Backmapping(
                    image_fullpath=image_fullpath,
                    image_name=image_name,
                    source=source,
                    votings=json.dumps(votings),
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                )
                session.add(backmapping)
                session.commit()
            else:
                logger.info("Dropping already existing backmapping entry.")

    def get_all_images(self):
        with Session(self.engine) as session:
            return session.execute(select(Backmapping.image_fullpath).distinct().order_by(Backmapping.image_fullpath)).scalars().all()

    def get_backmappings_for_image(self, image):
        with Session(self.engine) as session:
            return session.execute(select(Backmapping).filter_by(image_fullpath=image)).scalars().all()

if __name__ == "__main__":
    pass
