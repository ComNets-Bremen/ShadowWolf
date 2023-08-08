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


class Detection(Base):
    __tablename__ = "detected_image"
    id = mapped_column(Integer, primary_key=True)
    image_fullpath = mapped_column(String())
    cut_image =mapped_column(String())
    source_dataclass = mapped_column(String())
    source_datagetter = mapped_column(String())
    detection_class = mapped_column(String)
    detection_class_numeric = mapped_column(Integer)
    confidence = mapped_column(Float)
    x_min = mapped_column(Integer)
    y_min = mapped_column(Integer)
    x_max = mapped_column(Integer)
    y_max = mapped_column(Integer)

    def __repr__(self):
        return f"Detection of class {self.detection_class}"

## End Table definitions

class DetectionStorage:
    def __init__(self, file, input_dataclass=None, input_getter=None):
        self.file = file
        self.engine = create_engine(self.file)
        Base.metadata.create_all(self.engine)
        self.input_dataclass = input_dataclass
        self.input_getter = input_getter

    def store(self, fullpath, cut_image, detection_class, detection_class_numeric, confidence, x_min, y_min, x_max, y_max):
        with Session(self.engine) as session:
            detection = Detection(
                image_fullpath = fullpath,
                cut_image = cut_image,
                source_dataclass = self.input_dataclass,
                source_datagetter = self.input_getter,
                detection_class = detection_class,
                detection_class_numeric = detection_class_numeric,
                confidence = confidence,
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
            for i in session.execute(select(Detection)).all():
                ret.append({
                    "image_fullpath" : i[0].image_fullpath,
                    "cut_image" : i[0].cut_image,
                    "confidence" : i[0].confidence,
                    "detection_class" : i[0].detection_class,
                    "detection_class_numeric" : i[0].detection_class_numeric,
                    "x_min" : i[0].x_min,
                    "y_min" : i[0].y_min,
                    "x_max" : i[0].x_max,
                    "y_max" : i[0].y_max
                })

        return ret

    def getCutImages(self):
        with Session(self.engine) as session:
            return session.execute(select(Detection.cut_image)).scalars().all()


    def get_all_detections(self):
        with Session(self.engine) as session:
            return session.execute(select(Detection)).all()


if __name__ == "__main__":
    storage = DetectionStorage("sqlite:///test.sqlite")
    print(storage.get_all_detections())
