#!/usr/bin/env python3

import logging

from Storage.DataStorage import BaseStorage

logger = logging.getLogger(__name__)

from sqlalchemy import String, Integer, Float
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import mapped_column
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import select


## Table definitions
class Base(DeclarativeBase):
    pass


class Detection(Base):
    __tablename__ = "detected_image"
    id = mapped_column(Integer, primary_key=True)
    image_fullpath = mapped_column(String()) # The original image in the fullpath format
    cut_image = mapped_column(String())
    source_class = mapped_column(String())
    source_getter = mapped_column(String())
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
    def __init__(self, file, source_class=None, source_getter=None):
        self.file = file
        self.engine = create_engine(self.file)
        Base.metadata.create_all(self.engine)
        self.source_class = source_class
        self.source_getter = source_getter

    @staticmethod
    def get_class():
        return Detection

    @staticmethod
    def convert_to_voting_dict(image: Detection, bs: BaseStorage) -> dict:
        orig_image = bs.get_image_by_fullpath(image.image_fullpath)

        return {
                        "orig_image_name": orig_image.name,
                        "orig_image_fullpath": orig_image.fullpath,
                        "x_max": image.x_max,
                        "x_min": image.x_min,
                        "y_max": image.y_max,
                        "y_min": image.y_min,
                        "votings": [DetectionStorage.get_class().__name__, {str(image.detection_class_numeric): image.confidence}],
                        "source": DetectionStorage.get_class().__name__,
                    }


    def store(self, fullpath, cut_image, detection_class, detection_class_numeric, confidence, x_min, y_min, x_max,
              y_max):
        with Session(self.engine) as session:
            detection = Detection(
                image_fullpath=fullpath,
                cut_image=cut_image,
                source_class=self.source_class,
                source_getter=self.source_getter,
                detection_class=detection_class,
                detection_class_numeric=detection_class_numeric,
                confidence=confidence,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
            )

            session.add(detection)
            session.commit()

    def get_images(self, img=None):
        with Session(self.engine) as session:
            if img is None:
                q = session.execute(select(Detection)).scalars().all()
            else:
                q = session.execute(select(Detection).filter_by(cut_image=img)).scalars().all()
        return q

    def get_cut_images(self, img=None):
        if img is None:
            with Session(self.engine) as session:
                return session.execute(select(Detection.cut_image)).scalars().all()
        else:
            return self.get_images(img)

    def get_all_detections(self):
        with Session(self.engine) as session:
            return session.execute(select(Detection)).all()

    def get_class_name_by_id(self, class_id):
        with Session(self.engine) as session:
            return session.execute(select(Detection.detection_class).filter_by(detection_class_numeric=class_id).distinct()).scalar_one_or_none()


if __name__ == "__main__":
    storage = DetectionStorage("sqlite:///test.sqlite")
    print(storage.get_all_detections())
