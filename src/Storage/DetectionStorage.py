#!/usr/bin/env python3
import json
import logging

from Storage.DataStorage import Image, SegmentDataStorage, BaseStorage
from wolf_utils.misc import getter_factory
from wolf_utils.types import ReturnDetectionDict

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
    def get_fullscale_voting(image: Detection, sqlitefile: str) -> ReturnDetectionDict:
        src_image = getter_factory(image.source_class, image.source_getter, sqlitefile)(image.image_fullpath)
        if len(src_image) != 1:
            raise ValueError(f"Got wrong number of images. Should be one, got {len(src_image)}")
        src_image = src_image[0]

        if isinstance(src_image, Image):
            # Reached base image
            votings = BaseStorage.get_fullscale_voting(src_image, sqlitefile)

            votings["x_min"] = image.x_min
            votings["x_max"] = image.x_max
            votings["y_min"] = image.y_min
            votings["y_max"] = image.y_max
            votings["votings"].append(
                # "Detection is used for the weights later on
                ("Detection", {str(image.detection_class_numeric): image.confidence})
            )
            logger.info(f"votings: {votings}")
            return votings
        elif isinstance(src_image, SegmentDataStorage.get_class()):
            votings = SegmentDataStorage.get_fullscale_voting(src_image, sqlitefile)
            print(votings)
            if votings["x_min"] is None:
                votings["x_min"] = image.x_min
                votings["x_max"] = image.x_max
                votings["y_min"] = image.y_min
                votings["y_max"] = image.y_max
            else:
                votings["x_max"] = votings["x_min"] + image.x_max
                votings["x_min"] = votings["x_min"] + image.x_min
                votings["y_max"] = votings["y_min"] + image.y_max
                votings["y_min"] = votings["y_min"] + image.y_min


            votings["votings"].append(
                # "Detection is used for the weights later on
                ("Detection", {str(image.detection_class_numeric): image.confidence})
            )
            logger.info(f"votings: {votings}")
            return votings
        else:
            raise ValueError(f"Type not implemented: {type(src_image)}")



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
