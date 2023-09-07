#!/usr/bin/env python3
import logging

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


class WeightedDecision(Base):
    __tablename__ = "FinalDetectionStorage"
    id                      = mapped_column(Integer, primary_key=True)
    image_fullpath          = mapped_column(String())
    image_name              = mapped_column(String())
    detection_class_numeric = mapped_column(Integer)
    detection_class_str     = mapped_column(String)
    detection_probability   = mapped_column(Float)
    x_min                   = mapped_column(Integer)
    y_min                   = mapped_column(Integer)
    x_max                   = mapped_column(Integer)
    y_max                   = mapped_column(Integer)

    def __repr__(self):
        return f"Detection of class {self.detection_class_str} in image {self.image_name}"

    def get_box_center_wh(self):
        """
        Returns a box in the format xCenter, yCenter, w, h

        Returns the box in the format (x_center, y_center, w, h)
        -------
        """

        w = round(self.x_max - self.x_min)
        h = round(self.y_max - self.y_min)
        x_center = round(self.x_min + (w/2))
        y_center = round(self.y_min + (h/2))
        return x_center, y_center, w, h

    def get_box_xmin_xmax(self):
        """
        Get the boxes in the format (x_min, x_max, y_min, y_max)
        Returns (x_min, x_max, y_min, y_max)
        -------

        """
        return self.x_min, self.x_max, self.y_min, self.y_max

## End Table definitions


class WeightedDecisionStorage:
    def __init__(self, file):
        self.file = file
        self.engine = create_engine(self.file)
        Base.metadata.create_all(self.engine)

    @staticmethod
    def get_class():
        return WeightedDecision

    def store(self,
              image_fullpath,
              image_name,
              detection_class_numeric,
              detection_class_str,
              detection_probability,
              x_min,
              x_max,
              y_min,
              y_max
              ):
        with Session(self.engine) as session:
            if session.query(WeightedDecision).filter_by(
                    image_fullpath=image_fullpath,
                    image_name=image_name,
                    detection_class_numeric=detection_class_numeric,
                    detection_class_str=detection_class_str,
                    detection_probability=detection_probability,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max
            ).count() == 0:
                weighted_decision = WeightedDecision(
                    image_fullpath=image_fullpath,
                    image_name=image_name,
                    detection_class_numeric=detection_class_numeric,
                    detection_class_str=detection_class_str,
                    detection_probability=detection_probability,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                )
                session.add(weighted_decision)
                session.commit()
            else:
                logger.info("Dropping already existing decision entry.")

    def get_all_images(self):
        with Session(self.engine) as session:
            return session.execute(select(WeightedDecision.image_fullpath).distinct().order_by(WeightedDecision.image_fullpath)).scalars().all()

    def get_detections_for_image(self, image, detection_threshold=0):
        with Session(self.engine) as session:
            return session.execute(select(WeightedDecision).filter(WeightedDecision.image_fullpath==image, WeightedDecision.detection_probability >= detection_threshold)).scalars().all()


if __name__ == "__main__":
    pass
