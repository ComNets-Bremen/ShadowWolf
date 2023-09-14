#!/usr/bin/env python3

import logging

logger = logging.getLogger(__name__)

from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import Table
from sqlalchemy import String, Integer
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import select


## Table definitions
class Base(DeclarativeBase):
    pass


image_to_image = Table(
    "image_to_image",
    Base.metadata,
    Column("left_image_id", Integer, ForeignKey("duplicate_image.id"), primary_key=True),
    Column("right_image_id", Integer, ForeignKey("duplicate_image.id"), primary_key=True),
)


class DuplicateImage(Base):
    __tablename__ = "duplicate_image"
    id = mapped_column(Integer, primary_key=True)
    fullpath = mapped_column(String())
    dataclass = mapped_column(String())
    getter = mapped_column(String())
    right_images = relationship(
        "DuplicateImage",
        secondary=image_to_image,
        primaryjoin=id == image_to_image.c.left_image_id,
        secondaryjoin=id == image_to_image.c.right_image_id,
        backref="left_images",
    )

    def __repr__(self):
        return f"Image:  {self.fullpath}"


## End Table definitions


class DuplicateImageStorage:
    def __init__(self, file):
        self.file = file
        self.engine = create_engine(self.file)
        Base.metadata.create_all(self.engine)

    @staticmethod
    def get_class():
        return DuplicateImage

    def get_image(self, fullpath):
        with Session(self.engine) as session:
            return session.execute(select(DuplicateImage).where(DuplicateImage.fullpath == fullpath)).one_or_none()

    def add_duplicate(self, original_image, duplicate):
        with Session(self.engine) as session:
            orig_image = session.execute(
                select(DuplicateImage).where(DuplicateImage.fullpath == original_image[0])).one_or_none()
            if orig_image is None:
                session.add(DuplicateImage(
                    fullpath=original_image[0],
                    dataclass=original_image[1],
                    getter=original_image[2],
                ))
                session.commit()
                orig_image = session.execute(
                    select(DuplicateImage).where(DuplicateImage.fullpath == original_image[0])).one_or_none()

            dup_image = session.execute(
                select(DuplicateImage).where(DuplicateImage.fullpath == duplicate[0])).one_or_none()
            if dup_image is None:
                session.add(DuplicateImage(
                    fullpath=duplicate[0],
                    dataclass=duplicate[1],
                    getter=duplicate[2],
                ))
                session.commit()
                dup_image = session.execute(
                    select(DuplicateImage).where(DuplicateImage.fullpath == duplicate[0])).one_or_none()

            orig_image = orig_image[0]
            dup_image = dup_image[0]
            if dup_image not in orig_image.left_images:
                orig_image.left_images.append(dup_image)
            session.add(orig_image)
            session.commit()

    def get_similar(self, fullname):
        with Session(self.engine) as session:
            img = session.execute(select(DuplicateImage).where(DuplicateImage.fullpath == fullname)).one_or_none()
            if img is None:
                return None
            similar_images = [i for i in img[0].right_images]
            similar_images.extend([i for i in img[0].left_images])
            return list(set(similar_images))

    def get_main_similar(self, fullname):
        img = self.get_image(fullname)
        similar_images = self.get_similar(fullname)
        if similar_images is None:
            return fullname
        if img is not None:
            similar_images.append(img[0])
        similar_images.sort(key=lambda db_entry: db_entry.id)
        return similar_images[0]

    def get_all_images(self):
        with Session(self.engine) as session:
            return session.execute(select(DuplicateImage)).all()


if __name__ == "__main__":
    pass
