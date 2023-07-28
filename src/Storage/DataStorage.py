#!/usr/bin/env python3

import logging
logger = logging.getLogger(__name__)

import os
import cv2

from typing import List
from typing import Optional
from sqlalchemy import ForeignKey
from sqlalchemy import String, Boolean, Integer
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import select

## Table definitions
class Base(DeclarativeBase):
    pass

class Image(Base):
    __tablename__ = "image"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String())
    fullpath: Mapped[str] = mapped_column(String())

    width: Mapped[Optional[int]]  = mapped_column(Integer())
    height: Mapped[Optional[int]] = mapped_column(Integer())
    colors: Mapped[Optional[int]] = mapped_column(Integer())

    batch: Mapped["ImageBatch"] = relationship(back_populates="images")
    batch_id: Mapped[Optional[int]] = mapped_column(ForeignKey("image_batch.id"))

    segments = relationship("Segment", lazy="immediate")

    def __repr__(self):
        return f"Image:  {self.name}"


class ImageMetadata(Base):
    __tablename__ = "image_meta"
    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped["Image"] = mapped_column(ForeignKey("image.id"))
    imageIsGray: Mapped[bool] = mapped_column(Boolean())
    exifs: Mapped[Optional[List["Exif"]]] = relationship(back_populates="image_meta", lazy="immediate")

    def __repr__(self) -> str:
        return f"ImageMetadata({self.image_id})"


class Exif(Base):
    __tablename__ = "image_exif"

    id: Mapped[int] = mapped_column(primary_key=True)
    image_id : Mapped["Image"] = mapped_column(ForeignKey("image.id"))
    image_meta_id : Mapped[int] = mapped_column(ForeignKey("image_meta.id"))
    image_meta: Mapped["ImageMetadata"] = relationship(back_populates="exifs")
    exif_code: Mapped[int] = mapped_column(Integer())
    exif_name: Mapped[str] = mapped_column(String())
    exif_value: Mapped[str] = mapped_column(String())

    def __repr__(self) -> str:
        return f"Exif dataset: {self.exif_name} -> {self.exif_value}"



class ImageBatch(Base):
    __tablename__ = "image_batch"
    id: Mapped[int] = mapped_column(primary_key=True)
    images: Mapped[List["Image"]] = relationship(back_populates="batch", cascade="all, delete-orphan", lazy="immediate")
    creator: Mapped[str|None] = mapped_column(String())

    def __repr__(self):
        return f"Batch with {len(self.images)} images"


class Segment(Base):
    __tablename__ = "image_segments"
    id: Mapped[int] = mapped_column(primary_key=True)
    base_image: Mapped["Image"] = mapped_column(ForeignKey("image.id", onupdate="cascade"))

    creator: Mapped[Optional[str]] = mapped_column(String())
    segment_fullpath: Mapped[str] = mapped_column(String())

    y_min: Mapped[int] = mapped_column(Integer())
    y_max: Mapped[int] = mapped_column(Integer())
    x_min: Mapped[int] = mapped_column(Integer())
    x_max: Mapped[int] = mapped_column(Integer())
    output_width: Mapped[int] = mapped_column(Integer())
    output_height: Mapped[int] = mapped_column(Integer())

    def __repr__(self):
        return f"Segment of image {self.base_image}"

## End Table definitions



class BaseStorage:
    def __init__(self, file):
        self.file = file
        self.engine = create_engine(self.file)
        Base.metadata.create_all(self.engine)

    def get_image(self, fullpath):
        with Session(self.engine) as session:
            img = session.execute(select(Image).where(Image.fullpath == fullpath)).one_or_none()
            if img is not None:
                return img
            else: # Image does not exist in DB
                im = cv2.imread(fullpath)
                if im is None:
                    height, width, colors = None, None, None
                else:
                    height, width, colors = im.shape

                session.add(
                        Image(name=os.path.split(fullpath)[1],
                              fullpath=fullpath,
                              height=height,
                              width=width,
                              colors=colors
                              )
                        )
                session.commit()
                return session.execute(select(Image).where(Image.fullpath == fullpath)).one_or_none()

    def get_all_images(self):
        with Session(self.engine) as session:
            return session.execute(select(Image)).all()



class BasicAnalysisDataStorage(BaseStorage):
    def __init__(self, file):
        super().__init__(file)

    def store(self, fullpath, imageIsGray, exifs):
        img_instance = self.get_image(fullpath)[0]
        with Session(self.engine) as session:
            img_object = ImageMetadata(
                    image_id=img_instance.id,
                    imageIsGray = imageIsGray,
                    exifs = [Exif(image_id=img_instance.id, exif_name=exif[0], exif_code=exif[1], exif_value=(exif[2])) for exif in exifs]
                    )
            session.add_all([img_object,])
            session.commit()

    def getByInstance(self, img_instance):
        with Session(self.engine) as session:
            meta = session.execute(select(ImageMetadata).where(ImageMetadata.image_id == img_instance.id))
            exif = session.execute(select(Exif).where(Exif.image_id == img_instance.id))
            return meta.scalars().all(), exif.scalars().all()


class BatchingDataStorage(BaseStorage):
    def __init__(self, file):
        super().__init__(file)

    def store(self, instances, creator=None):
        with Session(self.engine) as session:
            bds = ImageBatch(images = instances, creator=creator)
            session.add(bds)
            session.commit()

    def getBatchFromImage(self, image_instance):
        with Session(self.engine) as session:
            image = session.execute(select(Image).where(Image.id == image_instance.id)).one_or_none()
            if image is None:
                return None
            image = image[0]
            print(image.batch.images)
            return image.batch

    def getBatches(self):
        with Session(self.engine) as session:
            return session.execute(select(ImageBatch)).all()


class SegmentDataStorage(BaseStorage):
    def __init__(self, file):
        super().__init__(file)

    def store(self, image, segment_fullpath, y_min, y_max, x_min, x_max, output_width, output_height, creator=None):
        with Session(self.engine) as session:
            # Make sure we have an instance of the image db representation
            if not isinstance(image, Image):
                image = self.get_image(image)[0]

            seg = Segment(
                base_image=image.id,
                segment_fullpath=segment_fullpath,
                creator=creator,
                y_min=y_min,
                y_max=y_max,
                x_min=x_min,
                x_max=x_max,
                output_width=output_width,
                output_height=output_height,
                )
            session.add(seg)
            session.commit()

    def getSegments(self):
        with Session(self.engine) as session:
            return session.execute(select(Segment)).all()

    def getImages(self):
        return [i[0].segment_fullpath for i in self.getSegments()]

if __name__ == "__main__":
    ds = SegmentDataStorage("sqlite:///test.sqlite")
    img_instance1 = ds.get_image("/test/files/testfile1.jpg")[0]
    img_instance2 = ds.get_image("/test/files/testfile2.jpg")[0]
    ds.store(img_instance1, "ABC", 1,2,3,4,5,6)

    img_instance1 = ds.get_image("/test/files/testfile1.jpg")[0]

    print(img_instance1.segments)


