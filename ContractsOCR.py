from trp import Document
import json,logging
from botocore.exceptions import ClientError
from json import JSONEncoder
import re
from pathlib import Path
import io
import time
import itertools
import json
import os
import pymongo
import datetime
import boto3
import pandas as pd
import gridfs,base64
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from pdf2image import convert_from_path
import numpy as np
from botocore.client import Config

tablesList = list()

config = Config(retries = dict(max_attempts = 15))
textract = boto3.client('textract',config=config)

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["ZContracts"]



class Grouper:
    """simple class to perform comparison when called, storing last element given"""

    def __init__(self, diff):
        self.last = None
        self.diff = diff

    def predicate(self, item):
        if self.last is None:
            return True
        return abs(self.last - item) < self.diff

    def __call__(self, item):
        """called with each item by takewhile"""
        result = self.predicate(item)
        self.last = item
        return result


def group_by_difference(items, diff=10):
    results = []
    start = 0
    remaining_items = items
    while remaining_items:
        g = Grouper(diff)
        group = [*itertools.takewhile(g, remaining_items)]
        results.append(group)
        start += len(group)
        remaining_items = items[start:]
    return results


def get_level(value, jinker):
    axe = iter(jinker.items())
    for x, a in axe:
        if value in a:
            return x


def get_REMatch(jaxx):
    if re.match(r'/\d\.\s+|\([a-z]\)\s+|\(.*?\)|[a-z]\)\s+|\[\d+\]$|\([0-9].*?\)|\w[.)]\s*|\([a-z]\)\s+|[a-z]\)\s+|•\s+|[A-Z]\.\s+|[IVX]+\.\s+/g', jaxx):
        return 1
    elif re.match(r'^-?\d+(?:\.\d+)$', jaxx):
        return 1
    else:
        return 0


def zero(x):
    if len(x) == 2:
        if x["Is_Bullet"].tolist()[0] == 1:
            return pd.DataFrame({"Width": x["Width"].sum(), "Is_Bullet": x["Is_Bullet"].iloc[0], "Y_level": x["Y_level"].iloc[0],
                                 "Page": x["Page"].iloc[0], "Left": x["Left"].iloc[0], "Top": x["Top"].min(), "Height": x["Height"].max(),
                                 "Text": ' '.join(x['Text'].tolist()), "X_level": x["X_level"].min()}, index=[0])

        else:
            return x
    else:
        return x


def hMerger(pgno, adf):
    ya = group_by_difference(list(set(adf["Top"].values.tolist())), 40)
    for sublist in ya:
        sublist.sort()
    ya.sort()
    qwe = list(range(len(ya)))
    ya = {q: a for q, a in zip(qwe, ya)}
    adf["Y_level"] = [get_level(r, ya) for r in adf["Top"].tolist()]
    adf = pd.concat([zero(x) for x in [x.sort_values(
        "X_level", inplace=False) for a, x in adf.groupby(["Y_level"])]])

    return adf


def get_Lines(pgno, data):
    a, x = data
    doc = Document(a)
    w, h = x.size
    lines = list()
    for page in doc.pages:
        tbList = list()
        for table in page.tables:
            is_table = True
            for row in table.rows:
                if len(row.cells) <= 2:
                    is_table = False
            if is_table:
                t = list()
                for row in table.rows:
                    r = list()
                    dummy = [r.append([cell.text]) for cell in row.cells]
                    t.append(r)
                tablesList.append((pgno, t))

                tbList.append(table.geometry)
        for line in page._lines:
            inside = False
            for rect in tbList:
                if not(line.geometry.boundingBox.left >= rect.boundingBox.left
                       and line.geometry.boundingBox.top >= rect.boundingBox.top
                       and line.geometry.boundingBox.left+line.geometry.boundingBox.width <= rect.boundingBox.left+rect.boundingBox.width
                       and line.geometry.boundingBox.top+line.geometry.boundingBox.height <= rect.boundingBox.top+rect.boundingBox.height):

                    inside = True
                    if len(tbList) == 1:
                        lines.append(line)
                        inside = False
            if not inside and len(tbList) > 1:
                lines.append(line)
                inside = False
            elif len(tbList) == 0:
                lines.append(line)
    lines = [{"Text": line.text, "Page": pgno, "Left": line.geometry.boundingBox.left*w, "Top": line.geometry.boundingBox.top *
              h, "Height": line.geometry.boundingBox.height*h, "Width": line.geometry.boundingBox.width*w} for line in lines]
    return pd.DataFrame(lines)


def get_jpeg(img):
    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format="jpeg")
    return imgByteArr.getvalue()


def txtrctAPI(img):
    return textract.analyze_document(
        Document={
            'Bytes': get_jpeg(img),
        },
        FeatureTypes=["TABLES"])


def get_images(pdf):
    print("Getting Image")
    imgs=list()
    example = Pdf.open(pdf)
    for page in example.pages:
        imkeys=list(page.images.keys())
        for key in imkeys:
            imgs.append(PdfImage(page.images[key]).as_pil_image())
            print("Width and Height:{}".format(PdfImage(page.images[key]).as_pil_image().size))
    x=[ a.save("images/{}.jpg".format(i)) for i,a in enumerate(imgs)]
    return imgs


def write_new_pdf(path,db):
    # db = MongoClient('mongodb://localhost:27017/').myDB
    fs = gridfs.GridFS(db)
    # Note, open with the "rb" flag for "read bytes"
    with open(path, "rb") as f:
        encoded_string = base64.b64encode(f.read())
    with fs.new_file(chunkSize=8000000,filename=path) as fp:
        fp.write(encoded_string)


def mongo_image(img,db):
    # db = MongoClient('mongodb://localhost:27017/').myDB
    fs = gridfs.GridFS(db)
    # Note, open with the "rb" flag for "read bytes"
    with open(path, "rb") as f:
        encoded_string = base64.b64encode(f.read())
    with fs.new_file(chunkSize=8000000,filename=path) as fp:
        fp.write(encoded_string)


def read_pdf(filename,db):
    # Usual setup
    # db = MongoClient('mongodb://localhost:27017/').myDB
    fs = gridfs.GridFS(db)
    # Standard query to Mongo
    data = fs.find_one(filter=dict(filename=filename))
    with open(filename, "wb") as f:
        f.write(base64.b64decode(data.read()))

def ocrDoc(nameX):

    print(datetime.datetime.now())
    start = time.time()
    print("Starting, {}".format(nameX))
    try:
        images = convert_from_path('uploads/{}'.format(nameX))
        print("GOT Images: {}".format(len(images)))
        print(time.time()-start)
        print(time.time()-start)

        p2 = ThreadPool(mp.cpu_count())
        axe = p2.map(txtrctAPI, images)
        
        print(time.time()-start)
        p2.close()

        df = pd.concat([get_Lines(a, x)
                        for a, x in enumerate(zip(axe, images))])
        print(df.head())
        print(time.time()-start)

        df = df.sort_values(["Page", "Top", "Left"])
        df["Is_Bullet"] = [get_REMatch(x["Text"]) for indx, x in df.iterrows()]

        print(time.time()-start)
        print(df.head())

        xa = group_by_difference(list(set(df["Left"].values.tolist())), 10)
        for sublist in xa:
            sublist.sort()
        xa.sort()
        qwe = list(range(len(xa)))
        xa = {q: a for q, a in zip(qwe, xa)}
        df["X_level"] = [get_level(x["Left"], xa) for i, x in df.iterrows()]
        print(time.time()-start)

        df = pd.concat([hMerger(a, x) for a, x in df.groupby(["Page"])])

        print(time.time()-start)

        print(df.head())
        df["Label"] = ["Heading" if fdf["X_level"] ==
                       1 else "Text" for i, fdf in df.iterrows()]

        print(df.head())

        df = df[["Text", "Page", "Left", "Top", "Height", "Width", "Label"]]
        print(df.head())
        print(time.time()-start)
        df=df.to_dict("records")
        final_results = {"Text": df, "Tables": tablesList}
        
        p2 = ThreadPool(mp.cpu_count())
        images = p2.map(get_jpeg, images)
        
        col=mydb["ContractsOCR"]
        write_new_pdf('uploads/{}'.format(nameX),mydb)
        
        fs = gridfs.GridFS(database=mydb,collection="ContractsOCR")
        
        images={ str(n):fs.put(data=i,filename="{}_Page_{}".format(nameX,n)) for n,i in enumerate(images)  }
        
        col.insert({
            "DocName":nameX,
            "Images":images,
            "Text": df,
            "Tables": tablesList
        })
        
        return final_results
    except Exception as ex:
        print(time.time()-start)
        print(ex)
        return "Error Found: {}".format(ex)
