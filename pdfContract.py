from pprint import pprint
from collections import Counter
import pdfminer
import pdfplumber
from pdfminer.pdfpage import PDFPage, PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.layout import LAParams
from pdfminer.converter import PDFPageAggregator
# from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
import json,itertools
import pandas as pd
import sys
import re
import pandas
from pathlib import Path
import io
import time
import json
import datetime
import pandas as pd
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import numpy as np
import math
import re,pdfplumber
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument


TEXT_ELEMENTS = [
    pdfminer.layout.LTTextBox,
    pdfminer.layout.LTTextBoxHorizontal,
    pdfminer.layout.LTTextLine,
    pdfminer.layout.LTTextLineHorizontal
]

def get_tables(a):
    tlist=list()
    page=dl[a]
    try:
        for table in page.find_tables():
            if table:
                t=table.extract()
                if len(t)>1:
                    colHeadings=t.pop(0)
                    jd=len(set.intersection(set(colHeadings),set(headText)))
                    if jd==len(set(colHeadings)) or jd==0:
                        l,tp,rt,bt=table.bbox
                        tlist.append({'Page':a+1,"Columns":colHeadings,"Table":t,"BBox": {'Left': l, 'Top': tp, 'Right': rt, 'Bottom': bt}})
    except Exception as ex:
        print(ex)
    if len(tlist)!=0:
        return tlist
    else:
        return None


def get_border_tables(file):
    global maxPageSize
    # # Table and Text Extraction
    pdf = pdfplumber.open("uploads/{}".format(file))
    page_1 = pdf.pages[0]
    maxPageSize=(page_1.height)
    print(time.time()-start)
    global dl
    dl={n:page for n,page in enumerate(pdf.pages)}
    pgs=dl.keys()
    t1=mp.Pool(mp.cpu_count())
    tbList=t1.map(get_tables,dl)
    tbList=[x for x in tbList if x is not None]
    tbList=[item for sublist in tbList for item in sublist]
    print(time.time()-start)
    return tbList


def get_toc(fp):
    # # Open a PDF document.
    # fp = open(file, 'rb')
    parser = PDFParser(fp)

    document = PDFDocument(parser)
    toc=list()
    # Get the outlines of the document.
    try:
        outlines = document.get_outlines()
        for (level,title,dest,a,se) in outlines:
            if level <= 2:
                toks=title.split()
                # if re.match(r"[.\d]+",toks[0]):
                if len(toks)<10:
                    # if not re.match(r"\([a-z]\)",toks[0]):
                    toc.append((level,title))
    except Exception as ex:
        print(ex)
    print(time.time()-start)
    if len(toc)==0:
        toc=None
    return toc


def get_intersection(value, jinker):
    minx = value['Left']
    dy = value['Top']
    maxx = value['Left']+value['Width']
    retDict = dict()
    for ax, a in iter(jinker.items()):
        x, y, w, h = a
        x2 = x+w
        if y > dy:
            if minx < x and maxx > x and maxx <= x2:
                retDict[ax] = True
            elif maxx < x and minx >= x and minx <= x2:
                retDict[ax] = True
            elif minx >= x and minx <= x2 and maxx >= x and maxx <= x2:
                retDict[ax] = True
            elif minx < x and maxx > x2:
                retDict[ax] = True
            else:
                retDict[ax] = False
    count = 0
    for key, values in retDict.items():
        if values == True:
            count += 1
    if count == 1:
        return retDict
    else:
        return None


def get_REMatch(jaxx):
    if re.match(r'/\d\.\s+|\([a-z]\)\s+|\(.*?\)|[a-z]\)\s+|\[\d+\]$|\([0-9].*?\)|\w[.)]\s*|\([a-z]\)\s+|[a-z]\)\s+|•\s+|[A-Z]\.\s+|[IVX]+\.\s+/g', jaxx):
        return 2
    elif re.match(r'[0-9]*\n', jaxx):
        return 1
    elif re.match(r'^-?\d+(?:\.\d+)$', jaxx):
        return 1
    else:
        return 0


def zero(x):
    if len(x) == 2:
        if x["Is_Bullet"].tolist()[0] == 2:
            return pd.DataFrame({"Width": x["Width"].sum(), "Is_Bullet": x["Is_Bullet"].iloc[0],
                                 "Page": x["Page"].iloc[0], "Left": x["Left"].iloc[0], "Top": x["Top"].min(), "Height": x["Height"].max(),
                                 "Text": ' '.join(x['Text'].tolist()), "Font": ' '.join(list(set(x['Font'].tolist()))), "Size": ' '.join(list(set((x['Size'].tolist())))),
                                 "Font Count": len(' '.join(list(set(x['Font'].tolist()))).split(' '))}, index=[0])
        else:
            return x
    else:
        return x


def hMerger(data):
    pgno, adf = data
    adt = [x.sort_values(
        "Left", inplace=False) for a, x in adf.groupby(["Top"])]
    p2 = ThreadPool(mp.cpu_count())
    adf = pd.concat(p2.map(zero, adt))
    p2.close()
    return adf


def lp(page):
    try:
        interpreter.process_page(page)
        return device.get_result()
    except Exception as ex:
        print(ex)


def extract_page_layouts(file):
    """
    Extracts LTPage objects from a pdf file.
    modified from: http://www.degeneratestate.org/posts/2016/Jun/15/extracting-tabular-data-from-pdfs/
    Tests show that using PDFQuery to extract the document is ~ 5 times faster than pdfminer.
    """
    global toc
    laparams = LAParams()

    with open('uploads/{}'.format(file), mode='rb') as pdf_file:
        print("Open document %s" % pdf_file.name)
        toc=get_toc(pdf_file)
        # document = PDFQuery(pdf_file).doc
        # parser = PDFParser(pdf_file)
        # Create a PDF document object that stores the document structure.
        # Supply the password for initialization.
        # document = PDFDocument(parser)
        # if not document.is_extractable:
        #     raise PDFTextExtractionNotAllowed
        rsrcmgr = PDFResourceManager()
        global device
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        global interpreter
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        layouts = []
        pages=PDFPage.get_pages(pdf_file,check_extractable=False)
        # pages = PDFPage.create_pages(document)

        print(time.time()-start)
        layouts=[lp(page) for page in pages]
        # t1 = ThreadPool(mp.cpu_count())
        # layouts=t1.map(lp,pages)
        print(time.time()-start)

    return layouts


def get_text_objects(page_layout):
    # do multi processing
    texts = []
    # seperate text and rectangle elements
    for elem in page_layout:
        if isinstance(elem, pdfminer.layout.LTTextBoxHorizontal) or isinstance(elem, pdfminer.layout.LTTextBox):
            texts.extend(elem)
        elif isinstance(elem, pdfminer.layout.LTTextLine) or isinstance(elem, pdfminer.layout.LTTextLineHorizontal):
            texts.append(elem)
    return texts


def get_data(element):
    pgno, element = element
    text = element.get_text()
    if not text.isspace():
        (x0, y0, x1, y1) = element.bbox
        w = x1-x0
        h = y1-y0
        deto = list(element)
        font = list()
        size = list()
        for deto in list(element):
            if isinstance(deto, pdfminer.layout.LTChar):
                font.append(deto.fontname)
                size.append(str(round(deto.size, 2)))
        font = list(set(font))
        size = list(set(size))
        return pd.DataFrame({"Width": int(float(w)), "Page": pgno, "Left": int(float(x0)), "Top": int(float(y1)), "Height": int(float(h)),
                             "Text": text, "Font": ' '.join(font), "Size": ' '.join(size), "Font Count": len(font)}, index=[0])


def get_text_data(objs, pgno):
    objs = [(pgno, obj) for obj in objs]
    p1 = ThreadPool(mp.cpu_count())
    dfs = p1.map(get_data, objs)
    p1.close()
    return pd.concat(dfs)


def get_table_rects(lst):
    if lst is not None:
        jk = list()
        if len(lst) > 1:
            ele = lst.pop(0)
            for i in lst:
                if not is_intersect(i['BBox'], ele['BBox']):
                    jk.append(i)

            jk.extend(get_table_rects(lst))
            return jk
        else:
            jk.extend(lst)
            return jk


def is_intersect(r1, r2):
    if r1['Left'] >= r2['Left'] and r1['Right'] <= r2['Right'] and r1['Top'] <= r2['Top'] and r1['Bottom'] >= r2['Bottom']:
        return False
    # # if r1['Left'] > r2['Right'] or r1['Right'] < r2['Left']:
    # #     return False
    # # if r1['Top'] > r2['Bottom'] or r1['Bottom'] < r2['Top']:
    # #     return False
    return True
    # if r2['Left'] < r1['Left'] < r2['Right'] and r2['Top'] < r1['Top'] < r2['Bottom']:
    #     return False
    # else:
    #     return True

def get_table_struct(tbdf):
    rows=list()
    for top,x in tbdf.groupby('Top'):
        row=dict()
        for jojo in x.to_dict('records'):
            row[jojo['column']]=jojo['Text']
        rows.append(row)
    return pd.DataFrame(rows)

def tb_detr(x):
    x,PgNO,ylvl=x
    a, x = x
    tbDF = pd.DataFrame()
    cols_cords = dict()
    cols_dict = dict()
    for jojo in x.to_dict('records'):
        cols_cords[jojo['Text']] = (
            jojo['Left'], jojo['Top'], jojo['Width'], jojo['Height'])

        cols_dict[jojo['Text']] = list()
        for b, y in ylvl:
            if b != a:
                for dodo in y.to_dict('records'):
                    rDic = get_intersection(dodo, cols_cords)
                    if rDic:
                        for key, value in rDic.items():
                            if value == True:
                                dodo['column'] = key
                                dodo['Right'] = dodo['Left']+dodo['Width']
                                dodo['Bottom'] = dodo['Top']+dodo['Height']
                                tbDF = tbDF.append(
                                    pd.DataFrame(dodo, index=[0]))
                                dodo['DP']=(dodo['Left'],dodo['Top'])

                                cols_dict[key].append(dodo)
    if set(['Left','Bottom','Right','Top']).issubset(tbDF.columns):
        dek = {'Left': tbDF['Left'].min(), 'Top': tbDF['Bottom'].max(
        ), 'Right': tbDF['Right'].max(), 'Bottom': tbDF['Top'].min()}
    else:
        dek=None


    rowId=list()
    tble=dict()
    jd=len(set.intersection(set(cols_dict.keys()),set(headText)))
    if bool(cols_dict) and dek is not None and len(cols_dict.keys())>2 and (jd==len(set(cols_dict.keys())) or jd==0):
        for key in cols_dict:
            try:
                if len(cols_dict[key]) != 0:
                    df=pd.DataFrame(cols_dict[key])
                    df=df.drop_duplicates(subset='DP')
                    rowId.extend(df['DP'].values.tolist())
                    tble[key]=df[["Text","DP"]].to_dict('records')
                else:
                    tble=None
                    break
            except Exception as ex:
                iop=0
    else:
        tble=None
    rowId=list(set(rowId))

    if bool(tble):
        return {"BBox":dek,"Page":PgNO,"Table": get_table_struct(tbDF),"Columns":list(tble.keys())}
    else:
        return None


def page_table_det(df):
    pgNo, df = df
    ylvl = df.groupby(["Top"])

    ado = [((a, x),pgNo,ylvl) for a, x in ylvl if len(x) > 2]

    t3 = mp.Pool(mp.cpu_count())
    tablesList = t3.map(tb_detr, ado)
    t3.close()
    jl=list()
    tablesList=[x for x in tablesList if x is not None]


    if len(tablesList) > 0:
        tl=get_table_rects(tablesList)
        print(tl[0].get('Table').keys())
        print('PageNo:{}'.format(pgNo))
        print(len(tl))
        return tl
    else:
        return None


def score(row):
    score = 0
    if 'Bold' in row['Font']:
        score += 1
    if maxFontStyle[0] not in row['Font']:
        score+=1
    if int(float(row['Size'])) > maxSize:
        score += int(float(row['Size'])) - maxSize
    if row['Font Count'] > 1:
        if len(row["Font"].split()) == len([x for x in row["Font"].split() if "Bold" in x]):
            score+=1
        else:
            score = 0
    # if row['TOC'] == 1:
    #     score = 0
    if row['Is_Bullet'] == 1:
        score=0
    # if len(row["Text"].split()) > 7:
    #     score=0
    # if re.search(r'(\d+(?:\.\d+)*\.?(\d?))\s(\D*?)(\s?)(\d*)',row['Text']):
    #     score = 0
    return score


def toc_lines(row):
    if re.match(r'/\d\.\s+|\([a-z]\)\s+|\(.*?\)|[a-z]\)\s+|\[\d+\]$|\([0-9].*?\)|\w[.)]\s*|\([a-z]\)\s+|[a-z]\)\s+|•\s+|[A-Z]\.\s+|[IVX]+\.\s+/g', row['Text']):
        return 1
    else:
        return 0


def parse_layouts(axe):
    global layouts
    pg_no = axe
    layout =layouts[axe]
    return get_text_data(get_text_objects(layout), pgno=pg_no+1)



def extract_from_pdf(file=None):
    global toc
    global maxPageSize
    interpreter = None
    device = None
    global layouts,start,maxFontStyle,maxSize,headText
    if file is not None:
        start = time.time()
        print(datetime.datetime.now())
        page_layouts = extract_page_layouts(file)

        print("Number of pages: %d" % len(page_layouts))
        text = list()
        final_df = pandas.DataFrame()
        adf = list()

        layouts = {pg_no: layout for pg_no, layout in enumerate(page_layouts)}
        print(time.time()-start)
        p1 = mp.Pool(mp.cpu_count())
        print('Starting multiprocessing')
        adf = p1.map(parse_layouts, layouts.keys())
        p1.close()
        final_df = pd.concat(adf)
        print(time.time()-start)
        p1 = mp.Pool(mp.cpu_count())
        final_df["Is_Bullet"] = [get_REMatch(x["Text"])
                                for indx, x in final_df.iterrows()]
        dst = [(a, x) for a, x in final_df.groupby(["Page"])]
        final_df = pd.concat(p1.map(hMerger, dst))
        print(time.time()-start)
        p1.close()


        final_df = final_df.sort_values(
            ["Page", "Top", "Left"], ascending=[True, False, True])
        jo = final_df[final_df['Font'].str.contains('Bold')]

        jo['Size'] = jo.Size.str.split(' ').apply(lambda x: max(
            [int(float(i)) for i in x if i is not None])).values.tolist()
        jox = jo['Text'].values.tolist()
        final_df['Size'] = final_df.Size.str.split(' ').apply(lambda x: max(
            [int(float(i)) for i in x if i is not None])).values.tolist()
        final_df['Heading'] = final_df.Text.apply(lambda x: 1 if x in jox else 0)
        maxSize = int(float(final_df['Size'].mode()[0]))
        print('Max Size: {}'.format(maxSize))
        fl=list()
        dd=[fl.extend(x.split()) for x in final_df['Font'].tolist()]

        fl=Counter(fl)
        pprint(fl)
        maxFontStyle=fl.most_common(1)[0]
        pprint(maxFontStyle[0])

        final_df['TOC']=final_df.apply(toc_lines, axis=1)
        final_df['SScore'] = final_df.apply(score, axis=1)

        final_df['Heading'] = final_df.SScore.apply(lambda x: 1 if x > 0 else 0)
        headText=final_df[final_df["SScore"] > 0]['Text'].values.tolist()


        print(time.time()-start)
        pdfs = [(a, x) for a, x in final_df.groupby(["Page"])]
        p1 = ThreadPool(mp.cpu_count())
        awe = p1.map(page_table_det, pdfs)
        awe=[x for x in awe if x is not None]
        awe=[item for sublist in awe for item in sublist if sublist is not None]
        print("Len before :{}".format(len(awe)))
        p1.close()
        awe = list(filter(None, awe))
        print(time.time()-start)
        bTables=get_border_tables(file)
        bless_Table=list()
        for (a, b) in itertools.product(awe, bTables):
            if a['Page']==b['Page'] and not is_intersect(a['BBox'], b['BBox']):
                bless_Table.append(a)
        print('len after:{}'.format(len(bless_Table)))
        data=final_df[final_df['SScore']>0]['Text'].values.tolist()
        data=Counter(data)
        dd=[k for k, c in data.items() if c < 5]
        data=final_df[final_df['Text'].isin( dd)]
        data=data[data['SScore']>0 ]
        reps={a:x for a,x in final_df['Text'].value_counts().to_dict().items() if x>3}

        adf=final_df[final_df['Text'].isin(reps.keys())]
        adf['Bottom']=adf['Top']+adf['Height']
        adf=adf.groupby(["Text"]).size().reset_index().rename(columns={0:'count'})

        adf=(adf[adf['count']>2])
        uper=(maxPageSize/4)*3
        lowr=maxPageSize/4
        print(uper,lowr)
        h1=adf['Text'].values.tolist()
        h1=final_df[final_df['Text'].isin(h1)]
        hheaders=h1[h1['Top']>=uper].groupby(["Text"]).size().reset_index().rename(columns={0:'count'})
        hfooters=h1[h1['Top']<=lowr].groupby(["Text"]).size().reset_index().rename(columns={0:'count'})
        hheaders=hheaders[hheaders['count']>=len(page_layouts)/2]
        pprint(h1[h1['Text'].isin(hheaders['Text'].values.tolist())].head(10))
        hheaders=h1[h1['Text'].isin(hheaders['Text'].values.tolist())]['Top'].min()

        hfooters=hfooters[hfooters['count']>=len(page_layouts)/2]
        hfooters=h1[h1['Text'].isin(hfooters['Text'].values.tolist())]['Top'].max()
        pprint(hheaders)
        pprint(hfooters)
        if hheaders==0 :
            hheaders=maxPageSize
        if math.isnan(hheaders):
            hheaders=maxPageSize


        final_df=final_df[final_df['Top'] < hheaders]
        if math.isnan(hfooters):
            hfooters=0
        final_df=final_df[final_df['Top'] > hfooters]
        final_df= final_df.sort_values(
            ["Page", "Top", "Left"], ascending=[True, False, True])

        data=final_df[["Text","SScore"]].to_dict("records")
        hnText=list()
        node={"Heading":None,"Text":" "}
        for x in data:
            if x['SScore']>0 and node["Heading"] is not None:
                hnText.append(node)
                node={"Heading":None,"Text":" "}
            if x['SScore']>0 and node["Heading"] is None:
                node['Heading']=x['Text']
                node['Score']=x['SScore']
            if x['SScore'] == 0:
                node['Text']=node['Text']+x['Text']

        if len(hnText)==0:
            data=final_df['Text'].values.tolist()
        else:
            data = hnText
        print(time.time()-start)
        return {'Text':data,'Bordered Tables':bless_Table,'Borderless Tables':bTables}
