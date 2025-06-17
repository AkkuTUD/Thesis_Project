from flask import Flask, request, render_template, redirect, url_for
from flask_cors import CORS
import os
import psycopg2
import base64
import os
import plotly
import plotly.express as px
import pandas as pd
import json
from collections import Counter
from flask_paginate import Pagination, get_page_parameter
import logging
import requests
from flask import Flask, request, jsonify
from basf_auth.framework.flask_auth import requires_basf_auth
from basf_auth.authorization.usernames import UsernameListAuthorization
from basf_auth.authorization.ldap_api import LDAPAPIAuthorization
from basf_auth.authorization.accessit_api import AccessITAuthorization
from basf_auth.authorization.combinators import Not
from basf_auth.authentication.basf_federation_services import (
    BASFFederationBearerTokenAuthentication,
    BASFFederationCookiesAuthentication,
)


app = Flask(__name__)
CORS(app, supports_credentials=True)
bal = logging.getLogger("basf_auth")
bal.handlers = app.logger.handlers



if "APPSTORE_ENV" in os.environ:
    from myProxyFix import ReverseProxied

    app.wsgi_app = ReverseProxied(app.wsgi_app)

def get_db_connection():
     connection = psycopg2.connect(                                                  
         user = os.getenv("DB_USER"),                                      
         password = os.getenv("DB_PASSWORD"),                                  
         host = os.getenv("DB_HOST"),                                            
         port = os.getenv("DB_PORT"),                                          
         database = os.getenv("DB_DATABASE")                                       
     )
     return connection


def convert_base64(result):
    result = list(result)
    result[1] = base64.b64encode(result[1]).decode("utf-8")
    result[6] = base64.b64encode(result[6]).decode("utf-8")
    result[7] = base64.b64encode(result[7]).decode("utf-8")
    result[8] = base64.b64encode(result[8]).decode("utf-8")
    result[9] = base64.b64encode(result[9]).decode("utf-8")
    result[10] = base64.b64encode(result[10]).decode("utf-8")
    result[11] = base64.b64encode(result[11]).decode("utf-8")
    result[12] = base64.b64encode(result[12]).decode("utf-8")
    result[13] = base64.b64encode(result[13]).decode("utf-8")
    result[14] = base64.b64encode(result[14]).decode("utf-8")
    result = tuple(result)
    return result

def convert_base64_2_classes(result):
    result = list(result)
    result[1] = base64.b64encode(result[1]).decode("utf-8")
    result[6] = base64.b64encode(result[6]).decode("utf-8")
    result[7] = base64.b64encode(result[7]).decode("utf-8")
    result = tuple(result)
    return result

def convert_base64_FN(result):
    result = list(result)
    result[0] = base64.b64encode(result[0]).decode("utf-8")
    result = tuple(result)
    return result


@app.route("/")
@requires_basf_auth(
    authorization_scheme=UsernameListAuthorization(
        ["TanwarA", "HerrmJ27", "MarxfeHA", "HofmanH2", "BenzinS"]
    ),
)
def index():
    username = request.basf_auth.get_username()
    # username = "TanwarA"
    return render_template('index.html', username=username) #, data_FP=data_FP, data_TP=data_TP, data_FN=data_FN, feedbackList=feedbackList, pagination=pagination)

@app.route("/tiles_split_2_cls")
@requires_basf_auth(
    authorization_scheme=UsernameListAuthorization(
        ["TanwarA", "HerrmJ27", "MarxfeHA", "HofmanH2", "BenzinS"]
    ),
)
def tiles_split_2_cls():
    
    username = request.basf_auth.get_username()

    tiles_data_TP_2=[]
    connection = get_db_connection()
    cursor = connection.cursor()

    page = request.args.get(get_page_parameter(), type=int, default=1)
    limit = 5
    offset = page*limit - limit
    
    outcome = 'TP'   
    xai_id = 'xai_6'
    if username=="TanwarA":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_tanwara FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_tanwara FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="HerrmJ27":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_herrmj27 FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_herrmj27 FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="MarxfeHA":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_marxfeha FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_marxfeha FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="HofmanH2":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_hofmanh2 FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_hofmanh2 FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="BenzinS":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_benzins FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_benzins FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()

    for row in result:
        print(row)
        row = convert_base64_2_classes(row)
        tiles_data_TP_2.append(row)
    
    
    pagination = Pagination(page=page, page_per=limit, total=total)
    cursor.execute('SELECT feedback FROM feedback_lrp_table;')
    feedbackList = cursor.fetchall()
    
    return render_template('tiles_split_2_cls.html', tiles_data_TP_2=tiles_data_TP_2, feedbackList=feedbackList, pagination=pagination, username=username)

@app.route("/tiles_split_2_cls/FN")
@requires_basf_auth(
    authorization_scheme=UsernameListAuthorization(
        ["TanwarA", "HerrmJ27", "MarxfeHA", "HofmanH2", "BenzinS"]
    ),
)
def tiles_split_2_cls_FN():

    username = request.basf_auth.get_username()
    # username = "TanwarA"

    tiles_data_FN_2=[]
    connection = get_db_connection()
    cursor = connection.cursor()

    page = request.args.get(get_page_parameter(), type=int, default=1)
    limit = 5
    offset = page*limit - limit
    
    outcome = 'FN'
    xai_id = 'xai_6'
    if username=="TanwarA":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_tanwara FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_tanwara FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="HerrmJ27":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_herrmj27 FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_herrmj27 FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="MarxfeHA":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_marxfeha FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_marxfeha FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="HofmanH2":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_hofmanh2 FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_hofmanh2 FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="BenzinS":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_benzins FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_benzins FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    

    for row in result:
        print(row)
        row = convert_base64_2_classes(row)
        tiles_data_FN_2.append(row)
    
    pagination = Pagination(page=page, page_per=limit, total=total)
    cursor.execute('SELECT feedback FROM feedback_lrp_table;')
    feedbackList = cursor.fetchall()
    
    return render_template('tiles_split_2_cls_FN.html', tiles_data_FN_2=tiles_data_FN_2, feedbackList=feedbackList, pagination=pagination, username=username)

@app.route("/tiles_split_2_cls/FP")
@requires_basf_auth(
    authorization_scheme=UsernameListAuthorization(
        ["TanwarA", "HerrmJ27", "MarxfeHA", "HofmanH2", "BenzinS"]
    ),
)
def tiles_split_2_cls_FP():

    username = request.basf_auth.get_username()

    tiles_data_FP_2=[]

    connection = get_db_connection()
    cursor = connection.cursor()

    page = request.args.get(get_page_parameter(), type=int, default=1)
    limit = 5
    offset = page*limit - limit
        
    outcome = 'FP'
    xai_id = 'xai_6'
    if username=="TanwarA":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_tanwara FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_tanwara FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="HerrmJ27":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_herrmj27 FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_herrmj27 FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="MarxfeHA":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_marxfeha FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_marxfeha FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="HofmanH2":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_hofmanh2 FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_hofmanh2 FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="BenzinS":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_benzins FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_benzins FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    
    for row in result:
        print(row)
        row = convert_base64_2_classes(row)
        tiles_data_FP_2.append(row)
    
    pagination = Pagination(page=page, page_per=limit, total=total)
    cursor.execute('SELECT feedback FROM feedback_lrp_table;')
    feedbackList = cursor.fetchall()

    return render_template('tiles_split_2_cls_FP.html', tiles_data_FP_2=tiles_data_FP_2, feedbackList=feedbackList, pagination=pagination, username=username)


@app.route("/tiles_split_9_cls")
@requires_basf_auth(
    authorization_scheme=UsernameListAuthorization(
        ["TanwarA", "HerrmJ27", "MarxfeHA", "HofmanH2", "BenzinS"]
    ),
)
def tiles_split_9_cls():
    
    username = request.basf_auth.get_username()
    

    data_TP=[]
    connection = get_db_connection()
    cursor = connection.cursor()

    page = request.args.get(get_page_parameter(), type=int, default=1)
    limit = 5
    offset = page*limit - limit
    
    outcome = 'TP'
    xai_id = 'xai_3'
    if username=="TanwarA":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_tanwara FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_tanwara FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="HerrmJ27":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_herrmj27 FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_herrmj27 FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="MarxfeHA":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_marxfeha FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_marxfeha FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="HofmanH2":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_hofmanh2 FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_hofmanh2 FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="BenzinS":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_benzins FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_benzins FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()

    for row in result:
        print(row)
        row = convert_base64(row)
        data_TP.append(row)
    
    
    pagination = Pagination(page=page, page_per=limit, total=total)
    cursor.execute('SELECT feedback FROM feedback_lrp_table;')
    feedbackList = cursor.fetchall()
    
    return render_template('tiles_split_9_cls.html', data_TP=data_TP, feedbackList=feedbackList, pagination=pagination, username=username)

@app.route("/tiles_split_9_cls/FN")
@requires_basf_auth(
    authorization_scheme=UsernameListAuthorization(
        ["TanwarA", "HerrmJ27", "MarxfeHA", "HofmanH2", "BenzinS"]
    ),
)
def tiles_split_9_cls_FN():
    
    username = request.basf_auth.get_username()

    data_FN=[]
    connection = get_db_connection()
    cursor = connection.cursor()

    page = request.args.get(get_page_parameter(), type=int, default=1)
    limit = 5
    offset = page*limit - limit
    
    outcome = 'FN'
    xai_id = 'xai_3'
    if username=="TanwarA":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_tanwara FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_tanwara FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="HerrmJ27":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_herrmj27 FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_herrmj27 FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="MarxfeHA":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_marxfeha FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_marxfeha FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="HofmanH2":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_hofmanh2 FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_hofmanh2 FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="BenzinS":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_benzins FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_benzins FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    
    for row in result:
        print(row)
        row = convert_base64(row)
        data_FN.append(row)
    
    pagination = Pagination(page=page, page_per=limit, total=total)
    cursor.execute('SELECT feedback FROM feedback_lrp_table;')
    feedbackList = cursor.fetchall()
    
    return render_template('tiles_split_9_cls_FN.html', data_FN=data_FN, feedbackList=feedbackList, pagination=pagination, username=username)


@app.route("/tiles_split_9_cls/FP")
@requires_basf_auth(
    authorization_scheme=UsernameListAuthorization(
        ["TanwarA", "HerrmJ27", "MarxfeHA", "HofmanH2", "BenzinS"]
    ),
)
def tiles_split_9_cls_FP():
    
    username = request.basf_auth.get_username()

    data_FP=[]

    connection = get_db_connection()
    cursor = connection.cursor()

    page = request.args.get(get_page_parameter(), type=int, default=1)
    limit = 5
    offset = page*limit - limit
        
    outcome = 'FP'
    xai_id = 'xai_3'
    if username=="TanwarA":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_tanwara FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_tanwara FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="HerrmJ27":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_herrmj27 FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_herrmj27 FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="MarxfeHA":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_marxfeha FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_marxfeha FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="HofmanH2":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_hofmanh2 FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_hofmanh2 FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="BenzinS":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_benzins FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_benzins FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    
    for row in result:
        print(row)
        row = convert_base64(row)
        data_FP.append(row)
    
    pagination = Pagination(page=page, page_per=limit, total=total)
    cursor.execute('SELECT feedback FROM feedback_lrp_table;')
    feedbackList = cursor.fetchall()

    return render_template('tiles_split_9_cls_FP.html', data_FP=data_FP, feedbackList=feedbackList, pagination=pagination, username=username)


@app.route("/animal_split_2_cls")
@requires_basf_auth(
    authorization_scheme=UsernameListAuthorization(
        ["TanwarA", "HerrmJ27", "MarxfeHA", "HofmanH2", "BenzinS"]
    ),
)
def animal_split_2_cls():
    
    username = request.basf_auth.get_username()    

    data_TP_2=[]
    connection = get_db_connection()
    cursor = connection.cursor()

    page = request.args.get(get_page_parameter(), type=int, default=1)
    limit = 5
    offset = page*limit - limit
    
    outcome = 'TP'   
    xai_id = 'xai_4'
    if username=="TanwarA":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_tanwara FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_tanwara FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="HerrmJ27":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_herrmj27 FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_herrmj27 FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="MarxfeHA":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_marxfeha FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_marxfeha FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="HofmanH2":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_hofmanh2 FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_hofmanh2 FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="BenzinS":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_benzins FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_benzins FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()

    for row in result:
        print(row)
        row = convert_base64_2_classes(row)
        data_TP_2.append(row)
    
    
    pagination = Pagination(page=page, page_per=limit, total=total)
    cursor.execute('SELECT feedback FROM feedback_lrp_table;')
    feedbackList = cursor.fetchall()
    
    return render_template('animal_split_2_cls.html', data_TP_2=data_TP_2, feedbackList=feedbackList, pagination=pagination, username=username)

@app.route("/animal_split_2_cls/FN")
@requires_basf_auth(
    authorization_scheme=UsernameListAuthorization(
        ["TanwarA", "HerrmJ27", "MarxfeHA", "HofmanH2", "BenzinS"]
    ),
)
def animal_split_2_cls_FN():

    username = request.basf_auth.get_username()
    

    data_FN_2=[]
    connection = get_db_connection()
    cursor = connection.cursor()

    page = request.args.get(get_page_parameter(), type=int, default=1)
    limit = 5
    offset = page*limit - limit
    
    outcome = 'FN'
    xai_id = 'xai_4'
    if username=="TanwarA":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_tanwara FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_tanwara FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="HerrmJ27":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_herrmj27 FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_herrmj27 FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="MarxfeHA":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_marxfeha FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_marxfeha FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="HofmanH2":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_hofmanh2 FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_hofmanh2 FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="BenzinS":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_benzins FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_benzins FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    
    for row in result:
        print(row)
        row = convert_base64_2_classes(row)
        data_FN_2.append(row)
    
    pagination = Pagination(page=page, page_per=limit, total=total)
    cursor.execute('SELECT feedback FROM feedback_lrp_table;')
    feedbackList = cursor.fetchall()
    
    return render_template('animal_split_2_cls_FN.html', data_FN_2=data_FN_2, feedbackList=feedbackList, pagination=pagination, username=username)

@app.route("/animal_split_2_cls/FP")
@requires_basf_auth(
    authorization_scheme=UsernameListAuthorization(
        ["TanwarA", "HerrmJ27", "MarxfeHA", "HofmanH2", "BenzinS"]
    ),
)
def animal_split_2_cls_FP():

    username = request.basf_auth.get_username()
    

    data_FP_2=[]

    connection = get_db_connection()
    cursor = connection.cursor()

    page = request.args.get(get_page_parameter(), type=int, default=1)
    limit = 5
    offset = page*limit - limit
        
    outcome = 'FP'
    xai_id = 'xai_4'
    if username=="TanwarA":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_tanwara FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_tanwara FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="HerrmJ27":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_herrmj27 FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_herrmj27 FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="MarxfeHA":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_marxfeha FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_marxfeha FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="HofmanH2":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_hofmanh2 FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_hofmanh2 FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="BenzinS":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_benzins FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, feedback_benzins FROM lrp_2_classes where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    
    for row in result:
        print(row)
        row = convert_base64_2_classes(row)
        data_FP_2.append(row)
    
    pagination = Pagination(page=page, page_per=limit, total=total)
    cursor.execute('SELECT feedback FROM feedback_lrp_table;')
    feedbackList = cursor.fetchall()

    return render_template('animal_split_2_cls_FP.html', data_FP_2=data_FP_2, feedbackList=feedbackList, pagination=pagination, username=username)

@app.route("/animal_split_9_cls")
@requires_basf_auth(
    authorization_scheme=UsernameListAuthorization(
        ["TanwarA", "HerrmJ27", "MarxfeHA", "HofmanH2", "BenzinS"]
    ),
)
def animal_split_9_cls():
    
    username = request.basf_auth.get_username()
    

    animal_data_TP=[]
    connection = get_db_connection()
    cursor = connection.cursor()

    page = request.args.get(get_page_parameter(), type=int, default=1)
    limit = 5
    offset = page*limit - limit
    
    outcome = 'TP'
    xai_id = 'xai_5'
    if username=="TanwarA":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_tanwara FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_tanwara FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="HerrmJ27":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_herrmj27 FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_herrmj27 FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="MarxfeHA":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_marxfeha FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_marxfeha FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="HofmanH2":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_hofmanh2 FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_hofmanh2 FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="BenzinS":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_benzins FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_benzins FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()

    for row in result:
        print(row)
        row = convert_base64(row)
        animal_data_TP.append(row)
    
    
    pagination = Pagination(page=page, page_per=limit, total=total)
    cursor.execute('SELECT feedback FROM feedback_lrp_table;')
    feedbackList = cursor.fetchall()
    
    return render_template('animal_split_9_cls.html', animal_data_TP=animal_data_TP, feedbackList=feedbackList, pagination=pagination, username=username)

@app.route("/animal_split_9_cls/FP")
@requires_basf_auth(
    authorization_scheme=UsernameListAuthorization(
        ["TanwarA", "HerrmJ27", "MarxfeHA", "HofmanH2", "BenzinS"]
    ),
)
def animal_split_9_cls_FP():
    
    username = request.basf_auth.get_username()
    

    animal_data_FP=[]

    connection = get_db_connection()
    cursor = connection.cursor()

    page = request.args.get(get_page_parameter(), type=int, default=1)
    limit = 5
    offset = page*limit - limit
        
    outcome = 'FP'
    xai_id = 'xai_5'
    if username=="TanwarA":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_tanwara FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_tanwara FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="HerrmJ27":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_herrmj27 FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_herrmj27 FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="MarxfeHA":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_marxfeha FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_marxfeha FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="HofmanH2":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_hofmanh2 FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_hofmanh2 FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="BenzinS":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_benzins FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_benzins FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    

    for row in result:
        print(row)
        row = convert_base64(row)
        animal_data_FP.append(row)
    
    pagination = Pagination(page=page, page_per=limit, total=total)
    cursor.execute('SELECT feedback FROM feedback_lrp_table;')
    feedbackList = cursor.fetchall()

    return render_template('animal_split_9_cls_FP.html', animal_data_FP=animal_data_FP, feedbackList=feedbackList, pagination=pagination, username=username)

@app.route("/animal_split_9_cls/FN")
@requires_basf_auth(
    authorization_scheme=UsernameListAuthorization(
        ["TanwarA", "HerrmJ27", "MarxfeHA", "HofmanH2", "BenzinS"]
    ),
)
def animal_split_9_cls_FN():
    
    username = request.basf_auth.get_username()
    

    animal_data_FN=[]
    connection = get_db_connection()
    cursor = connection.cursor()

    page = request.args.get(get_page_parameter(), type=int, default=1)
    limit = 5
    offset = page*limit - limit
    
    outcome = 'FN'
    xai_id = 'xai_5'
    if username=="TanwarA":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_tanwara FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_tanwara FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="HerrmJ27":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_herrmj27 FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_herrmj27 FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="MarxfeHA":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_marxfeha FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_marxfeha FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="HofmanH2":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_hofmanh2 FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_hofmanh2 FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    elif username=="BenzinS":
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_benzins FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence;',(outcome,xai_id,))
        total_data = cursor.fetchall()
        total = len(total_data)
        cursor.execute('SELECT image_index, original_files, original_class, predicted_class, confidence, objectiveness, heatmap_c0, heatmap_c1, heatmap_c2, heatmap_c3, heatmap_c4, heatmap_c5, heatmap_c6, heatmap_c7, heatmap_c8, feedback_benzins FROM lrp where outcome=(%s) and xai_id=(%s) order by confidence limit (%s) offset (%s);', (outcome, xai_id, limit, offset,))
        result = cursor.fetchall()
    
    for row in result:
        print(row)
        row = convert_base64(row)
        animal_data_FN.append(row)
    
    pagination = Pagination(page=page, page_per=limit, total=total)
    cursor.execute('SELECT feedback FROM feedback_lrp_table;')
    feedbackList = cursor.fetchall()

    return render_template('animal_split_9_cls_FN.html', animal_data_FN=animal_data_FN, feedbackList=feedbackList, pagination=pagination, username=username)

#insert value from dropdown into the database
@app.route("/add_feedback/<int:id>", methods=['GET', 'POST'])
@requires_basf_auth(
    authorization_scheme=UsernameListAuthorization(
        ["TanwarA", "herrmj27", "MarxfeHA", "HofmanH2", "BenzinS"]
    ),
)
def add_feedback(id):
    username = request.basf_auth.get_username()

    connection = get_db_connection()
    cursor = connection.cursor()
    print("id", id)
    feedback = request.form.get('feedback')
    print("username:", username)
    feedback = request.form.get('feedback')
    if (request.method=='POST' and request.form['submit']=='Submit'):
        if username=="TanwarA":
            print("username:", username)
            cursor.execute("UPDATE lrp SET feedback_tanwara=(%s) WHERE image_index=(%s)", (feedback,id,))
            connection.commit()
            print("Added")
            return redirect(request.referrer)
        elif username=="HerrmJ27":
            print("username:", username)
            cursor.execute("UPDATE lrp SET feedback_herrmj27=(%s) WHERE image_index=(%s)", (feedback,id,))
            connection.commit()
            print("Added")
            return redirect(request.referrer)
        elif username=="MarxfeHA":
            print("username:", username)
            cursor.execute("UPDATE lrp SET feedback_marxfeha=(%s) WHERE image_index=(%s)", (feedback,id,))
            connection.commit()
            print("Added")
            return redirect(request.referrer)
        elif username=="HofmanH2":
            print("username:", username)
            cursor.execute("UPDATE lrp SET feedback_hofmanh2=(%s) WHERE image_index=(%s)", (feedback,id,))
            connection.commit()
            print("Added")
            return redirect(request.referrer)
        elif username=="BenzinS":
            print("username:", username)
            cursor.execute("UPDATE lrp SET feedback_benzins=(%s) WHERE image_index=(%s)", (feedback,id,))
            connection.commit()
            print("Added")
            return redirect(request.referrer)
        else: 
            print("error")
            return redirect(request.referrer)

@app.route("/add_feedback_2/<int:id>", methods=['GET', 'POST'])
@requires_basf_auth(
    authorization_scheme=UsernameListAuthorization(
        ["TanwarA", "herrmj27", "MarxfeHA", "HofmanH2", "BenzinS"]
    ),
)
def add_feedback_2(id):
    username = request.basf_auth.get_username()
    connection = get_db_connection()
    cursor = connection.cursor()
    print("id", id)
    print("username:", username)
    feedback = request.form.get('feedback')
    if (request.method=='POST' and request.form['submit']=='Submit'):
        if username=="HerrmJ27":
            print("username:", request.method)
            print("username:", request.form['submit'])
            cursor.execute("UPDATE lrp_2_classes SET feedback_herrmj27=(%s) WHERE image_index=(%s)", (feedback,id,))
            connection.commit()
            print("Added")
            return redirect(request.referrer)
        elif username=="MarxfeHA":
            print("username:", request.method)
            print("username:", request.form['submit'])
            cursor.execute("UPDATE lrp_2_classes SET feedback_marxfeha=(%s) WHERE image_index=(%s)", (feedback,id,))
            connection.commit()
            print("Added")
            return redirect(request.referrer)
        elif username=="TanwarA":
            print("username_TanwarA", request.method)
            print("username_TanwarA", request.form['submit'])
            cursor.execute("UPDATE lrp_2_classes SET feedback_tanwara=(%s) WHERE image_index=(%s)", (feedback,id,))
            connection.commit()
            print("Added")
            return redirect(request.referrer)
        elif username=="HofmanH2":
            print("username:", request.method)
            print("username:", request.form['submit'])
            cursor.execute("UPDATE lrp_2_classes SET feedback_hofmanh2=(%s) WHERE image_index=(%s)", (feedback,id,))
            connection.commit()
            print("Added")
            return redirect(request.referrer)
        elif username=="BenzinS":
            print("username:", request.method)
            print("username:", request.form['submit'])
            cursor.execute("UPDATE lrp_2_classes SET feedback_benzins=(%s) WHERE image_index=(%s)", (feedback,id,))
            connection.commit()
            print("Added")
            return redirect(request.referrer)
        else: 
            print("error")
            return redirect(request.referrer)

#insert value from the input into the database
@app.route("/update_feedback/<int:id>", methods=['GET', 'POST'])
@requires_basf_auth(
    authorization_scheme=UsernameListAuthorization(
        ["TanwarA", "HerrmJ27", "MarxfeHA", "HofmanH2", "BenzinS"]
    ),
)
def update_feedback(id):
    username = request.basf_auth.get_username()
    # username = "TanwarA"
    

    connection = get_db_connection()
    cursor = connection.cursor()
    print("id", id)
    feedback = request.form.get('feedback')
    print("feedback", feedback)
    cursor.execute("SELECT feedback from feedback_lrp_table")
    all_feedback = [r[0] for r in cursor.fetchall()]
    print("all_feedback",all_feedback)
    result = any(ele==feedback for ele in all_feedback)
    print(result)
    print(request.form['submit'])
    if (result!=True and request.form['submit']=='Update'):
        if username=="TanwarA":
            print("username:", username)
            cursor.execute("UPDATE lrp SET feedback_tanwara=(%s) WHERE image_index=(%s)", (feedback,id,))
            connection.commit()
            cursor.execute("INSERT INTO feedback_lrp_table(feedback) VALUES (%s)", (feedback,))		
            connection.commit()
            print("updated")
        elif username=="HerrmJ27":
            print("username:", username)
            cursor.execute("UPDATE lrp SET feedback_herrmj27=(%s) WHERE image_index=(%s)", (feedback,id,))
            connection.commit()
            cursor.execute("INSERT INTO feedback_lrp_table(feedback) VALUES (%s)", (feedback,))		
            connection.commit()
            print("updated")
        elif username=="MarxfeHA":
            print("username:", username)
            cursor.execute("UPDATE lrp SET feedback_marxfeha=(%s) WHERE image_index=(%s)", (feedback,id,))
            connection.commit()
            cursor.execute("INSERT INTO feedback_lrp_table(feedback) VALUES (%s)", (feedback,))		
            connection.commit()
            print("updated")
        elif username=="HofmanH2":
            print("username:", username)
            cursor.execute("UPDATE lrp SET feedback_hofmanh2=(%s) WHERE image_index=(%s)", (feedback,id,))
            connection.commit()
            cursor.execute("INSERT INTO feedback_lrp_table(feedback) VALUES (%s)", (feedback,))		
            connection.commit()
            print("updated")
        elif username=="BenzinS":
            print("username:", username)
            cursor.execute("UPDATE lrp SET feedback_benzins=(%s) WHERE image_index=(%s)", (feedback,id,))
            connection.commit()
            cursor.execute("INSERT INTO feedback_lrp_table(feedback) VALUES (%s)", (feedback,))		
            connection.commit()
            print("updated")
    return redirect(request.referrer)

@app.route("/update_feedback_2/<int:id>", methods=['GET', 'POST'])
@requires_basf_auth(
    authorization_scheme=UsernameListAuthorization(
        ["TanwarA", "HerrmJ27", "MarxfeHA", "HofmanH2", "BenzinS"]
    ),
)
def update_feedback_2(id):
    
    username = request.basf_auth.get_username()    

    connection = get_db_connection()
    cursor = connection.cursor()
    print("id", id)
    feedback = request.form.get('feedback')
    print("feedback", feedback)
    cursor.execute("SELECT feedback from feedback_lrp_table")
    all_feedback = [r[0] for r in cursor.fetchall()]
    print("all_feedback",all_feedback)
    result = any(ele in feedback for ele in all_feedback)
    print(result)
    print(request.form['submit'])
    if (result!=True and request.form['submit']=='Update'):
        if username=="TanwarA":
            print("username:", username)
            cursor.execute("UPDATE lrp_2_classes SET feedback_tanwara=(%s) WHERE image_index=(%s)", (feedback,id,))
            connection.commit()
            cursor.execute("INSERT INTO feedback_lrp_table(feedback) VALUES (%s)", (feedback,))		
            connection.commit()
            print("updated")
        elif username=="HerrmJ27":
            print("username:", username)
            cursor.execute("UPDATE lrp_2_classes SET feedback_herrmj27=(%s) WHERE image_index=(%s)", (feedback,id,))
            connection.commit()
            cursor.execute("INSERT INTO feedback_lrp_table(feedback) VALUES (%s)", (feedback,))		
            connection.commit()
            print("updated")
        elif username=="MarxfeHA":
            print("username:", username)
            cursor.execute("UPDATE lrp_2_classes SET feedback_marxfeha=(%s) WHERE image_index=(%s)", (feedback,id,))
            connection.commit()
            cursor.execute("INSERT INTO feedback_lrp_table(feedback) VALUES (%s)", (feedback,))		
            connection.commit()
            print("updated")
        elif username=="HofmanH2":
            print("username:", username)
            cursor.execute("UPDATE lrp_2_classes SET feedback_hofmanh2=(%s) WHERE image_index=(%s)", (feedback,id,))
            connection.commit()
            cursor.execute("INSERT INTO feedback_lrp_table(feedback) VALUES (%s)", (feedback,))		
            connection.commit()
            print("updated")
        elif username=="BenzinS":
            print("username:", username)
            cursor.execute("UPDATE lrp_2_classes SET feedback_benzins=(%s) WHERE image_index=(%s)", (feedback,id,))
            connection.commit()
            cursor.execute("INSERT INTO feedback_lrp_table(feedback) VALUES (%s)", (feedback,))		
            connection.commit()
            print("updated")
    return redirect(request.referrer)

@app.route("/dashboard")
@requires_basf_auth(
    authorization_scheme=UsernameListAuthorization(
        ["TanwarA", "HerrmJ27", "MarxfeHA", "HofmanH2", "BenzinS"]
    ),
)
def dashboard():
    
    username = request.basf_auth.get_username()
    # username = "TanwarA"
    
    connection = get_db_connection()

    #random split 2 classes
    cursor = connection.cursor()
    ###################True positive#############################
    outcome='TP'
    xai_id='xai_6'
    if username=="TanwarA":
        cursor.execute('SELECT original_class, feedback_marxfeha FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="HerrmJ27":
        cursor.execute('SELECT original_class, feedback_herrmj27 FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="MarxfeHA":
        cursor.execute('SELECT original_class, feedback_marxfeha FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="HofmanH2":
        cursor.execute('SELECT original_class, feedback_hofmanh2 FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="BenzinS":
        cursor.execute('SELECT original_class, feedback_benzins FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    
    dict = Counter(original_class)
    df = pd.DataFrame(dict.keys(), columns=['original_class', 'feedback'])
    df['frequency'] = dict.values()
    print(df)

    fig = px.bar(df, x='feedback', y='frequency', color='original_class', barmode='group')
    graphJSON_random_2_tp = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    #random split 9 classes
    cursor = connection.cursor()
    xai_id='xai_3'
    if username=="TanwarA":
        cursor.execute('SELECT original_class, feedback_marxfeha FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="HerrmJ27":
        cursor.execute('SELECT original_class, feedback_herrmj27 FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="MarxfeHA":
        cursor.execute('SELECT original_class, feedback_marxfeha FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="HofmanH2":
        cursor.execute('SELECT original_class, feedback_hofmanh2 FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="BenzinS":
        cursor.execute('SELECT original_class, feedback_benzins FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    
    dict = Counter(original_class)
    df = pd.DataFrame(dict.keys(), columns=['original_class', 'feedback'])
    df['frequency'] = dict.values()
    print(df)

    fig = px.bar(df, x='feedback', y='frequency', color='original_class', barmode='group')
    graphJSON_random_9_tp = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    #animal split 2 classes
    cursor = connection.cursor()
    xai_id='xai_4'
    if username=="TanwarA":
        cursor.execute('SELECT original_class, feedback_marxfeha FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="HerrmJ27":
        cursor.execute('SELECT original_class, feedback_herrmj27 FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="MarxfeHA":
        cursor.execute('SELECT original_class, feedback_marxfeha FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="HofmanH2":
        cursor.execute('SELECT original_class, feedback_hofmanh2 FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="BenzinS":
        cursor.execute('SELECT original_class, feedback_benzins FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    
    dict = Counter(original_class)
    df = pd.DataFrame(dict.keys(), columns=['original_class', 'feedback'])
    df['frequency'] = dict.values()
    print(df)

    fig = px.bar(df, x='feedback', y='frequency', color='original_class', barmode='group')
    graphJSON_animal_2_tp = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    #animal split 9 classes
    cursor = connection.cursor()
    xai_id='xai_5'
    if username=="TanwarA":
        cursor.execute('SELECT original_class, feedback_marxfeha FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="HerrmJ27":
        cursor.execute('SELECT original_class, feedback_herrmj27 FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="MarxfeHA":
        cursor.execute('SELECT original_class, feedback_marxfeha FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="HofmanH2":
        cursor.execute('SELECT original_class, feedback_hofmanh2 FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="BenzinS":
        cursor.execute('SELECT original_class, feedback_benzins FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    
    dict = Counter(original_class)
    df = pd.DataFrame(dict.keys(), columns=['original_class', 'feedback'])
    df['frequency'] = dict.values()
    print(df)

    fig = px.bar(df, x='feedback', y='frequency', color='original_class', barmode='group')
    graphJSON_animal_9_tp = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    ###################False positive#############################
    outcome='FP'
    xai_id='xai_6'
    if username=="TanwarA":
        cursor.execute('SELECT original_class, feedback_marxfeha FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="HerrmJ27":
        cursor.execute('SELECT original_class, feedback_herrmj27 FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="MarxfeHA":
        cursor.execute('SELECT original_class, feedback_marxfeha FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="HofmanH2":
        cursor.execute('SELECT original_class, feedback_hofmanh2 FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="BenzinS":
        cursor.execute('SELECT original_class, feedback_benzins FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    
    dict = Counter(original_class)
    df = pd.DataFrame(dict.keys(), columns=['original_class', 'feedback'])
    df['frequency'] = dict.values()
    print(df)

    fig = px.bar(df, x='feedback', y='frequency', color='original_class', barmode='group')
    graphJSON_random_2_fp = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    #random split 9 classes
    cursor = connection.cursor()
    xai_id='xai_3'
    if username=="TanwarA":
        cursor.execute('SELECT original_class, feedback_marxfeha FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="HerrmJ27":
        cursor.execute('SELECT original_class, feedback_herrmj27 FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="MarxfeHA":
        cursor.execute('SELECT original_class, feedback_marxfeha FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="HofmanH2":
        cursor.execute('SELECT original_class, feedback_hofmanh2 FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="BenzinS":
        cursor.execute('SELECT original_class, feedback_benzins FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    
    dict = Counter(original_class)
    df = pd.DataFrame(dict.keys(), columns=['original_class', 'feedback'])
    df['frequency'] = dict.values()
    print(df)

    fig = px.bar(df, x='feedback', y='frequency', color='original_class', barmode='group')
    graphJSON_random_9_fp = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    #animal split 2 classes
    cursor = connection.cursor()
    xai_id='xai_4'
    if username=="TanwarA":
        cursor.execute('SELECT original_class, feedback_marxfeha FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="HerrmJ27":
        cursor.execute('SELECT original_class, feedback_herrmj27 FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="MarxfeHA":
        cursor.execute('SELECT original_class, feedback_marxfeha FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="HofmanH2":
        cursor.execute('SELECT original_class, feedback_hofmanh2 FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="BenzinS":
        cursor.execute('SELECT original_class, feedback_benzins FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    
    dict = Counter(original_class)
    df = pd.DataFrame(dict.keys(), columns=['original_class', 'feedback'])
    df['frequency'] = dict.values()
    print(df)

    fig = px.bar(df, x='feedback', y='frequency', color='original_class', barmode='group')
    graphJSON_animal_2_fp = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    #animal split 9 classes
    cursor = connection.cursor()
    xai_id='xai_5'
    if username=="TanwarA":
        cursor.execute('SELECT original_class, feedback_marxfeha FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="HerrmJ27":
        cursor.execute('SELECT original_class, feedback_herrmj27 FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="MarxfeHA":
        cursor.execute('SELECT original_class, feedback_marxfeha FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="HofmanH2":
        cursor.execute('SELECT original_class, feedback_hofmanh2 FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="BenzinS":
        cursor.execute('SELECT original_class, feedback_benzins FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    
    dict = Counter(original_class)
    df = pd.DataFrame(dict.keys(), columns=['original_class', 'feedback'])
    df['frequency'] = dict.values()
    print(df)

    fig = px.bar(df, x='feedback', y='frequency', color='original_class', barmode='group')
    graphJSON_animal_9_fp = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    ###################False negative#############################
    outcome='FN'
    xai_id='xai_6'
    if username=="TanwarA":
        cursor.execute('SELECT original_class, feedback_marxfeha FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="HerrmJ27":
        cursor.execute('SELECT original_class, feedback_herrmj27 FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="MarxfeHA":
        cursor.execute('SELECT original_class, feedback_marxfeha FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="HofmanH2":
        cursor.execute('SELECT original_class, feedback_hofmanh2 FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="BenzinS":
        cursor.execute('SELECT original_class, feedback_benzins FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    
    dict = Counter(original_class)
    df = pd.DataFrame(dict.keys(), columns=['original_class', 'feedback'])
    df['frequency'] = dict.values()
    print(df)

    fig = px.bar(df, x='feedback', y='frequency', color='original_class', barmode='group')
    graphJSON_random_2_fn = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    #random split 9 classes
    cursor = connection.cursor()
    xai_id='xai_3'
    if username=="TanwarA":
        cursor.execute('SELECT original_class, feedback_marxfeha FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="HerrmJ27":
        cursor.execute('SELECT original_class, feedback_herrmj27 FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="MarxfeHA":
        cursor.execute('SELECT original_class, feedback_marxfeha FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="HofmanH2":
        cursor.execute('SELECT original_class, feedback_hofmanh2 FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="BenzinS":
        cursor.execute('SELECT original_class, feedback_benzins FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    
    dict = Counter(original_class)
    df = pd.DataFrame(dict.keys(), columns=['original_class', 'feedback'])
    df['frequency'] = dict.values()
    print(df)

    fig = px.bar(df, x='feedback', y='frequency', color='original_class', barmode='group')
    graphJSON_random_9_fn = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    #animal split 2 classes
    cursor = connection.cursor()
    xai_id='xai_4'
    if username=="TanwarA":
        cursor.execute('SELECT original_class, feedback_marxfeha FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="HerrmJ27":
        cursor.execute('SELECT original_class, feedback_herrmj27 FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="MarxfeHA":
        cursor.execute('SELECT original_class, feedback_marxfeha FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="HofmanH2":
        cursor.execute('SELECT original_class, feedback_hofmanh2 FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="BenzinS":
        cursor.execute('SELECT original_class, feedback_benzins FROM lrp_2_classes where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    
    dict = Counter(original_class)
    df = pd.DataFrame(dict.keys(), columns=['original_class', 'feedback'])
    df['frequency'] = dict.values()
    print(df)

    fig = px.bar(df, x='feedback', y='frequency', color='original_class', barmode='group')
    graphJSON_animal_2_fn = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    #animal split 9 classes
    cursor = connection.cursor()
    xai_id='xai_5'
    if username=="TanwarA":
        cursor.execute('SELECT original_class, feedback_marxfeha FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="HerrmJ27":
        cursor.execute('SELECT original_class, feedback_herrmj27 FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="MarxfeHA":
        cursor.execute('SELECT original_class, feedback_marxfeha FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="HofmanH2":
        cursor.execute('SELECT original_class, feedback_hofmanh2 FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    elif username=="BenzinS":
        cursor.execute('SELECT original_class, feedback_benzins FROM lrp where xai_id=(%s) and outcome=(%s) order by image_index;',(xai_id,outcome,))
        original_class =  cursor.fetchall()
    
    dict = Counter(original_class)
    df = pd.DataFrame(dict.keys(), columns=['original_class', 'feedback'])
    df['frequency'] = dict.values()
    print(df)

    fig = px.bar(df, x='feedback', y='frequency', color='original_class', barmode='group')
    graphJSON_animal_9_fn = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('dashboard.html', graphJSON_random_2_tp=graphJSON_random_2_tp, 
                           graphJSON_random_9_tp=graphJSON_random_9_tp, 
                           graphJSON_animal_2_tp=graphJSON_animal_2_tp, 
                           graphJSON_animal_9_tp=graphJSON_animal_9_tp, 
                           graphJSON_random_2_fp=graphJSON_random_2_fp,
                           graphJSON_random_9_fp=graphJSON_random_9_fp,
                           graphJSON_animal_2_fp=graphJSON_animal_2_fp,
                           graphJSON_animal_9_fp=graphJSON_animal_9_fp,
                           graphJSON_random_2_fn=graphJSON_random_2_fn,
                           graphJSON_random_9_fn=graphJSON_random_9_fn,
                           graphJSON_animal_2_fn=graphJSON_animal_2_fn,
                           graphJSON_animal_9_fn=graphJSON_animal_9_fn,
                           username=username)

if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0")
