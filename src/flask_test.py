# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 10:31:15 2019

@author: zhangdong0626
"""
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
app = Flask(__name__)  # flask实例
from meeting_det import meeting_det #导入检测模型
import json

def test(imagepath):
    return imagepath,imagepath
# 设置HTTP请求方法
@app.route("/get_meeting_info", methods = ['POST', "GET"])
def get_meeting_info(): 
    if request.method == 'POST': 
        # args = request.args
        # print('args:', args)
        new_img1=request.files['file']  # 'file'对应前端表单name属性
        new_img1.save(os.path.join(os.getcwd(),'media/')+ secure_filename(new_img1.filename)) 
        imagepath=os.path.join(os.getcwd(),'media/'+secure_filename(new_img1.filename))   # 获取图片地址
        final_result_dict = {}
        concentrate_rate, activity_rate = meeting_det(imagepath) #返回需要的数据
        final_result_dict["concentrate_rate"] = concentrate_rate
        final_result_dict["activity_rate"] = activity_rate
#        return jsonify(str(final_result_dict))
        return json.dumps(final_result_dict, ensure_ascii=False) #将数据打包成json文件
    return render_template('uploadimg.html') #网页界面设计


@app.route('/api_get_meeting_info?url=<path:url>')
def api_get_meeting_info(url): 
    imagepath = url[1:]
    final_result_dict = {}
    concentrate_rate, activity_rate = meeting_det(imagepath)
    final_result_dict["concentrate_rate"] = concentrate_rate
    final_result_dict["activity_rate"] = activity_rate
    return json.dumps(final_result_dict, ensure_ascii=False)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10100, debug=False)
