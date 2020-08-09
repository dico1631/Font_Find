import tensorflow as tf
from django.shortcuts import render
from django.views.generic import TemplateView
from .forms import ImageForm
from .main import detect
from imgur import imguruploader
import cv2
import numpy as np
from PIL import Image
import json

class FontDetectorView(TemplateView):
  # 생성자
  def __init__(self):
    self.params = {'result_list':[],
                   'result_name':"",
                   'result':[],
                   'str_list':[],
                   'form': ImageForm()}
    self.imgur = imguruploader()
    self.imgur.setpath('imgtemp/target.jpg')
    self.imgur.setConfig(None, 'Target', '', '')

  # GET request (index.html 파일 초기 표시)
  def get(self, req):
    return render(req, 'font_detect/index.html', self.params)

  # POST request (index.html 파일에 결과 표시)
  def post(self, req):
    # POST 메소드에 의해서 전달되는 FORM DATA 
    form = ImageForm(req.POST, req.FILES)
    # FORM DATA 에러 체크 
    if not form.is_valid():  
      raise ValueForm('invalid form')
    # FORM DATA에서 이미지 파일 얻기 
    image = form.cleaned_data['image']
    # 이미지 파일을 지정해서 얼굴 인식
    # image_url = "https://postfiles.pstatic.net/MjAyMDA0MDlfMTQx/MDAxNTg2NDA3MTg5NDc1.Kuc8FIqt1-LpjVjCzUwengPd8GlqHPKkWG0Jd1W5avEg.82sPT_dhaFTCF0TQKnsNBb26fn-tWzbPcjMuehAEtwEg.JPEG.cancel9/image1.jpg"

    # image_url = "https://user-images.githubusercontent.com/56625356/80015883-5841d580-850d-11ea-8060-e2e3169f96df.jpg"

    origin_image = np.asarray(Image.open(image))

    cv2.imwrite('imgtemp/target.jpg', origin_image)
    self.imgur.imgurUpload()
    items = self.imgur.client.get_account_images('adianktw')

    image_url = "https://user-images.githubusercontent.com/56625356/80015519-d6ea4300-850c-11ea-9ee0-3178f12ffdcd.jpg"

    result_list, result_name, result_img, result = detect(items[0].link)
    print("아아아아"*20)
    print(result)
    # 얼굴 분류된 결과 저장
    str_list = []
    for idx in range(len(result_img)):
      str_html = ''
      str_html +=  f'<img src="data:image/png;base64,{result_img[idx]}" />'
      str_html += f'<div class="progress"><div class="progress-bar progress-bar-success progress-bar-striped" role="progressbar"aria-valuenow="40"aria-valuemin="0" aria-valuemax="100" style="width:{result[idx][0]}%">배달의 민족체</div></div>'
      str_html += f'<div class="progress"><div class="progress-bar progress-bar-info progress-bar-striped" role="progressbar"aria-valuenow="50" ria-valuemin="0" aria-valuemax="100" style="width:{result[idx][1]}%">카페24 폰트</div></div>'
      str_html += f'<div class="progress"><div class="progress-bar progress-bar-warning progress-bar-striped" role="progressbar"aria-valuenow="60"aria-valuemin="0" aria-valuemax="100" style="width:{result[idx][2]}%">조선일보명조체</div></div>'
      str_html += f'<div class="progress"><div class="progress-bar progress-bar-danger progress-bar-striped" role="progressbar"aria-valuenow="70" ria-valuemin="0" aria-valuemax="100" style="width:{result[idx][3]}%">마포구민체</div></div>'
      str_html += f'<div class="progress"><div class="progress-bar progress-bar-success progress-bar-striped" role="progressbar"aria-valuenow="40" aria-valuemin="0" aria-valuemax="100" style="width:{result[idx][4]}%">나눔손글씨펜</div></div>'
      str_html += f'<div class="progress"><div class="progress-bar progress-bar-info progress-bar-striped" role="progressbar"aria-valuenow="50" ria-valuemin="0" aria-valuemax="100" style="width:{result[idx][5]}%">나눔스퀘어</div></div>'
      str_list.append(str_html)

    self.params['result_list'], self.params['result_name'], self.params['result'], self.params['str_list'] = result_list, result_name, result, str_list

    # 페이지에 화면 표시
    return render(req, 'font_detect/index.html', self.params)