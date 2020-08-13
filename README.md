# Font_Find

- 실행 방법

  - python 3.6 이상이 필요합니다. (아나콘다 가상환경 사용을 추천합니다.)

    > django의 secrets 모듈이 python 3.6부터 추가되어서 3.6 이상 버전을 사용해야합니다.

  - `pip install -r requirements.txt` 로 requirements.txt의 패키지들을 모두 설치합니다.

    > tensorflow 2.0 이상 버전을 사용하면 model not found 에러가 뜹니다.

  -  `python manage.py runserver` 를 입력해 프로젝트를 실행합니다.



### 화면 및 기능 설명

- 메인 화면

![image](https://user-images.githubusercontent.com/58927491/90126017-efbd8380-dd9d-11ea-9bf9-c11a9f35c0b7.png)

- 기능 설명

  1. 찾고 싶은 폰트가 있는 사진을 업로드 한다.

  2. '분석' 버튼을 누른다.

     > 주의사항 : 사진에서 글자 부분을 인식하는 imgur 모듈은 가로 글자만 인식 가능하다.

     ![image](https://user-images.githubusercontent.com/58927491/90126219-4460fe80-dd9e-11ea-92ab-992ed971203c.png)

- 결과 화면

  - 사진 속 각 폰트가 어떤 폰트에 가까운지 확률로 나타남.

  ![image](https://user-images.githubusercontent.com/58927491/90127437-23011200-dda0-11ea-8ba5-d561387f7c13.png)

