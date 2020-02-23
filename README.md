# [Project] DogClassification 
> Classification Models for Korean Dog Breeds

## dataset

#### 1. Dog Breed Standard: 한국 애견 협회(견종 표준)

* [한국 애견 협회](https://www.kkc.or.kr/megazine/megazine_02.html)

#### 2. dataset

* [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html)
* [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification/data)
---
## [Data preprocessing](https://github.com/shiney5213/Project-DogClassification/blob/master/data_preprocessing/data_preprocessing.md)

>  Making a dataset based on the dog breed standard of the Korean Dog Association
>  
#### 1. Dog Breed Identification data set 
- 데이터셋이 이미지와 라벨 text문서가  따로 있기 때문에, 품종별 폴더에 이미지 라벨별로 나누는 작업 진행

#### 2. Merge datasets and Crawrling
- Dog Breed Identification, Stanford Dogs Dataset 같은 품종끼리 합치기
- 한국 애견 협회 품종 기준으로 위의 데이터 셋에 없는 품종은 크롤링하여 데이터 수집

#### 3. Check valid images
- 이미지 중 오류가 있는 이미지가 있는지 확인
- 사용할 수 없는 이미지 처리

#### 4. Image crop by YOLO3
- 이미지 중 사람과 같이 찍거나, 두마리가 있거나, 강아지가 너무 작은 사진 등 학습에 적합하지 않은 이미지가 많음.
- YOLO3 모델을 이용하여  dog로 deteching 한 box만 잘른 결과 모두 34,767장 이미지 모음
- reference: : [darknet](https://pjreddie.com/darknet/) 참고

#### 5. [Image crop by SSD512](
- weight download: [pretrained weight](https://drive.google.com/file/d/1a-64b6y6xsQr5puUsHX_wxI1orQDercM/view)
- SSD512 모델을 이용하여  dog로 deteching 한 box만 잘른 결과 모두 39,282장 이미지 모음
- reference: [SSD-Object-Detection](https://github.com/InsiderPants/SSD-Object-Detection)
#### 6. Merge images 
- 위의 5,6의 데이터셋 중 강아지를 잘 찾아서 적절하게 자른 이미지 선택
- deteching 못한 이미지는 수작업으로 잘라 모두 35,176장의 이미지 데이터셋 완성
- SSD512모델이 YOLO3보다 느리지만, 성능이 좋다고 알려져있는데,  이 데이터셋에서는 YOLO3가 더 좋은 것 같음. 

#### 7. [Split images trainset and testset]()
- 전체 데이터를 train(0.8), test set(0.2)으로 나눔( train set:  28,070장, test set: 7,106장)
- 전체 데이터 수가 200장이 안되는 강아지 : 추후 모델링 결과를 살펴보고 데이터셋 추가 확보 여부 결정

---
## train/ test

#### 1.  [1st_train](https://github.com/shiney5213/Project-DogClassification/blob/master/train%2C test/200216_train_1.py) : [accuracy 75.94%](https://github.com/shiney5213/Project-DogClassification/blob/master/train%2C test/1. train_1.md)
- keras의 inceptionV3 모델을 pretrain model로 사용
- epoch 4~5부터 overfitting 발생
- dropout, Regularization, 모델의 복잡도를 줄이는 방법 등을 적용해봐야겠음.

#### 2. [2nd_train](https://github.com/shiney5213/Project-DogClassification/blob/master/train%2C%20test/200218_train_1.py) : [accuracy : 72.38%](https://github.com/shiney5213/Project-DogClassification/blob/master/train%2C%20test/1.%20train_2.md)
- train_1의 overfitting문제를 해결하기 위해 dropout layer(0.5)추가
- train_1보다 정확도는 낮게 나왔지만, 오버피팅이 개선되었음.
- Confusion Metrics, Precision, Recall 등의 다양한 평가 지표를 보면서 모델을 개선해야겠음.

#### 3. [3th_train](https://github.com/shiney5213/Project-DogClassification/blob/master/model/3.200219_1/200219_train_1.py) : accuracy: 1.97%https://github.com/shiney5213/Project-DogClassification/blob/master/model/3.200219_1/README.md)
- 모델을 InceptionResnetV2로 바꾼 후, 성능이 너무 낮아짐. 
- loss는 계속 낮아지는 것으로 보아, 학습은 진행 중이지만, 성능이 개선되지 않음.
- local minima에 빠진 것으로 판단되어, optimizer을 바꾸어 진행해보기로 함.

#### 4. [4th_train](https://github.com/shiney5213/Project-DogClassification/blob/master/model/4.200223_1/200223_train_1.py) : accuracy: 2.60%https://github.com/shiney5213/Project-DogClassification/tree/master/model/4.200223_1)
- optimiser을 Adam에서 Rdam으로 바꾸어 진행
- 성능이 크게 좋아지지 않음.
- 모델의 구조를 조금씩 변경해보고, 성능이 나아지지 않으면 다시 InceptionV3모델을 사용해야겠음.


#### 5. [5th_train]: accuracy: %

---
## Reference

- [SSD-Object-Detection](https://github.com/InsiderPants/SSD-Object-Detection)

