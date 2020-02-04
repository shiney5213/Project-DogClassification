# [Project] DogClassification 
> Classification Models for Korean Dog Breeds

## dataset

### Dog Breed Standard: 한국 애견 협회(견종 표준)

* [한국 애견 협회](https://www.kkc.or.kr/megazine/megazine_02.html)

### dataset

* [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html)
* [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification/data)

## Data preprocessing

>  Making a dataset based on the dog breed standard of the Koreadn Dog Association
* [1. dividing Ientification datasets by directory]
* [2. Merge dataset Stanford Dogs Dataset and Dog Breed Identification]
* [3.Crawrling images by irawler]
* [4.Check valid images]
* [5.Image crop by Yolo3]
   *  cfg, weight download: [YOLOv3-416](https://pjreddie.com/darknet/yolo/)
* [6.Image crop by SSD512]
   * weight download: [pretrained weight](https://drive.google.com/file/d/1a-64b6y6xsQr5puUsHX_wxI1orQDercM/view)
*  [7.Split images trainset and testset]
*  



## Reference

- [SSD-Object-Detection](https://github.com/InsiderPants/SSD-Object-Detection)

