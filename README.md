## How to use
1. Install pycoco first
```
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI/
make
python setup.py build
python setup.py install

chmod +x download.sh
./download.sh
```

2. Configure `settings.py`

3. Preprocess data
```
python build_vocab.py # produce vocab.pkl file
python resize.py # resize the image for ResNet
```
4. Train the model
```
python train.py
```

5. Evaluate model
```
python sample.py
```

## Inference
![surf](data/surf.jpg)

+ a person riding a wave in the ocean .
+ a person riding a wave on a wave .
+ a person riding a wave on a surfboard .

![giraffe](data/giraffe.png)

+ a giraffe standing in a giraffe standing in a field
+ a giraffe standing in a giraffe standing in a field .
+ a giraffe standing in a giraffe standing next to a tree .

## Scores
8208 images, MSCOCO test images
```
Bleu_1: 0.381045
Bleu_2: 0.223908
Bleu_3: 0.139023
Bleu_4: 0.090117
METEOR: 0.152553
ROUGE_L: 0.392893
CIDEr: 0.680351
```
