cd /content

[ -f train2014.zip ] || wget http://images.cocodataset.org/zips/train2014.zip
[ -f val2014.zip ] || wget http://images.cocodataset.org/zips/val2014.zip
[ -f annotations_trainval2014.zip ] || wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip

[ -d train2014 ] || unzip -q train2014.zip
[ -d val2014 ] || unzip -q val2014.zip
[ -d annotations ] || unzip -q annotations_trainval2014.zip

rm ./data/captions_train-val2014.zip
rm ./data/train2014.zip 
rm ./data/val2014.zip 

!pip install nltk
import nltk
nltk.download('punkt')

git clone https://github.com/pdollar/coco.git 
cd coco/PythonAPI 
make 
python setup.py build
python setup.py install

# os.chdir("/content/coco/PythonAPI")
# !make
# !python setup.py build
# !python setup.py install

