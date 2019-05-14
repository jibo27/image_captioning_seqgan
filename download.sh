cd /content/gdrive/My Drive/gitrepo/image_captioning_seqgan
if [ -f "data/resized2014.tar.gz" ]
then :
else
    cd /content
    [ -f train2014.zip ] || wget http://images.cocodataset.org/zips/train2014.zip
    [ -f val2014.zip ] || wget http://images.cocodataset.org/zips/val2014.zip

    [ -d train2014 ] || unzip -q train2014.zip
    [ -d val2014 ] || unzip -q val2014.zip

    rm train2014.zip 
    rm val2014.zip 
fi

cd /content
[ -f annotations_trainval2014.zip ] || wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
[ -d annotations ] || unzip -q annotations_trainval2014.zip
rm captions_train-val2014.zip

pip install nltk
python -c 'import nltk'
python -c 'nltk.download("punkt")'

git clone https://github.com/pdollar/coco.git 
cd coco/PythonAPI 
make 
python setup.py build
python setup.py install

# os.chdir("/content/coco/PythonAPI")
# !make
# !python setup.py build
# !python setup.py install

