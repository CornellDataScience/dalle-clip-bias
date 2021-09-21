import os
import json
import shutil
import pandas as pd
import argparse

def parse_annotations(file, data_dir):
    """
      Parse Captioning File

      :param file: Annotation file to parse 
      :type file: String
      :param data_dir: Directory to dataset (either train or val)
      :type data_dir: String
    """
    dir = 'train-dataset'
    if not os.path.isdir(dir):
        os.mkdir(dir)

    f = open(file, 'r')
    j = json.load(f)
    images = pd.DataFrame(j['images'], )
    annotations = pd.DataFrame(j['annotations'])
    print('Moving data...')
    for i in range(len(images)):
      try:
        image = images.iloc[i]
        image_id = image['id']
        res = annotations[annotations['image_id'] == image_id]

        # get the name of image in train2014 and parse it , 
        # i.e. 'COCO_train2014_000000057870.jpg' -> 'COCO_train2014_000000057870'
        image_name = image['file_name'].rpartition('.')[0]

        # Move image from one place to another
        target_file = os.path.join(dir, image_name)
        print(f'Saving to {target_file}')
        source_location = os.path.join(data_dir, image['file_name'])

        shutil.move(source_location, f'{target_file}.jpg')
        
        # define the path of parsed caption
        f = open(f'{target_file}.txt', 'w+')
        f.write(' '.join(res['caption'].tolist()))
        f.close()
      except Exception as e:
        print(f'An Exception has occured: {e}')


if __name__ == '__main__':
  parse_annotations(f'annotations/captions_train2014.json', 'train2014')
  parse_annotations(f'annotations/captions_val2014.json', 'val2014')
