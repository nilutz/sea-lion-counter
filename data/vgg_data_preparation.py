# VGG Cell Dataset: http://www.robots.ox.ac.uk/~vgg/research/counting/cells.zip
# run vgg_generate_coords first

import sys
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import time
import math
import cv2
import time
import csv
import flock


# Make paths absolute and independent from where the python script is called.
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

from vgg_generate_coords import CellData

dry_run = True
cls_nb = 1

tile_size = 128
tile_margin = 32
coords_margin = 128
scale_factor = 1.0
num_processes = 1

background_rate = 0.02
scale_factor = 1.0

storage_split_nb = 10
clear_files = True
recompute_coords = False   
train_val_split = 0.1
train_suffix = 'train'
val_suffix = 'val'

data_dir = os.path.join(script_dir)
input_dir = os.path.join(script_dir, 'cells')
debug_dir = os.path.join(script_dir,'..' ,'debug')

tf.gfile.MakeDirs(debug_dir)

coords_file = os.path.join(script_dir, 'outdir/coords.csv')

cld = CellData(input_dir, data_dir)


#aug_rot = (0, 1, 2, 3)
#aug_scale = (1.0, 1. / 1.1, 1.1)
#aug_flip = (False, True)

#no augmentation
aug_rot = (0)
aug_scale = (1.0)
aug_flip = (False)


aug_params =[aug_rot, aug_scale, aug_flip]

#aug_params = [(rot, scale, flip) for rot in aug_rot \
#                                 for scale in aug_scale \
#                                 for flip in aug_flip]

if recompute_coords:
  print('Recomputing coords')
  cld.save_coords()

  # Error analysis

  #TODO
  #sld.verbosity = VERBOSITY.VERBOSE
  #tid_counts = sld.count_coords(sld.tid_coords)
  #rmse, frac = sld.rmse(tid_counts)

  #print('\nRMSE: %f' % rmse)
  #print('Error frac: %f' % frac)

cld_coords = pd.read_csv(coords_file)
cls_counts = [len(cld_coords.loc[cld_coords['cls'] == cls_idx]) for cls_idx in range(cls_nb)]

actual_tile_size = round(tile_size * (1. / scale_factor))
actual_tile_margin = round(tile_margin * (1. / scale_factor))
actual_coords_margin = round(coords_margin * (1. / scale_factor))
img_pad = math.ceil(((aug_scale * actual_tile_size + 2 *
                      actual_tile_margin + 1) - actual_tile_size) / 2.)

def getCoordsInSquareWithMargin(coords, y, x, size, margin):
  return coords.loc[(coords['row'] >= y - margin)
                    & (coords['row'] < y + size + margin) \
                    & (coords['col'] >= x - margin)
                    & (coords['col'] < x + size + margin)]

def countCoords(coords):
  return np.asarray([len(coords.loc[coords['cls'] == cls_idx])
                     for cls_idx in range(cls_nb)], dtype=np.int32)

def storeExampleRoundRobin(img_data, coords, scale=1.):
  global curr_writer, writers, writer_indices

  if coords is not None:
    coords_data = coords[['cls', 'row', 'col']].as_matrix().reshape((-1)).astype(np.int64)    
    coords_data_len = coords.shape[0]
  else:
    coords_data = np.empty((0), dtype=np.int64)
    coords_data_len = 0

  dst_size = tile_size + 2 * tile_margin
  feature_dict = {
    'image/height':  tf.train.Feature(int64_list=tf.train.Int64List(value=[dst_size])),
    'image/width':   tf.train.Feature(int64_list=tf.train.Int64List(value=[dst_size])),
    'image/scale':   tf.train.Feature(float_list=tf.train.FloatList(value=[scale])),
    'image':         tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_data])),
    'coords/length': tf.train.Feature(int64_list=tf.train.Int64List(value=[coords_data_len])),
    'coords':        tf.train.Feature(int64_list=tf.train.Int64List(value=coords_data))
  }

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  if not dry_run:
    writers[writer_indices[curr_writer]].write(example.SerializeToString())
  curr_writer += 1
  if curr_writer >= len(writers):
    np.random.shuffle(writer_indices) 
    curr_writer = 0

def getTransformedTileAndCoords(img, ref_y, ref_x, coords=None, \
                                rot=0, scale=1., flip=False):
  
  src_size = round(scale * actual_tile_size + 2 * actual_tile_margin)
  half_size = src_size / 2.
  margin = (src_size - actual_tile_size) // 2
  y = ref_y - margin
  x = ref_x - margin
  dst_size = tile_size + 2 * tile_margin
  coords_pad = actual_coords_margin - actual_tile_margin

  if y + img_pad < 0 or y + src_size + img_pad < 0 \
    or y + img_pad > img.shape[0] or y + src_size + img_pad > img.shape[0] \
    or x + img_pad < 0 or x + src_size + img_pad < 0 \
    or x + img_pad > img.shape[1] or x + src_size + img_pad > img.shape[1]:

    raise ValueError('Indices out of bounds: %i, %i, %i, %i' %
                     (y + img_pad, y + src_size + img_pad, x + img_pad, x + src_size + img_pad))

  tile = cv2.resize(img[y + img_pad:y + src_size + img_pad, \
                        x + img_pad:x + src_size + img_pad], \
                    (dst_size, dst_size), \
                    interpolation=cv2.INTER_LANCZOS4) 
                  # interpolation=cv2.INTER_AREA)

  if coords is not None:
    # Needs a deep copy because now we are going to transform the coordinates.
    coords = getCoordsInSquareWithMargin(coords, y, x, src_size, coords_pad).copy()

    corr_scale = float(dst_size - 1) / float(src_size - 1)

    coords.loc[:, 'row'] = corr_scale * (coords.loc[:, 'row'] - y - half_size)
    coords.loc[:, 'col'] = corr_scale * (coords.loc[:, 'col'] - x - half_size)
    

    if flip:
      tile = tile[::-1]
      coords.loc[:, 'row'] = -coords.loc[:, 'row']

    if rot != 0:
      tile = np.rot90(tile, rot)
      
      M_rot = None
      if rot == 1: M_rot = np.asarray([[ 0., -1.], [ 1.,  0.]])
      if rot == 2: M_rot = np.asarray([[-1.,  0.], [ 0., -1.]])
      if rot == 3: M_rot = np.asarray([[ 0.,  1.], [-1.,  0.]])

      coords.loc[:, ['row', 'col']] = np.dot(M_rot, coords.loc[:, ['row', 'col']].T).T

    coords.loc[:, 'row'] += corr_scale * half_size
    coords.loc[:, 'col'] += corr_scale * half_size
    coords = np.round(coords)
    
    if dry_run:
      title = 'Transformed_'+str(tile_size)+'x'+str(tile_size)+'_'
      imshow(tile, coords=coords, save=True, title=title)

  if dry_run:
    tile_data = b''
  else:
    tile_data = cv2.imencode('.png', np.asarray(tile)[..., ::-1].astype(np.uint8))[1].tostring()
    
  return tile_data, coords, tile

imshow_counter = 0
def imshow(img, coords=None, title='Image', wait=True, destroy=True, save=False, normalize=False):
  global imshow_counter

  img = img.copy().astype(np.float32)

  def fill_region(dst, y, x, h, w, v):
    h2 = h // 2
    w2 = w // 2
    py = y - h2 
    px = x - w2 
    y_min = max(0, py)
    y_max = min(dst.shape[0], y + h2)
    x_min = max(0, px)
    x_max = min(dst.shape[1], x + w2)
    if y_max - y_min <= 0 or x_max - x_min <= 0:
      return

    dst[y_min:y_max, x_min:x_max] = v

  if normalize:
    img -= np.min(img)
    m = np.max(img)
    if m != 0.:
      img /= m

  if save:
    if np.all(img <= 1.0):
      img *= 255.
      
  if coords is not None:
    img = np.copy(img)
    if isinstance(coords, pd.DataFrame):
      for coord in coords.itertuples():
        fill_region(img, int(round(coord.row)) - 2, int(round(coord.col)) - 2, \
                    5, 5, np.asarray(cld.cls_colors[coord.cls]))
    else:
      for coord in coords:
        fill_region(img, int(round(coord[1])) - 2, int(round(coord[2])) - 2, \
                    5, 5, np.asarray(cld.cls_colors[coord[0]]))

  if save:
    if len(img.shape) == 2:
      img = img[:, :, None]
    lockfile = os.path.join(debug_dir, '.lock')
    with open(lockfile, 'w') as fp:
        with flock.Flock(fp, flock.LOCK_EX) as lock:
          curr_num = len(os.listdir(debug_dir))
          filename = os.path.join(debug_dir, 'imshow_%s_%i.png' % (title, curr_num)) 
          cv2.imwrite(filename, img[..., ::-1])

    return

  plt.title(title)
  plt.imshow(img)
  plt.show()

  if wait:
    input('Press enter to continue...')
  if destroy:
    plt.close()

def processExample(tid):

  np.random.seed(tid)
  
  coords = cld_coords[cld_coords['tid'] == tid]
  img = cld.load_sample_image(tid, 1., 0)
  shape = img.shape
  #dot_img = cld.load_dot_image(tid, 1., 0)

  #dot_img_sum = dot_img.astype(np.uint16).sum(axis=-1)[..., None]
  #img = np.where(dot_img_sum < masked_region_threshold, dot_img, img)
  #del dot_img, dot_img_sum
  img = np.pad(img, ((img_pad, img_pad), (img_pad, img_pad), (0, 0)),
               mode='constant')
  tile_counter = 0
  background_tile_counter = 0
  class_counts = np.zeros((cls_nb,), dtype=np.int64)
  dist_list = []
  examples = []
  y_count = math.floor(shape[0] / actual_tile_size)
  x_count = math.floor(shape[1] / actual_tile_size)
  start_y = (shape[0] - actual_tile_size * y_count) // 2
  start_x = (shape[1] - actual_tile_size * x_count) // 2
  for y_idx in range(y_count):
    for x_idx in range(x_count):
      y = start_y + y_idx * actual_tile_size
      x = start_x + x_idx * actual_tile_size
      coords_ = getCoordsInSquareWithMargin(coords, y, x, actual_tile_size, 0)
      counts = countCoords(coords_)

      if np.all(np.asarray(counts) == 0):
        if np.random.random() <= background_rate:
          example = getTransformedTileAndCoords(img, y, x)
          examples.append(example[:2])  # Only append the encoded tile.
          background_tile_counter += 1
          tile_counter += 1
      else:
        #np.random.shuffle(aug_indices)
        #for aug_idx in [0] + aug_indices[:math.floor(aug_frac * len(aug_params))]:
        #example = getTransformedTileAndCoords(img, y, x, coords, *aug_params[aug_idx])
        
        example = getTransformedTileAndCoords(img, y, x, coords)
        examples.append(example[:2])
        coords_ = getCoordsInSquareWithMargin(example[1], 0, 0, tile_size, -tile_margin)
        class_counts += countCoords(coords_)
        tile_counter += 1

  return tile_counter, background_tile_counter, class_counts, examples, dist_list



def create_set(train_ids, split_nb, suffix, shall_clear_files=False):
  global curr_writer, writers, writer_indices

  #pool = multiprocessing.Pool(processes=num_processes)
  sess = tf.Session()

  # Prepare workspace.
  curr_writer = 0
  writer_indices = list(range(split_nb))
  writers = []
  if not dry_run:
    if clear_files and shall_clear_files:
      filenames = tf.gfile.Glob(os.path.join(data_dir, '*_%s.tfrecords' % suffix))
      for filename in filenames:
        tf.gfile.Remove(filename)
      print('All *_%s.tfrecords have been deleted.' % suffix)
  
  for i in range(split_nb):
    filename = os.path.join(data_dir, 'data_%03i_%s.tfrecords' % (i, suffix))
    writers.append(tf.python_io.TFRecordWriter(filename))
  

  tile_counter = 0
  background_tile_counter = 0
  class_counts = np.zeros((cls_nb,))
  dist_list = []

  print('Generating training set...')

  image_counter = 0 
  prev_time = None
  curr_rate = 280.
  start_time = time.perf_counter()
  #for tile_counter_, background_tile_counter_, class_counts_, examples_, dist_list_ \
  #   in pool.imap_unordered(processExample, train_ids):

  for tid in train_ids:
    tile_counter_, background_tile_counter_, class_counts_, examples_, dist_list_  = processExample(tid)

    tile_counter += tile_counter_
    background_tile_counter += background_tile_counter_

    for i in range(len(class_counts)):
      class_counts[i]+=class_counts_[i]

    #pca
    dist_list += dist_list_
    image_counter += 1

    for example in examples_:
      storeExampleRoundRobin(*example)

    if prev_time is None:
      prev_time = time.perf_counter()
    else:
      curr_time = time.perf_counter()
      elapsed = curr_time - prev_time
      curr_rate = (29. * curr_rate + (60. / elapsed)) / 30.
      prev_time = curr_time
    time_str = time.strftime('%H:%M:%S', time.gmtime(time.perf_counter() - start_time))
    eta_str = time.strftime('%H:%M:%S', time.gmtime(60. * (len(train_ids) - image_counter) \
                                                        / curr_rate))

    print('Progress: %3.1f%%, Tiles: %i, Histogram: %s, Images/Min.: %3.1f, Elapsed: %s, ETA: %s' \
              % (float(image_counter) / float(len(train_ids)) * 100, \
                             tile_counter, str(class_counts), curr_rate, time_str, eta_str))
      
    print('')
      
    #pool.close()
    #spool.join()


    #pca


  print('-' * 61)
  print('Final report:')
  print('-' * 61)
  print('Number of processed images:')
  print(len(train_ids))
  print('Number of tiles:')
  print(tile_counter)
  print('Number of background tiles:')
  print(background_tile_counter)
  print('Class histogram:')
  print(class_counts)
  print('Normalized class histogram:')
  print(class_counts / (np.sum(class_counts) + 1e-10))
  print('Normalized class histogram (overall):')
  print(cls_counts / (np.sum(cls_counts) + 1e-10))
  print('Histogram of class distances:')
  print(np.histogram(np.asarray(dist_list), bins=32))
    



def main(argv):

  train_ids = np.asarray(cld.tids)
  train_val_split = float(input('Train/val split? '))
  if train_val_split < 0.1 or train_val_split > 1:
    print('Type in something meaningful, please!, we will use 0.1')
    train_val_split = 0.1
  split_index = round(train_val_split * len(train_ids))
  shuffled_ids = list(range(len(train_ids)))
  np.random.shuffle(shuffled_ids)
  train_split_nb = round((1. - train_val_split) * storage_split_nb)
  val_split_nb = round(train_val_split * storage_split_nb)
  shall_clear_files = input('Delete all .tfrecords? (y/N) ').lower() == 'y'

  create_set(train_ids[shuffled_ids[split_index:]], train_split_nb, suffix=train_suffix, shall_clear_files=shall_clear_files)
  create_set(train_ids[shuffled_ids[:split_index]], val_split_nb, suffix=val_suffix, shall_clear_files=shall_clear_files)

  print('train_ids=', repr(train_ids[shuffled_ids[split_index:]]))
  print('validation_ids=', repr(train_ids[shuffled_ids[:split_index]]))




if __name__ == '__main__':
  main(sys.argv)