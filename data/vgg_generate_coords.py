# load the annotated images and generates a csv with the 
# corresponding coordinates
# inspired by sealionengine: https://github.com/gecrooks/sealionengine

import sys
import os
import csv
from collections import namedtuple
import argparse
import numpy as np
import PIL
from PIL import Image
import skimage
import skimage.io
import skimage.draw
import skimage.measure
import shapely
import shapely.geometry
from shapely.geometry import Polygon


# ================ Meta ====================
__description__ = 'VGG cell csv coordinate generator'
__version__ = '0.1.0'
__license__ = 'MIT'
__author__ = 'Nico Lutz'
__status__ = "Prototype"
__copyright__ = "Copyright 2017"

def package_versions():
    print('python        \t', sys.version[0:5])
    print('numpy         \t', np.__version__)
    print('skimage       \t', skimage.__version__)
    print('pillow (PIL)  \t', PIL.__version__)
    print('shapely       \t', shapely.__version__)

#but cell folder right next to this file
SOURCEDIR = os.path.join('cells')

#produces csv in outdir folders
OUTDIR = os.path.join('.', 'outdir')

PRINTCIRCLES = False

TILE_SIZE = 256   # Default tile size

CellCoords = namedtuple('CellData', ['tid','cls','row','col'])

class CellData(object):

  def __init__(self, sourcedir= SOURCEDIR, outdir=OUTDIR):
    
    self.sourcedir = sourcedir    
    self.outdir = outdir

    self.cls_nb = 1

    self.dot_radius = 1

    #RGB
    self.cls_colors = (
      (255, 0, 23)#red
    )

    self.cls_names = (
            'cell')

    self.tids = list(range(1,201))

    self.paths = {

    'cell': os.path.join(sourcedir, '{tid}cell.png'),
    'dots': os.path.join(sourcedir, '{tid}dots.png'),
    'coords': os.path.join(outdir,'coords.csv')
    }
  
  def path(self, name, **kwargs):
        """Return path to various source files"""
        path = self.paths[name].format(**kwargs)
        return path
  
  def _load_image(self, itype, tid, scale = 1):

    tid_s = '{num:03d}'.format(num=tid)
    fn = self.path(itype, tid = tid_s)

    with open(fn, 'rb') as img:
      with Image.open(img) as image:

        im = np.asarray(image)

    return im

  def load_sample_image(self, tid, scale = 1, border = 0):
    '''

    Returns:
      uint8 numpy array
    '''

    img = self._load_image('cell', tid)

    return img

  def load_dot_image(self, tid, scale = 1, border = 0):

    img  = self._load_image('dots',tid)

    return img

  def find_coords(self, tid):
    '''
    find coords in dotted images

    '''
    MAX_AREA = 3
    MIN_AREA = 0
    MAX_COLOR_DIFF = 30

    #src_img = np.asarray(self.load_sample_image(tid), dtype=np.float)
    dot_img = np.asarray(self.load_dot_image(tid), dtype=np.float)

    #implemt multiple classes
    #MIN_DIFFERENCE = 14
    #MAX_AVG_DIFF = 40
    #MAX_MASK = 8

    #img_diff = np.abs(src_img - dot_img)

    # Detect bad data. If train and dotted images are very different then somethings wrong.
    #avg_diff = img_diff.sum() / (img_diff.shape[0] * img_diff.shape[1])
    #if avg_diff > MAX_AVG_DIFF:
        #print('( Bad train image -- exceeds MAX_AVG_DIFF: {} )'.format(tid))
        #return ()
    #    pass

    #img_diff = np.max(img_diff, axis=-1)

    #img_diff[img_diff < MIN_DIFFERENCE] = 0
    #img_diff[img_diff >= MIN_DIFFERENCE] = 255

    img_grey = dot_img[:,:,0]/255

    celldots = []

    #for cls, color in enumerate(self.cls_colors):
    
    cls = 0
    #color = self.cls_colors

    #color_array = np.array(color)[None, None, :]
    #color_diff = dot_img * (img_diff > 0)[:, :, None] - color_array
    #has_color = np.sqrt(np.sum(np.square(color_diff), axis=-1)) < MAX_COLOR_DIFF
    #contours = skimage.measure.find_contours(has_color.astype(float), 0.5)
    contours = skimage.measure.find_contours(img_grey, 0.9)

    for cnt in contours:

      #otherwise raises an polygon error
      if len(cnt) < 3:
        print('Len cnt < 2')
        continue

      p = Polygon(shell = cnt)
      area = p.area

      if area > MIN_AREA and area < MAX_AREA:
        row, col = p.centroid.coords[0]#DANGER: skimage and cv2 coordinates transposed

        row = int(round(row))
        col = int(round(col))

        celldots.append(CellCoords(tid, cls, row, col))
      
    return celldots

  def save_coords(self,tid=None):

    if tid is None:
      tids = self.tids

    fn = self.path('coords')

    if os.path.exists(fn):
      raise IOError('Output file exists')


    all_coord_list = map(self.find_coords, tids)

    with open(fn, 'w') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(CellCoords._fields)
      for coord in all_coord_list:
        for c in coord:
          writer.writerow(c)

    print('Done')

  def draw_circles(self, img, coords) :
    radius = 3*3

    cls_colors = (
      (255, 0, 23)#red
    )

    for tid, cls, row, col in coords:
        rr, cc = skimage.draw.circle_perimeter(row, col, radius, shape = img.shape)
        img[rr, cc] = cls_colors[cls]
    return img

  def load_csv(self):
    import pandas as pd
    fn = self.path('coords')
    return pd.read_csv(fn)

  def tid_csv(self, csv ,tid):
    return csv[csv['tid']==tid].values.tolist()

  def draw_circles_for_tid(self,tid):
    img = self.draw_circles(np.copy(self.load_dot_image(tid)),self.tid_csv(self.load_csv(),tid))
    fn = os.path.join(self.outdir, 'circle_{}.png'.format(tid))
    Image.fromarray(img).save(fn)


def _cli():

  parser = argparse.ArgumentParser(
        description=__description__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  cmdparser = parser.add_subparsers(
      title='Commands',
      description=None,
      help="-h, --help Additional help",)

  parser.add_argument('--version', action='version', version=__version__)

  parser.add_argument('-s', '--sourcedir', action='store', dest='sourcedir', 
                      default=SOURCEDIR, metavar='PATH', help='Location of input data')

  parser.add_argument('-o', '--outdir', action='store', dest='outdir',
                        default=OUTDIR, metavar='PATH', help='Location of processed data')

  
  # == draw circles pictures ==#
  def circle(self,tids):
    img = self.draw_circles(np.copy(self.load_dot_image(tids)),self.tid_csv(self.load_csv(),tids))
    fn = os.path.join(self.outdir, 'circle_{}.png'.format(tids))
    Image.fromarray(img).save(fn)

  p = cmdparser.add_parser('circle', help='Generate image with known dots circled.')
  p.set_defaults(func=circle)
  p.add_argument('tids', action='store', type=int)

    # == draw cell circles pictures ==#
  def cellcircle(self,tids):
    img = self.draw_circles(np.copy(self.load_sample_image(tids)),self.tid_csv(self.load_csv(),tids))
    fn = os.path.join(self.outdir, 'cellcircle_{}.png'.format(tids))
    Image.fromarray(img).save(fn)

  p = cmdparser.add_parser('cellcircle', help='Generate image with known dots circled.')
  p.set_defaults(func=cellcircle)
  p.add_argument('tids', action='store', type=int)
  
  # == generate csv ==#
  def gen(self):
    cld.save_coords()

  p = cmdparser.add_parser('gen', help='Generates coords csv in outdir')
  p.set_defaults(func=gen)

  # Run
  opts = vars(parser.parse_args())

  sourcedir = opts.pop('sourcedir')
  outdir = opts.pop('outdir')

  cld = CellData(sourcedir=sourcedir, outdir=outdir)
  
  func = opts.pop('func')
  func(cld, **opts)
  

if __name__ == '__main__':
  _cli()

