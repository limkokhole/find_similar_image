# -*- coding: utf-8 -*-
# The MIT License (MIT)
# Copyright (c) 2019 limkokhole@gmail.com
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.
from __future__ import print_function #to make python2 print color like python3 do

__author__ = 'Lim Kok Hole'
__copyright__ = 'Copyright 2019'
__credits__ = [
    'Dr. Neal Krawetz at http://www.hackerfactor.com/about.php', #research of average/difference hash
    'David Oftedal at http://01101001.net/', #average/difference hash code (without comparision metric)
    'Sastanin at https://stackoverflow.com/users/25450/sastanin' #Manhattan/zero norm code (without resize and comparison metric)
]
__license__ = 'MIT'
__version__ = 1.1
__cache_file_version__ = 1.1 #only incresae this if cache file structure change
__maintainer__ = 'Lim Kok Hole'
__email__ = 'limkokhole@gmail.com'
__status__ = 'Production'

import sys
import os
import time
import uuid
import traceback
from PIL import Image

# Python 2 and 3: backward-compatible
from past.builtins import xrange

# Not going to use this since 3.5's walk() already default to scandir
# https://github.com/benhoyt/scandir
#try: # python 3
#    from os import scandir
#except ImportError:
#    from scandir import scandir

try:
    import readline # allow edit input by LEFT key
except ImportError:
    print("Module readline not available.")
else:
#    import rlcompleter
    readline.set_completer_delims(' \t\n=')
    readline.parse_and_bind("tab: complete")

import argparse
from argparse import RawTextHelpFormatter #make --help's \n works and indentation pretty
arg_parser = argparse.ArgumentParser(
    # don't specify prefix_chars='-+/' here which causes / in path is not option value
    description='Find simlar images in specific directory by specific image', formatter_class=RawTextHelpFormatter)
args = ""
PY3 = sys.version_info[0] >= 3
if PY3:
    pass
else:
    input = raw_input

OS_SEP = os.sep

RESIZE_X = 0 #256
RESIZE_Y = 0 #128

def quit(msgs, exit=True):
    if not isinstance(msgs, list):
        msgs = [msgs]
    for msg in msgs:
        print('\x1b[1;41m%s\x1b[0m\x1b[K' % msg)
    if exit:
        print('\x1b[1;41m%s\x1b[0m\x1b[K' % 'Abort.')
        sys.exit()

from PIL import ImageStat
#Research by http://www.hackerfactor.com/blog/?/archives/432-Looks-Like-It.html
#Hash code by http://01101001.net/DifferenceHash.py and http://01101001.net/AverageHash.py
#hole: but I decided combine both and pick higher of them as "matched" metric.
def AverageHash(theImage):

	# Convert the image to 8-bit grayscale.
	theImage = theImage.convert("L") # 8-bit grayscale

	# Squeeze it down to an 8x8 image.
	theImage = theImage.resize((8,8), Image.ANTIALIAS)

	# Calculate the average value.
	averageValue = ImageStat.Stat(theImage).mean[0]

	# Go through the image pixel by pixel.
	# Return 1-bits when the tone is equal to or above the average,
	# and 0-bits when it's below the average.
	averageHash = 0
	for row in xrange(8):
		for col in xrange(8):
			averageHash <<= 1
			averageHash |= 1 * ( theImage.getpixel((col, row)) >= averageValue)

	return averageHash

def DifferenceHash(theImage):

    # Convert the image to 8-bit grayscale.
    theImage = theImage.convert("L")  # 8-bit grayscale #orig
    # [todo:0] some files warning:
    # /home/xiaobai/.local/lib/python3.6/site-packages/PIL/Image.py:993: UserWarning
    # : Palette images with Transparency expressed in bytes should be converted to RGBA images

    #theImage = theImage.convert("1")  # 8-bit grayscale # hole

    # Squeeze it down to an 8x8 image.
    theImage = theImage.resize((8, 8), Image.ANTIALIAS) #orig
    #theImage = theImage.resize((64, 64), Image.ANTIALIAS) # hole

    # Go through the image pixel by pixel.
    # Return 1-bits when a pixel is equal to or brighter than the previous
    # pixel, and 0-bits when it's below.

    # Use the 64th pixel as the 0th pixel.
    previousPixel = theImage.getpixel((0, 7))

    differenceHash = 0
    for row in xrange(0, 8, 2):

        # Go left to right on odd rows.
        for col in xrange(8):
            differenceHash <<= 1
            pixel = theImage.getpixel((col, row))
            differenceHash |= 1 * (pixel >= previousPixel)
            previousPixel = pixel

        row += 1

        # Go right to left on even rows.
        for col in xrange(7, -1, -1):
            differenceHash <<= 1
            pixel = theImage.getpixel((col, row))
            differenceHash |= 1 * (pixel >= previousPixel)
            previousPixel = pixel

    return differenceHash


def share_print(img_f_hash_diff, img_f_hash_avg, dir_img_f_hash_diff, dir_img_f_hash_avg, path, ei, total_matched):

    if args.verbose:
        print('dir_img_f_hash_diff: ' + str(dir_img_f_hash_diff))
        print('dir_img_f_hash_avg: ' + str(dir_img_f_hash_avg))
        print('img_f_hash_diff: ' + str(img_f_hash_diff))
        print('img_f_hash_avg: ' + str(img_f_hash_avg))

    d_diff = ((64 - bin(img_f_hash_diff ^ dir_img_f_hash_diff).count("1"))*100.0)/64.0 #orig
    d_avg = ((64 - bin(img_f_hash_avg ^ dir_img_f_hash_avg).count("1"))*100.0)/64.0 #orig
    #d = ((4096 - bin(img_f_hash ^ dir_img_f_hash).count("1"))*100.0)/4096.0 # hole
    #d_both = ((d_diff+d_avg)/2.0) #no longer print both avg since it's misleading
    # ... due to current comparision metric is "pick higher of both"

    if args.verbose:
        print('img percentage diff: ' + str(d_diff))
        print('img percentage avg: ' + str(d_avg))
        print('arg percentage: ' + str(args.percentage))
    
    # no need update cache file if want to test this since cache store the original hash, not the both hash
    #if d_both >= args.percentage:
    if (d_diff >= args.percentage) or (d_avg >= args.percentage):

        if d_diff >= args.percentage: #note that this doesn't means avg not match, only means first match
            print('[' + str(ei) + '] \x1b[1;35m' + path + '\x1b[0m\x1b[K \x1b[1;32m[Matched]\x1b[0m\x1b[K diff: ' + str(d_diff) + '%' + ' avg: ' + str(d_avg) + '%' )
        else:
            print('[' + str(ei) + '] \x1b[1;35m' + path + ' \x1b[0m\x1b[K diff: ' + str(d_diff) + '%' + ' \x1b[1;32m[Matched]\x1b[0m\x1b[K avg: ' + str(d_avg) + '%' )
        # total_matched not increment yet, so +1
        if args.show is not None:
            if (args.show == 0) or ((total_matched+1) <= args.show):
                try:
                    Image.open(path).show()
                except OSError:
                    print('\x1b[1;41m-s failed. Probably file no longer exists\x1b[0m\x1b[K: ' + path)
        return True
    else:
        if not args.m:
            print('[' + str(ei) + '] \x1b[1;35m' + path + '\x1b[0m\x1b[K \x1b[1;31m[Not Match]\x1b[0m\x1b[K diff: ' + str(d_diff) + '%' + ' avg: ' + str(d_avg) + '%' )
        return False

def diff_file(path, hash_cache_f, write_once, dir_path, hash_type):
    dir_img_f = Image.open(path)
    dir_img_f.load()
    
    dir_img_f_hash_diff = DifferenceHash(dir_img_f)
    dir_img_f_hash_avg = AverageHash(dir_img_f)

    if args.w is not False:

        if not write_once:
            dir_full_path = os.path.abspath(os.path.expanduser(os.path.expandvars(dir_path))) 
            path_total_line = dir_full_path.count('\n') + 1
            hash_cache_f.writelines( str(__cache_file_version__) + '\n')
            hash_cache_f.writelines( str(path_total_line) + '\n')
            hash_cache_f.writelines(dir_full_path + '\n') #store directory path on 1st line
            write_once = True

        path_total_line = path.count('\n') + 1
        hash_cache_f.writelines( str(path_total_line) + '\n')
        hash_cache_f.writelines(path + '\n')
        # no nid worry str() truncate, it's not float
        hash_cache_f.writelines(str(dir_img_f_hash_diff) + '\n')
        hash_cache_f.writelines(str(dir_img_f_hash_avg) + '\n')

    if args.verbose:
        print('hash diff: ' + str(dir_img_f_hash_diff))
        print('hash avg: ' + str(dir_img_f_hash_avg))

    # [todo:0] need .close() or not ? which faster ?
    # dir_img_f.close()

    return (dir_img_f_hash_diff, dir_img_f_hash_avg, write_once)

#shared by args.ln and args.lnm
def create_symlink(symlink_dir_path, path):
    if symlink_dir_path:
        try:
            # path already full path by abspath(), and symlink target need full path 
            if args.unique_ln:
                os.symlink( path, os.path.join(symlink_dir_path, str(uuid.uuid4()) + '_' + os.path.basename(path)) )
            else:
                os.symlink( path, os.path.join(symlink_dir_path, os.path.basename(path)) )
        except OSError:
            print('Failed to create symlink: ' + path)
            if args.verbose:
                print('This error is normal if file already exists.')

def to_grayscale(arr):
	"If arr is a color image (3D array), convert it to grayscale (2D array)."
	if len(arr.shape) == 3:
		return average(arr, -1)  # average over the last axis (color channels)
	else:
		return arr

def normalize(arr):

    #[todo:1/2] move on top same ?
    amin = arr.min()

    rng = arr.max() - amin #arr.min()

    # https://gist.github.com/astanin/626356#gistcomment-2568216
    # prevent divide by 0 error
    # hole: can generate test image by `convert -size 128x256 xc:white empty128x256_white.jpg`
    #, to reproduce since all items' value is same and max() - min() will produce rng as 0 here
    #, which causes all values is nan and impossible match by float metric, even though no exception
    if rng == 0:
        rng = 1

    #amin = arr.min()
    return (arr - amin) * 255 / rng

def compare_images(img_f_np_ndarray, dir_img_f_np_ndarray):
	# calculate the difference and its norms

	diff = img_f_np_ndarray - dir_img_f_np_ndarray  # elementwise for scipy arrays
	m_norm = sum( abs(diff) )  # Manhattan norm #hole: seems like impossible -, so abs() useless here
	#z_norm = norm( diff.ravel(), 0 )  # Zero norm
	#return (m_norm, z_norm)
	return m_norm

if __name__ == "__main__":
    arg_parser.add_argument('-d', '--dir', dest='dir_to_scan', help='Directory path to scan recursively.\
    \nYou can\'t use -d together with -c')
    arg_parser.add_argument('-md', '--max-depth', dest='max_depth', type=int, help='Specify max depth of directory path to scan.\
    \nIf you use -md together with -c then its max depth depends on the path of cache file without re-scan.')
    arg_parser.add_argument('-v', '--verbose', action='store_true', help='Shows more log.\
    \nYou probably want to remove -m too.')
    arg_parser.add_argument('-n', '--norm', dest='norm', action='store_true'
                            , help='Use Manhattan norm which not allow cache.\
    \nIt has higher tolerance in different size but lower tolerance in bright.\
    \nDefault are Average and Difference hash(pick higher) which allow cache with -c/-w.')
    arg_parser.add_argument('-m', action='store_true',
                            help='Only shows matched line.')
    arg_parser.add_argument('-c', nargs='?', default=False, help='Read hash values from this cache file without re-scan images. Default is hash.cache if -c without value.\
    \nYou should use -w once to generate new cache file before use -c')
    arg_parser.add_argument('-w', nargs='?', default=False,
                            help='Write hash values to this cache file for re-use by -c. Default is hash.cache if -w without value.')
    arg_parser.add_argument('-s', '--show', type=int, help='Shows Nth matched images in popup viewer.\
    \n0 means unlimited but be aware it probably hang your system if too much viewers popup.')
    arg_parser.add_argument('-l', '--link-match', dest='ln', help='Create symlink of matched images in this directory.')
    arg_parser.add_argument('-lnm', '--link-not-match', dest='lnm', help='Create symlink of not-matched images in this directory.')
    arg_parser.add_argument('-f', '--follow-symlink', dest='follow_symlink', action='store_true'
                            , help='Follow symlink for files and directory.\
    \nDefault is don\'t follow symlink to avoid scan duplicated (-s will popup twice) files in -l/-lnm directory.')
    arg_parser.add_argument('-u', '--unique-ln', dest='unique_ln', action='store_true'
                            , help='Create symlink with unique filename by uuid.\
    \nThis option do nothing without -l or -lnm')
    arg_parser.add_argument('-p', '--percentage',  default=80.0, type=float
                            , help='Match by diff less or equal to this percentage floating value (without %%).\
    \nDefault is 80.0%%. You should try lower it to 75%% or 70%% if it doesn\'t match a few.')
    arg_parser.add_argument('image_path', nargs='?', help='Specify single image path to check.')
    arg_parser.add_argument('--version', action='version', version='%(prog)s ' + str(__version__))
    args, remaining = arg_parser.parse_known_args() #--version will stop here
    
    if remaining:
        quit('Redundant arguments: ' + ' '.join(remaining))
    if args.dir_to_scan and (args.c is not False):
        quit('You can\'t use -d together with -c')
    # Need exclude -c without value means is None and -c with value
    if (args.c is not False) and (args.w is not False):
        # so need absence of -c means is False and its reverse "is not False" here
        # -c without value -> None
        # no -c -> False
        quit('You can\'t use -c together with -w')
    if args.norm and ( (args.c is not False) or (args.w is not False) ):
        quit('You can\'t use -n together with -w/-c')
        
    if args.image_path:
        img_path = args.image_path
    else:
        if PY3:
            img_path = input('Image path/name: ')
        else:
            img_path = input('Image path/name: '.encode('utf-8'))
        if not img_path:  # don't strip()
            quit('Image path should not be empty. Abort.')

    img_path = os.path.abspath(os.path.expanduser(os.path.expandvars(img_path)))

    if args.c is False:
        if args.dir_to_scan:
            dir_path = args.dir_to_scan
        else:
            if PY3:
                dir_path = input('Directory path/name ( "." for current directory): ')
            else:
                dir_path = input('Directory path/name ( "." for current directory): '.encode('utf-8'))
            if not dir_path:  # don't strip()
                quit('Directory path should not be empty. Abort.')

        # [1] abspath already act as normpath to remove trailing os.sep
        # [2] expanduser expands ~
        # [3] expandvars expands $HOME
        dir_path = os.path.abspath(os.path.expanduser(os.path.expandvars(dir_path)))
        if not os.path.exists(dir_path):
            quit('Directory ' + dir_path + ' does not exist.')
        elif not os.path.isdir(dir_path):
            quit(dir_path + ' is not a directory.')

    if args.norm:

        # https://stackoverflow.com/a/3935002/1074998
        # Modified(added resize and comparision metric) from  https://gist.github.com/astanin/626356
        # only import if use norm algo
        import cv2
        #from scipy.misc import imread
        from scipy.linalg import norm
        from scipy import sum, average
        #import numpy as np #test cache .save/.load

        # [todo:0] support alpha with non-alpha
        # test case: vegito_black.jpg and vegito_by_al3x796-dakqk7k.png
        img_f_np_ndarray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        #img_f_np_ndarray = cv2.imread(img_path, cv2.IMREAD_COLOR )
        #img_f_np_ndarray = cv2.imread(img_path, cv2.IMREAD_REDUCED_GRAYSCALE_8)
        
        try:
            RESIZE_Y, RESIZE_X = img_f_np_ndarray.shape[:2] #(height, width)
        except AttributeError:
            quit('Probably the image path not exists:\n' + img_path)

        if args.percentage > 100:
            arg_percentage = 100
        elif  args.percentage < 0:
            arg_percentage = 0
        else:
            arg_percentage = args.percentage

        # [rule 1] to make manhattan comparision works, you should resize it to same image as source image
        # , so source image no need resize, while the dir images must resize bsaed on the shape of source image
        # [rule 2] since 100% means perfect match while 0.0 means perfect match in manhhattan norm
        # , so we need 100 - args_percentage to make the metric "reverse".
        # [rule 3] 255 is the maximum value of array_1_max(all 255) - array_2_min(all 0)
        # , which depends on the resized shape. So we need mutiply them with percentage
        # ... to get the final acceptable norm
        acceptable_manhattan_diff = RESIZE_Y * RESIZE_X * 255 * (100 - arg_percentage) / 100.0
        #hole: no nid check zero norm, AFAIS it's same
        #acceptable_zero_diff = RESIZE_Y * RESIZE_X * (100 - arg_percentage) / 100.0

        print('RESIZE_X: ' + str(RESIZE_X) + ' RESIZE_Y: ' + str(RESIZE_Y))
        print('Acceptable percentage is: ' + str(arg_percentage))
        print('Acceptable manhattan norm is: ' + str(acceptable_manhattan_diff))
        #print('Acceptable zero norm is: ' + str(acceptable_zero_diff))

        #height, width, channels = img_f_np_ndarray.shape
        #hole: in case got channel, see https://stackoverflow.com/questions/19098104
        #try:
        #    img_f_np_ndarray = cv2.resize(img_f_np_ndarray, (RESIZE_X, RESIZE_Y)) # hole
        #except cv2.error:
        #    quit('Probably the image path not exists:\n' + img_path)
        img_f_np_ndarray = to_grayscale(img_f_np_ndarray.astype(float))
        # normalize to compensate for exposure difference
        # print('before norm: ' + repr(img_f_np_ndarray))
        img_f_np_ndarray = normalize(img_f_np_ndarray)
        # print('after norm: ' + repr(img_f_np_ndarray))

    else:
        try:
            img_f = Image.open(img_path)
            img_f.load()  # careful this is not use return result
            img_f_hash_avg = AverageHash(img_f)
            img_f_hash_diff = DifferenceHash(img_f)
            if args.verbose:
                print('hash diff: ' + str(img_f_hash_diff) + ' hash avg: ' + str(img_f_hash_avg))

        except OSError: # python 2 no such FileNotFoundError, so need catch ancestor OSError
            quit('No such image file exists. Abort.')

        hash_cache = 'hash.cache'

        if args.w is not False:
            if args.w:
                hash_cache = args.w
                if args.verbose:
                    print('w filename: ' + hash_cache)
            open(hash_cache, "w").close  # empty the file
            hash_cache_f = open(hash_cache, "a")

        elif args.c is not False:
            if args.c:
                hash_cache = args.c
                if args.verbose:
                    print('c filename: ' + hash_cache)
            try:
                hash_cache_f = open(hash_cache, "r")
            except OSError:
                quit('-c\'s file ' + hash_cache +
                    ' doesn\'t exist. You should use -w once to generate cache file before use -c')
        else:
            hash_cache_f = None

    ln_dir_path = None
    if args.ln:
        ln_dir_path = os.path.abspath(os.path.expanduser(os.path.expandvars(args.ln)))
        if not os.path.exists(ln_dir_path):
            try:
                os.makedirs(ln_dir_path)
            except OSError: # FileExistsError in python 3 has ancestor OSError of python 2
                if args.verbose:
                    print('Symlink directory already exists.')
        elif not os.path.isdir(ln_dir_path):
            quit('The -ln ' + ln_dir_path + ' path already exists but not a directory.') 
        elif os.path.exists(ln_dir_path):
            cont = input('\n-ln ' + ln_dir_path + ' directory already exists\n, continue to write to this directory ? [yes/no] : ')
            if not cont or cont[0].lower() != 'y':
                quit('')

    lnm_dir_path = None
    if args.lnm:
        lnm_dir_path = os.path.abspath(os.path.expanduser(os.path.expandvars(args.lnm)))
        if not os.path.exists(lnm_dir_path):
            try:
                os.makedirs(lnm_dir_path)
            except OSError: # FileExistsError in python 3 has ancestor OSError of python 2
                if args.verbose:
                    print('Symlink directory already exists.')
        elif not os.path.isdir(lnm_dir_path):
            quit('The -ln ' + lnm_dir_path + ' path already exists but not a directory.') 
        elif os.path.exists(lnm_dir_path):
            cont = input('\n-ln ' + lnm_dir_path + ' directory already exists\n, continue to write to this directory ? [yes/no] : ')
            if not cont or cont[0].lower() != 'y':
                quit('')

    total_matched = 0
    total_not_match = 0
    total_not_a_image = 0
    max_depth = args.max_depth

    if args.c is False or args.norm:

        if args.verbose:
            print('read from files')
        
        ei = 0
        # for ei, entry in enumerate(scandir(dir_path)): #scandir
        write_once = False
        d_path_len = len(dir_path)

        #high_accuracy = args.percentage >= 100

        # Default already followlinks=False for directory in os.walk() arg
        for subdir, dirs, files in os.walk(dir_path, topdown=True, followlinks=args.follow_symlink): #[toprove:0] will topdown slower ?

            # --max-depth 0 should do nothing, conforms to find command
            if (max_depth is None) or ( subdir[d_path_len:].count(OS_SEP) < max_depth ):

                for file in files:

                    ei+=1
                    path = OS_SEP.join([subdir, file])
                    # if file.is_file(): #scandir

                    # Due to this script main feature is store as symlink, so it may walk to -l directory and causes duplicated -s
                    # Default should not follows symlink
                    if not args.follow_symlink and os.path.islink(path):
                        if args.verbose:
                            print('Skipping symlink file: ' + path)
                        continue

                    if args.verbose:
                        print('Trying... [' + str(ei) + ']' + path)

                    if args.norm:
                        #img_f_np_ndarray
                        dir_img_f_np_ndarray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                        try:
                            dir_img_f_np_ndarray = cv2.resize(dir_img_f_np_ndarray, (RESIZE_X, RESIZE_Y)) # hole
                        except cv2.error:
                            # [1] 592631bd925880533f5e26993ef9eb4247b02a34r1-540-303_hq.gif with anim throws here
                            # [2] invalid image aHR0cDovL2ltZzQuaW1ndG4uYmRpbWcuY29tL2l0L3U9MzQwNzA2Mjk2LDIwMjg4MDgxNzYmZm09MjYmZ3A9MC5qcGc=.jpg
                            # ... also throws here
                            # Both of above able shows some value with avg/diff hash algo method.
                            if args.verbose:
                                print('\x1b[0;35mProbably not an image.\x1b[0m\x1b[K')
                            total_not_a_image += 1
                            continue

                        dir_img_f_np_ndarray = to_grayscale(dir_img_f_np_ndarray.astype(float))

                        # print('before norm dir: ' + repr(dir_img_f_np_ndarray))
                        dir_img_f_np_ndarray = normalize(dir_img_f_np_ndarray)
                        # print('after norm dir: ' + repr(dir_img_f_np_ndarray))


                        #AFAIS img_f_np_ndarray.size doesn't change after return from normalize() above
                        #n_m, n_0 = compare_images(img_f_np_ndarray, dir_img_f_np_ndarray)
                        n_m = compare_images(img_f_np_ndarray, dir_img_f_np_ndarray)

                        #n_0 < 15958 #goku_black_ssj_rose_palette_xenoverse_by_al3x796-dag4hgd.png #15800)
                        #careful if text "by pixel" nid to take into account of 0 which 100% match 
                        #10648, 9600
                        #1000000
                        #9000, 16357
                        #if ( ( high_accuracy and ( ((n_m < 900000) and (n_0 < 7000)) or ( (n_m < 760000) and (n_0 < 8000) ) or ( (n_m < 660000) and (n_0 < 9000) ) ) ) or (not high_accuracy and (n_m < 1173136)) ): #1100000: 
                        #if ( ( high_accuracy and ( (n_m < 27606771) and (n_0 < 334953) ) )  or (not high_accuracy and (n_m < 1173136)) ): #1100000: 
                        #need <= to match 0.0
                        if n_m <= acceptable_manhattan_diff: #27606771: #1100000: 

                            #sample:
                            #> 1022339.0 for 2012-08-13-205321_1920x1080_scrot.png
                            #print((n_0 * 1.0 / img_f_np_ndarray.size) )

                            print('\n[' + str(ei) + '] \x1b[1;35m[ %s\x1b[0m\x1b[K\x1b[1;35m ]\x1b[0m\x1b[K' % path)
                            #print('\x1b[1;32m[Matched]\x1b[0m\x1b[K Manhattan norm:', n_m, '/ per pixel:', n_m / img_f_np_ndarray.size,\
                            #        ' Zero norm:', n_0, '/ per pixel:', n_0 * 1.0 / img_f_np_ndarray.size)
                            print('\x1b[1;32m[Matched]\x1b[0m\x1b[K Manhattan norm:', n_m, '/ per pixel:', n_m / img_f_np_ndarray.size)
                            total_matched += 1
                            create_symlink(ln_dir_path, path)

                            if args.show is not None:
                                if (args.show == 0) or ((total_matched) <= args.show):
                                    Image.open(path).show()
    
                            '''
                            elif n_0 <= 334953: #(not high_accuracy and n_0 < 14580): # 2nd priority #16380 #16160 #15882: 
                            print('\n[' + str(ei) + '] \x1b[1;35m[ %s\x1b[0m\x1b[K\x1b[1;35m ]\x1b[0m\x1b[K' % path)
                            print('Manhattan norm:', n_m, '/ per pixel:', n_m / img_f_np_ndarray.size,\
                                    ' \x1b[1;32m[Matched]\x1b[0m\x1b[K Zero norm:', n_0, '/ per pixel:', n_0 * 1.0 / img_f_np_ndarray.size)
                            total_matched += 1
                            create_symlink(ln_dir_path, path)

                            if args.show is not None:
                                if (args.show == 0) or ((total_matched) <= args.show):
                                    Image.open(path).show()
                            '''
                            #elif n_0 <= acceptable_zero_diff:
                            #    quit('Something wrong in algo, pls check.')
                        else:
                            create_symlink(lnm_dir_path, path)

                            total_not_match += 1
                            if not args.m:
                                print('\n[' + str(ei) + '] \x1b[1;35m[ %s\x1b[0m\x1b[K\x1b[1;35m ]\x1b[0m\x1b[K' % path)
                                #print('\x1b[1;31mNot Match: \x1b[0m\x1b[K Manhattan norm:', n_m, '/ per pixel:', n_m / img_f_np_ndarray.size,\
                                #     ' \x1b[1;31mNot Match: \x1b[0m\x1b[K Zero norm:', n_0, '/ per pixel:', n_0 * 1.0 / img_f_np_ndarray.size)
                                print('\x1b[1;31mNot Match: \x1b[0m\x1b[K Manhattan norm:', n_m, '/ per pixel:', n_m / img_f_np_ndarray.size)
                                
                    else:

                        if PY3: # 3.3 might different exception, but I only has latest version 3.6.8 to test
                            # , see https://stackoverflow.com/questions/28633555/how-to-handle-filenotfounderror-when-try-except-ioerror-does-not-catch-it
                            try:
                                dir_img_f_hash_diff, dir_img_f_hash_avg, write_once = diff_file(path, hash_cache_f, write_once, dir_path, False)
                            except FileNotFoundError:
                                if args.verbose:
                                    print('No such file exists.')
                                continue
                            except OSError:
                                if args.verbose:
                                    print('\x1b[0;35mProbably not an image.\x1b[0m\x1b[K')
                                total_not_a_image += 1
                                continue
                            except:
                                if args.verbose:
                                    print('\x1b[0;35mUnexpected error:\x1b[0m', sys.exc_info()[0],'\x1b[K')
                                continue
                        else:
                            try:
                                dir_img_f_hash_diff, dir_img_f_hash_avg, write_once = diff_file(path, hash_cache_f, write_once, dir_path, False)
                            except IOError: # python 2 hard to distinct it's not an image VS no file exception
                                if args.verbose:
                                    print('\x1b[0;35mProbably not an image.\x1b[0m\x1b[K')
                                total_not_a_image += 1
                                continue
                            except:
                                if args.verbose:
                                    print('\x1b[0;35mUnexpected error:\x1b[0m', sys.exc_info()[0],'\x1b[K')
                                continue

                        if share_print(img_f_hash_diff, img_f_hash_avg, dir_img_f_hash_diff, dir_img_f_hash_avg, path, ei, total_matched):
                            total_matched += 1
                            create_symlink(ln_dir_path, path)
                        else:
                            total_not_match += 1
                            create_symlink(lnm_dir_path, path)

            else:
                if args.verbose:
                    print('Skipping directory by -md: ' + subdir)
    else:
        if args.verbose:
            print('read cache')
        dir_img_f_hash_diff = 0
        dir_img_f_hash_avg = 0
        from_path_len = 0

        # the first 3 lines is: cache file structure version -> total of from_path's lines -> from_path
        # the next 4 lines and so on would be: total of path's lines -> path -> diff hash -> avg hash
        # The reason put total of lines is to support file path containing multiple lines
        from_path = ''
        done_read_header_version = False
        done_read_header_path_count = False
        done_read_header_path = False
        done_read_header = False

        path = ''
        done_read_path_count = False
        done_read_path = False        
        done_read_hash_diff = False

        ei = 0
        for l in hash_cache_f.readlines():
            #print('reading ? ' + str(l))
            if done_read_header:

                if done_read_path_count:

                    if done_read_path:

                        if done_read_hash_diff: # done read hash diff and avg

                            dir_img_f_hash_avg = int(l)

                            ei+=1

                            path = path[:-1] #strip \n (only last line of multiple lines need to strip)

                            co = path[from_path_len+1:].count(OS_SEP)
                            #print('co path: ' + path)
                            #print('co: ' + str(co) + ' from_path_len: ' + str(from_path_len))
                            # --max-depth 0 should do nothing, conforms to find command
                            if (max_depth is None) or (co < max_depth):

                                if share_print(img_f_hash_diff, img_f_hash_avg, dir_img_f_hash_diff, dir_img_f_hash_avg, path,  ei, total_matched):
                                    total_matched += 1
                                    create_symlink(ln_dir_path, path)
                                else:
                                    total_not_match += 1
                                    create_symlink(lnm_dir_path, path)
                            else:
                                if args.verbose:
                                    print('Skipping directory by -md: ' + path)

                            #reset before next read file block
                            path = ''
                            done_read_path_count = False
                            done_read_path = False
                            done_read_hash_diff = False

                        else: # read hash avg
                            dir_img_f_hash_diff = int(l)
                            done_read_hash_diff = True

                    else:
                        path_total-=1
                        path+=l #[:-1] 
                        if path_total == 0:
                            done_read_path = True

                else:
                    #print('reading: ' + str(l))
                    path_total = int(l) 
                    done_read_path_count = True
                            
            elif not done_read_header_version:
                done_read_header_version  = True
                try:
                    curr_cache_file_version = float(l)
                except ValueError:
                    curr_cache_file_version = 1.0 #first version
                if curr_cache_file_version != __cache_file_version__:
                    quit('Cache file structure already changed in new version. Please use -w to re-cache.')

            elif not done_read_header_path_count:
                done_read_header_path_count = True
                path_total = int(l) 
            else:
                path_total-=1
                if path_total == 0:
                    from_path+=l[:-1] # remove trailing \n
                    from_path_len+=len(from_path) #count len in last line, not accum every line
                    if args.verbose:
                        print('cache header path is: ' + from_path)
                    done_read_header = True
                    done_read_path_count = False
                else:
                    from_path+=l

    if not args.norm:
        try:
            img_f.fp.close()
        except AttributeError:
            pass
        if hash_cache_f is not None:
            hash_cache_f.close()
    print()
    if args.c is False:
        # Since -c always 0 so -c no nid print
        # Even file change to non-image intentionally also fine since it only read cache without re-scan at all 
        #, except scan input image.
        if total_not_a_image > 0:
            print(
                '\x1b[0;52mTotal of\x1b[0m\x1b[K \x1b[0;35mnot an image: \x1b[0m\x1b[K' + str(total_not_a_image))

    print('\x1b[1;52mTotal of\x1b[0m\x1b[K \x1b[1;35mimages: \x1b[0m\x1b[K' +
          str(total_not_match + total_matched))
    print('\x1b[1;52mTotal of\x1b[0m\x1b[K \x1b[1;31mnot match: \x1b[0m\x1b[K' +
          str(total_not_match))
    print(
        '\x1b[1;52mTotal of\x1b[0m\x1b[K \x1b[1;32mmatched: \x1b[0m\x1b[K' + str(total_matched))
