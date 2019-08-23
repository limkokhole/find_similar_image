# find_similar_image
Find similar image by specific image.

### Usage:

    xb@dnxb:~/Downloads/find_similar_image$ python3 similar_img.py --help
    usage: similar_img.py [-h] [-d DIR_TO_SCAN] [-md MAX_DEPTH] [-v] [-n] [-m]
                        [-c [C]] [-w [W]] [-s SHOW] [-l LN] [-lnm LNM] [-f] [-u]
                        [-p PERCENTAGE] [--version]
                        [image_path]

    Find simlar images in specific directory by specific image

    positional arguments:
    image_path            Specify single image path to check.

    optional arguments:
    -h, --help            show this help message and exit
    -d DIR_TO_SCAN, --dir DIR_TO_SCAN
                            Directory path to scan recursively.    
                            You can't use -d together with -c
    -md MAX_DEPTH, --max-depth MAX_DEPTH
                            Specify max depth of directory path to scan.    
                            If you use -md together with -c then its max depth depends on the path of cache file without re-scan.
    -v, --verbose         Shows more log.    
                            You probably want to remove -m too.
    -n, --norm            Use Manhattan norm which not allow cache.    
                            It has higher tolerance in different size but lower tolerance in bright.    
                            Default are Average and Difference hash(pick higher) which allow cache with -c/-w.
    -m                    Only shows matched line.
    -c [C]                Read this cache file without rescan images. Default is hash.cache if -c without value.    
                            You should use -w once to generate new cache file before use -c
    -w [W]                Write current result to this cache file. Default is hash.cache if -w without value.
    -s SHOW, --show SHOW  Shows Nth matched images in popup viewer.    
                            0 means unlimited but be aware it probably hang your system if too much viewers popup.
    -l LN, --link-match LN
                            Create symlink of matched images in this directory.
    -lnm LNM, --link-not-macth LNM
                            Create symlink of not-matched images in this directory.
    -f, --follows-symlink
                            Follows symlink for files and directory.    
                            Default is don't follows symlink to avoid scan duplicated (-s will popup twice) files in -l directory.
    -u, --unique-ln       Create symlink with unique filename by uuid.    
                            This option do nothing without -l
    -p PERCENTAGE, --percentage PERCENTAGE
                            Match by diff more or equal to this percentage floating value (without %).    
                            Default is 80.0%. You should try lower it to 75% or 70% if it doesn't match a few.
    --version             show program's version number and exit

