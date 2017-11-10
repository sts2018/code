import string
import urllib
import bs4
import ssl
import sys
import getopt
from PIL import Image
import threading
import timeit
import os

def set_up_dir(top):
    # Create the dir if it does not exist
    if not os.path.exists(top):
        os.makedirs(top)

    # Delete everything reachable from the directory named in 'top',
    # assuming there are no symbolic links.
    # CAUTION:  This is dangerous!  For example, if top == '/', it
    # could delete all your disk files.

    for root, dirs, files in os.walk(top, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
            # for name in dirs:
            #    os.rmdir(os.path.join(root, name))

# stripe 82
address='http://third.ucllnl.org/cgi-bin/stripe82image'

# parameters
Equinox='J2000'
ImageSize='1'
ImageType='FITS Image'
MaxInt='0'
Dir='index'
SourceFile='test_data.txt'
TypePos=20
Index=0
First=0
NoCrop=0

def process_config(line):
    global address, First, Index, ImageSize, TypePos, Dir, NoCrop, ImageType, MaxInt
    items=string.split(line)
    kv=items[1].split('=')

    if (kv[0] == "Dir"): 
        Dir=kv[1]
        set_up_dir(Dir)
    elif (kv[0] == "First"):
        First=1
        address="http://third.ucllnl.org/cgi-bin/firstimage"
    elif (kv[0] == "First_FITS"):
        First=1
        address="http://third.ucllnl.org/cgi-bin/firstcutout"
        ImageType='FITS File'
    elif (kv[0] == "ImageSize"):
        ImageSize=int(kv[1])
    elif (kv[0] == "MaxInt"):
        MaxInt=int(kv[1])
    elif (kv[0] == "TypePos"):
        TypePos=int(kv[1])
    elif (kv[0] == "NoCrop"):
        NoCrop = 1
    elif (kv[0] == "Print"):
        print 'Config:'
    else:
        print "Invalid config."
        exit(1)
    
def write_to_file(fileName, data):
    f=open(fileName,'w')
    f.write(data)
    f.close()


def process_one_image_by_br(objName, RA, Dec):
    import re
    from mechanize import Browser
    br = Browser()

    # Ignore robots.txt
    br.set_handle_robots( False )
    br.addheaders = [('User-agent', 'Firefox')]

    br.open("http://third.ucllnl.org/cgi-bin/stripe82cutout")
    
    # Form is unnamed
    br.form = list(br.forms())[0] 
    br.form[ 'RA' ] = RA + ' ' + Dec
    br.form[ 'ImageSize'] = ImageSize
    br.form[ 'ImageType' ] = ['FITS File']
    br.form[ 'MaxInt' ] = MaxInt

    # Get the results
    response = br.submit()
    returned = response.read()

    write_to_file(objName+'.fits', returned)    


def process_one_image(objName, RA, Dec):

    global First, ImageSize, NoCrop, ImageType
    one_url_time = 0

    if ImageType == 'FITS File':
        outfile=objName+'.fits'
    else:
        outfile=objName+'.gif'
    
    # GET this will download the file
    parameters = urllib.urlencode({'Equinox': Equinox, 'RA': RA, 'Dec': Dec, 'ImageSize': ImageSize, 'ImageType': ImageType, 'MaxInt': MaxInt})
    # print parameters

    # GET this will return the page with image embedded
    # parameters = urllib.urlencode({'Equinox': Equinox, 'RA': RA, 'Dec': Dec, 'ImageSize': ImageSize, 'MaxInt': MaxInt})

    start_time = timeit.default_timer()

    if (sys.version_info[0] >= 2 and sys.version_info[1] >= 7 and sys.version_info[2] >= 9):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        form = urllib.urlopen(address+'?%s' % parameters, context=ctx)
    else:
        form = urllib.urlopen(address+'?%s' % parameters)

    one_url_time += ((timeit.default_timer() - start_time))

    # print form.geturl()
    returned = form.read()
    # print returned
    # soup = bs4.BeautifulSoup(form, "lxml")

    # imgs = soup.findAll("img",{"alt":True, "src":True})
    # for img in imgs:
    #    img_url = img["src"]
    #    print address+img_url
    #    img_data = urllib.urlopen(address+img_url)
    #    data = img_data.read()
    #    f = open("test.gif","wb")
    #    f.write(data)
    #    f.close()
    #    break

    if 'Field Not Found' in returned:
        print 'Field Not Found'
        #objName,RA,Dec
    
    f=open(outfile,'w')
    f.write(returned)
    f.close()

    if (NoCrop == 1):
        return one_url_time

    im = Image.open(outfile)
    if (ImageSize == 1):
        if (First == 0):
            imc = im.crop((55, 16, 255, 216))
        else:
            imc = im.crop((56, 16, 220, 182))
    elif (ImageSize == 2):
        if (First == 0):
            imc = im.crop((59, 16, 260, 218))
        else:
            imc = im.crop((59, 16, 261, 218))
    elif (ImageSize == 4):
        if (First == 0):
            imc = im.crop((51, 16, 451, 416))
        else:
            imc = im.crop((51, 16, 316, 281))

    else:
        imc = im
    imc.save(outfile)    

    return one_url_time


def get_count(list):
    return (int(list[0])*3600 + int(list[1])*60 + int(list[2]))


def process_one_line(line):

    count=0
    total_url_time = 0
    items=string.split(line)

    RAlist=items[1].split(':')
    Declist=items[2].split(':')
    RAlist_end = items[3].split(':')
    duration = get_count(RAlist_end) - get_count(RAlist)

    set_up_dir(items[0])

    for i in range(duration):
        # print RAlist[0], RAlist_end[0], RAlist[1], RAlist_end[1], RAlist[2], RAlist_end[2]
        objName = items[0]+'/'+str(RAlist[0])+'_'+str(RAlist[1])+'_'+str(RAlist[2]) + '_' + Declist[0] + '_' + Declist[1] + '_' + Declist[2]
        # print 'Name is : ',objName
        RA=str(RAlist[0])+' '+str(RAlist[1])+' '+str(RAlist[2])
        Dec=Declist[0]+' '+Declist[1]+' '+Declist[2]
        total_url_time += process_one_image(objName, RA, Dec)
        # process_one_image_by_br(objName, RA, Dec)

        RAlist[2] = int(RAlist[2]) + 1;
        if (RAlist[2] >= 60):
            RAlist[2] = 0 
            RAlist[1] = int(RAlist[1]) + 1;
            if (RAlist[1] >= 60) :
                      RAlist[1] = 0
                      RAlist[0] = int(RAlist[0]) + 1

    print 'Fetched', duration, 'coorindates in', round(total_url_time, 2)


def scan_images(file):
    while 1:
        line=file.readline()
        if not line: break
        if '#' in line:
            process_config(line)
            continue
        else:
            t = threading.Thread(target=process_one_line, args=(line,))
            threads.append(t)
            t.start()

    for t in threads:
        t.join()


def process_one_index(line):

    line=line.replace("+","") # Hack: remove + sign
    items=string.split(line)

    RAlist=[items[0], items[1], items[2]]
    Declist=[items[3], items[4], items[5]]

    # print Dir+'/'+str(RAlist[0])+'_'+str(RAlist[1])+'_'+str(RAlist[2]) + '_' + Declist[0] + '_' + Declist[1] + '_' + Declist[2] + "_" + items[TypePos]
    objName = Dir+'/'+str(RAlist[0])+'_'+str(RAlist[1])+'_'+str(RAlist[2]) + '_' + Declist[0] + '_' + Declist[1] + '_' + Declist[2] + "_" + items[TypePos]
    RA=RAlist[0]+' '+RAlist[1]+' '+RAlist[2]
    Dec=Declist[0]+' '+Declist[1]+' '+Declist[2]
    return process_one_image(objName, RA, Dec)


def index_images(file):
    count=0
    total_url_time = 0
    while 1:
        line=file.readline()
        if not line: break
        if '#' in line:
            process_config(line)
            continue
        total_url_time += process_one_index(line)
        count = count + 1
    print 'Fetched', count, 'coorindates in', round(total_url_time, 2)
    
# Main function
uname = ''
pword = ''
num_days = 1
replace = 0
total_time = 0

try:
    opts, args = getopt.getopt(sys.argv[1:],"if:at:")
except getopt.GetoptError:
   print '-f'
   sys.exit(2)

for opt, arg in opts:
    if opt in '-f':
        SourceFile = arg
    elif opt in '-i':
        Index = 1
    elif opt in '-a':
        First = 1;
        address = 'http://third.ucllnl.org/cgi-bin/firstimage'
    elif opt in '-t':
        type_pos = int(arg)
    else:
        print '-f'
        sys.exit(2)

file  = open(SourceFile, 'r')
threads = []
start_time = timeit.default_timer()

if (Index):
    index_images(file)
else:
    scan_images(file)

total_time = ((timeit.default_timer() - start_time))
print 'Done', len(threads), 'in', round(total_time, 2)
