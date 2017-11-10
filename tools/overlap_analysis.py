import string
import urllib
import sys
import getopt

import deg2HMS as d2t

last_pos = 0

def check_if_exist(i_ra, i_dec, file):
    global last_pos
    file.seek(last_pos)
    while 1:
        line=file.readline()
        if not line: break
        items=line.split(' ')
        ra = items[0] + ' ' + items[1] + ' ' + items[2]
        dec = items[3] +' ' + items[4] + ' ' + items[5]
        s_delta_ra = d2t.s_deltaRA(ra, i_ra)
        delta_ra = abs(s_delta_ra)
        delta_dec = d2t.deltaDEC(dec, i_dec)
        # DEC 5 arcmin; RA 20 arcsecond
        if (delta_ra <= float(0.25/3) and delta_dec <= float(0.25/3)):
            # print i_ra, i_dec, ', ', line
            return 1
        if (s_delta_ra > float(0.25/3)):
            break
        if (s_delta_ra <= float(-0.25/3)):
            # move up the file seek location
            last_pos = file.tell()
    print i_ra, i_dec, ', ', line        
    return 0

def process_overlap_2_files(i_file, c_file):

    i_count = 0
    count_1 = 0
    while 1:
        line=i_file.readline()
        if not line: break
        i_count = i_count + 1
        items=line.split(' ')
        ra = items[0] + ' ' + items[1] + ' ' + items[2]
        dec = items[3] +' ' + items[4] + ' ' + items[5]
        if check_if_exist(ra, dec, c_file):
            count_1 = count_1 + 1
    print 'Processed', i_count, 'Found', count_1

def process_overlap(file):

    last_line = ' '
    last_ra = '0 0 0'
    last_dec = '0 0 0'
    last_cat = ' '
    count = 0
    count_1 = 0
    delta = 0 
    count_no_dr1 = 0
    while 1:
        line=file.readline()
        if not line: break
        count = count + 1
        # print line
        items=line.split(' ')
        ra = items[0] + ' ' + items[1] + ' ' + items[2]
        dec = items[3] +' ' + items[4] + ' ' + items[5]
        cat = items[6]
        delta_ra = d2t.deltaRA(ra, last_ra)
        delta_dec = d2t.deltaDEC(dec, last_dec)
        # DEC 5 arcmin; RA 20 arcsecond
        # if ('dr1' not in cat):
        #    count_no_dr1 = count_no_dr1 + 1
        #    print cat
        if (delta_ra <= float(0.25/3) and delta_dec <= float(0.25/3)):
            # print delta_ra, delta_dec
            if (cat != last_cat):
                print cat, last_line[:-1], ',', line,
                count_1 = count_1 + 1
        # elif ('dr1' not in cat):        
        #    print last_line, line
        last_ra = ra
        last_dec = dec
        last_cat = cat
        last_line = line
        #if is_in_first_range(ra, dec):
        #    print ra_dec[0], ra_dec[1], 'dr1'
        # curr_seq = int(items[0])
        # if (curr_seq == (last_seq + 1)):
        #    last_ra = curr_ra
        #else:
        #    print last_ra, curr_ra
        #    last_seq = curr_seq
    print 'Processed', count, 'No DR1', count_no_dr1, 'Found', count_1
    # print first_count, first_count[0]+first_count[1]+first_count[2],first_count[3]
    # print s82_count, s82_count[0]+s82_count[1]

# Main function
try:
    opts, args = getopt.getopt(sys.argv[1:],"f:g:")
except getopt.GetoptError:
   print '-f'
   sys.exit(2)

for opt, arg in opts:
    if opt in '-f':
        SourceFile = arg
        file  = open(SourceFile, 'r')
        process_overlap(file)
    elif opt in '-g':
        Files = arg.split(',')
        i_file = open(Files[0], 'r')
        c_file = open(Files[1], 'r')
        process_overlap_2_files(i_file, c_file)
    else:
        print '-f'
        sys.exit(2)


