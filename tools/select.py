# Uaage:
#
# python select.py -n 10 stripe82_2010may25_conv (random shuffle and pick item 1-10)
# python select.py -b 10 -n 10 -f stripe82_2010may25_conv (random shuffle and pick item 11-20)
#
import sys
import random
import getopt

try:
    opts, args = getopt.getopt(sys.argv[1:],"b:n:f:")
except getopt.GetoptError:
   print '-f'
   sys.exit(2)

SourceFile = ''
begin_line = 0
num_lines = 1

for opt, arg in opts:
    if opt in '-b':
        begin_line = int(arg)
    elif opt in '-n':
        num_lines = int(arg)
    elif opt in '-f':
        SourceFile = arg
    else:
        print '-n'
        sys.exit(2)

file = open(SourceFile, 'r')
while 1:
    line=file.readline()
    if '#' in line:
        print line.strip()
        continue
    else:
        break

flist = [line, ]
flist.extend(file.readlines())
random.seed(0)
random.shuffle(flist)

count = 0
for line in flist:
    if '#' in line:
	continue
    if begin_line > 0:
        begin_line = begin_line- 1
        continue
    print line.strip()
    count = count + 1
    if (count == num_lines):
        sys.exit(0)

