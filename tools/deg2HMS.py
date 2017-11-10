# http://www.bdnyc.org/2012/10/decimal-deg-to-hms/

# 
# ra dec rastring decstring
# 309.3204 -1.1483592 37:16.9 -01:08:54.1
# https://www.swift.psu.edu/secure/toop/convert.htm
# RA: 20:37:17 - > 309.3208

def deg2HMS(ra='', dec='', round=False):
  RA, DEC, rs, ds = '', '', '', ''
  if dec:
    if str(dec)[0] == '-':
      ds, dec = '-', abs(dec)
    deg = int(dec)
    decM = abs(int((dec-deg)*60))
    if round:
      decS = int((abs((dec-deg)*60)-decM)*60)
    else:
      decS = (abs((dec-deg)*60)-decM)*60
    DEC = '{0}{1} {2} {3}'.format(ds, deg, decM, decS)
  
  if ra:
    if str(ra)[0] == '-':
      rs, ra = '-', abs(ra)
    raH = int(ra/15)
    raM = int(((ra/15)-raH)*60)
    if round:
      raS = int(((((ra/15)-raH)*60)-raM)*60)
    else:
      raS = ((((ra/15)-raH)*60)-raM)*60
    RA = '{0}{1} {2} {3}'.format(rs, raH, raM, raS)
  
  if ra and dec:
    return (RA, DEC)
  else:
    return RA or DEC

# http://www.bdnyc.org/tag/python/

def HMS2deg(ra='', dec=''):
  RA, DEC, rs, ds = '', '', 1, 1
  if dec:
    D, M, S = [float(i) for i in dec.split()]
    if str(D)[0] == '-':
      ds, D = -1, abs(D)
    deg = D + (M/60) + (S/3600)
    # DEC = '{0}'.format(deg*ds)
    DEC = deg*ds

  if ra:
    H, M, S = [float(i) for i in ra.split()]
    if str(H)[0] == '-':
      rs, H = -1, abs(H)
    deg = (H*15) + (M/4) + (S/240)
    # RA = '{0}'.format(deg*rs)
    RA = deg*rs

  if ra and dec:
    return (RA, DEC)
  else:
    return RA or DEC


def deltaRA(ra1='', ra2=''):
  return abs(s_deltaRA(ra1, ra2))

def s_deltaRA(ra1='', ra2=''):
  RA1, RA2, rs1, rs2 = 0, 0, 1, 1

  if ra1:
    H, M, S = [float(i) for i in ra1.split()]
    if str(H)[0] == '-':
      rs1, H = -1, abs(H)
    deg = (H*15) + (M/4) + (S/240)
    RA1 = deg*rs1

  if ra2:
    H, M, S = [float(i) for i in ra2.split()]
    if str(H)[0] == '-':
      rs2, H = -1, abs(H)
    deg = (H*15) + (M/4) + (S/240)
    RA2 = deg*rs1

  return (RA1 - RA2)


def deltaDEC(dec1='', dec2=''):
  DEC1, DEC2, ds1, ds2 = 0, 0, 1, 1

  if dec1:
    D, M, S = [float(i) for i in dec1.split()]
    if str(D)[0] == '-':
      ds1, D = -1, abs(D)
    deg = D + (M/60) + (S/3600)
    DEC1 = deg*ds1

  if dec1:
    D, M, S = [float(i) for i in dec2.split()]
    if str(D)[0] == '-':
      ds2, D = -1, abs(D)
    deg = D + (M/60) + (S/3600)
    DEC2 = deg*ds2

  return abs(DEC1 - DEC2)

# DEC 10 armin = RA 40 arcsecond
# DEC 5 armin = RA 20 arcsecond                                                                                                                                                                            

# RA
# 1arcmin = 0.25
# 20 arcsecond = 0.0833333333333
# print deltaRA(ra1='16 46 43.0186284', ra2='16 47 43.0186284')
# print s_deltaRA(ra1='16 46 03.0186284', ra2='16 46 23.0186284')
                                                                                                                                             
# DEC 
# 1arcmin = 0.0166666666667
# 5arcmin = 0.0833333333333
# print deltaDEC(dec1='23 23 -56.584354', dec2='23 22 -56.584354')
# print deltaDEC(dec1='23 23 -56.584354', dec2='23 28 -56.584354')                                                                                                                                          

#print deg2HMS(ra=082603,dec=471910.3)
#print deg2HMS(251.679244285,23.382106765)
#print HMS2deg(ra='16 46 43.0186284', dec='23 22 55.584354')
#print HMS2deg(ra='16 47 43.0186284', dec='23 23 55.584354')
#print deg2HMS(ra=309.3204)
#print deg2HMS(ra=82603)
#print deg2HMS(334.438, +00.413)
