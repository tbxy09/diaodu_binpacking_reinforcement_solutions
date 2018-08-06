import re
t=[
    'app_5732 app_5732app_5732app_5223app_1148'
   +'app_1148app353'
   +'app_1148app_1148',
    'app_5732 app_5732app_5223app_1148'
   +'app_1148app353',
    'app_5732app_5732app_5223app_1148',
    'app_5732app_5732app_5223app_453app_1148',
  ]

p= ['app_5732[^(  )]*app_1148[^(  )]*app_1148[^(  )]*app_1148[^( )]*app_1148']
p.append('app_5732.*app_1148')
p.append('app_5732?.*?app_1148')
p.append('app_5732{1}[^( )]*app_1148')
p.append('app_5732[^( 7)]*?app_1148')
# t=['app_3432','app3222','app6422','5712',env.app_inter.ab[5],'app_5732.*?app_1148.*?app_1148']
print('-----------')
[print(re.findall(p[0],each),p[0]) for each in t]
print('-----------')
[print(re.findall(p[1],each),p[1]) for each in t]
print('-----------')
[print(re.findall(p[2],each),p[2]) for each in t]
print('-----------')
[print(re.findall(p[3],each),p[3]) for each in t]
print('-----------')
[print(re.findall(p[4],each),p[4]) for each in t]
