import os
import sys
import argparse
import json
import pdb

parser = argparse.ArgumentParser(description='Process results')
parser.add_argument('--folder', type=str, help='location of the results folder')
parser.add_argument('--pos', action='store_true', default=False, help='decode based on POS')
parser.add_argument('--number', type=int, help='file number to process')
parser.add_argument('--termdir', type=str, help='directory for old, new, and shared terms')
parser.add_argument('--sett', type=str, help='test set kwd')
args = parser.parse_args()

total = 0
types10 = {}
types1 = {}
top10 = 0
top1 = 0
mrr = 0

with open(args.termdir+'high_'+args.sett, 'r') as fp:
    high, highf = json.load(fp)

with open(args.termdir+'mid1_'+args.sett, 'r') as fp:
    mid1, mid1f = json.load(fp)
    
with open(args.termdir+'mid2_'+args.sett, 'r') as fp:
    mid2, mid2f = json.load(fp)

with open(args.termdir+'low_'+args.sett, 'r') as fp:
    low, lowf = json.load(fp)


high_found = {}
mid1_found = {}
mid2_found = {}
low_found = {}

sum_high = 0
sum_mid1 = 0
sum_mid2 = 0
sum_low = 0

for i in range(args.number):
    i = str(i)
    exists = os.path.isfile(args.folder+'/results/summary_'+i)
    if not exists:
        continue 
    if args.pos:
        fname = args.folder+'/results/summary_pos_'+i
        with open(args.folder+'/results/type1_pos_'+i, 'r') as fp:
            type1 = json.load(fp)
        with open(args.folder+'/results/type10_pos_'+i, 'r') as fp:
            type10 = json.load(fp)
    else:
        fname = args.folder+'/results/summary_'+i
        with open(args.folder+'/results/type1_'+i, 'r') as fp:
            type1 = json.load(fp)
        with open(args.folder+'/results/type10_'+i, 'r') as fp:
            type10 = json.load(fp)

    for key in type1.keys():
        if key in high:
            sum_high+=type1[key]
            high_found[key]=True
        elif key in mid1:
            sum_mid1+=type1[key]
            mid1_found[key]=True
        elif key in mid2:
            sum_mid2+=type1[key]
            mid2_found[key]=True
        elif key in low:
           sum_low+=type1[key]
           low_found[key]=True
        else:
           print(key)

    for key in type1.keys():
        types1[key]=True
    for key in type10.keys():
        types10[key]=True
    for line in open(fname, 'r').readlines():
        line = line.strip()
        line = line.split()
        total+=int(line[2])
        top1+=int(line[7]) 
        top10+=int(line[13]) 
        mrr+=float(line[10])

if args.pos:
    f = open(args.folder+'/final_pos', 'w')
else:
    f = open(args.folder+'/final', 'w')

print(sum_high, highf, len(high_found), len(high))
print(sum_mid2, mid2f, len(mid2_found), len(mid2))
print(sum_mid1, mid1f, len(mid1_found), len(mid1))
print(sum_low, lowf, len(low_found), len(low))
print(sum_oov, oovf, len(oov_found), len(oov))

f.write('{} pos is {}\n'.format(args.folder, args.pos))
f.write('total is {}\n'.format(total)) 
f.write('top10 {}\n'.format(top10/total*100)) 
f.write('top1 {}\n'.format(top1/total*100)) 
f.write('mrr {}\n'.format(mrr/total)) 
f.write('types10 {}\n'.format(len(types10)))
f.write('types1 {}\n'.format(len(types1)))
f.write('& {:2.2f}({:2.2f}) & {}({}) & {:1.3f} \\\\ \n'.format(top1/total*100, top10/total*100, len(types1.keys()), len(types10.keys()), mrr/total))

# types
f.write('low {}\n'.format(len(low_found)/len(low)*100))
f.write('mid1 {}\n'.format(len(mid1_found)/len(mid1)*100))
f.write('mid2 {}\n'.format(len(mid2_found)/len(mid2)*100))
f.write('high {}\n'.format(len(high_found)/len(high)*100))

# events
f.write('low {}\n'.format(sum_low/lowf*100))
f.write('mid1 {}\n'.format(sum_mid1/mid1f*100))
f.write('mid2 {}\n'.format(sum_mid2/mid2f*100))
f.write('high {}\n'.format(sum_high/highf*100))
