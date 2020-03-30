import sys
import os
import argparse
import json
import pdb
import json

parser = argparse.ArgumentParser(description='Process results')
parser.add_argument('--folder', type=str, help='location of the results folder')
parser.add_argument('--sett', type=str, help='test set kwd')
parser.add_argument('--termdir', type=str, help='directory for old, new, and shared terms')
parser.add_argument('--number', type=int, help='file number to process')
args = parser.parse_args()

with open(args.termdir+'bl_high_'+args.sett, 'r') as fp:
    high, highf = json.load(fp)

with open(args.termdir+'bl_mid1_'+args.sett, 'r') as fp:
    mid1, mid1f = json.load(fp)

with open(args.termdir+'bl_mid2_'+args.sett, 'r') as fp:
    mid2, mid2f = json.load(fp)

with open(args.termdir+'bl_low_'+args.sett, 'r') as fp:
    low, lowf = json.load(fp)

with open(args.termdir+'bl_oov_'+args.sett, 'r') as fp:
    oov, oovf = json.load(fp)

total = 0
types10 = {}
types1 = {}
top10 = 0
top1 = 0
mrr = 0

high_found = {}
mid1_found = {}
mid2_found = {}
low_found = {}
oov_found = {}
rest_found = {}

sum_high = 0
sum_mid1 = 0
sum_mid2 = 0
sum_low = 0
sum_oov = 0
rest = 0

for i in range(args.number):
    i = str(i)
    exists = os.path.isfile(args.folder+'/preds/results_'+i+'.txt')
    if not exists:
        continue 
    else:
        fname = args.folder+'/preds/results_'+i+'.txt'
        with open(fname, 'r') as fp:
            d = json.load(fp)
        fname = args.folder+'/preds/tk_results_'+i+'.txt'
        with open(fname, 'r') as fp:
            d_tk = json.load(fp)

    for key in d_tk['tokens']:
        if key in high:
            sum_high+=d_tk['tokens'][key]
            high_found[key]=True
        elif key in mid1:
            sum_mid1+=d_tk['tokens'][key]
            mid1_found[key]=True
        elif key in mid2:
            sum_mid2+=d_tk['tokens'][key]
            mid2_found[key]=True
        elif key in low:
           sum_low+=d_tk['tokens'][key]
           low_found[key]=True
        elif key in oov:
           sum_oov+=d_tk['tokens'][key]
           oov_found[key]=True
        else:
           rest+=d_tk['tokens'][key]
           rest_found[key] = True

    for key in d['type1']:
        types1[key]=True
    for key in d['type10']:
        types10[key]=True

    total+= d['total']
    top1+=  d['top1']
    top10+= d['top10']
    mrr+=   d['mrr']

f = open(args.folder+'/final', 'w')

print(sum_high, highf, len(high_found), len(high))
print(sum_mid2, mid2f, len(mid2_found), len(mid2))
print(sum_mid1, mid1f, len(mid1_found), len(mid1))
print(sum_low, lowf, len(low_found), len(low))
print(sum_oov, oovf, len(oov_found), len(oov))

f.write('total is {}\n'.format(total)) 
f.write('top10 {}\n'.format(top10/total*100)) 
f.write('top1 {}\n'.format(top1/total*100)) 
f.write('mrr {}\n'.format(mrr/total)) 
f.write('types 10 {}\n'.format(len(list(types10.keys()))))
f.write('types 1 {}\n'.format(len(list(types1.keys()))))
f.write('& {:2.2f}({:2.2f}) & {}({}) & {:1.3f} \\\\ \n'.format(top1/total*100, top10/total*100, len(list(types1.keys())), len(list(types10.keys())), mrr/total))
# types (partial)
f.write('low {}\n'.format(len(low_found)/len(low)*100))
f.write('mid1 {}\n'.format(len(mid1_found)/len(mid1)*100))
f.write('mid2 {}\n'.format(len(mid2_found)/len(mid2)*100))
f.write('high {}\n'.format(len(high_found)/len(high)*100))

# events (partial)
f.write('low {}\n'.format(sum_low/lowf*100))
f.write('mid1 {}\n'.format(sum_mid1/mid1f*100))
f.write('mid2 {}\n'.format(sum_mid2/mid2f*100))
f.write('high {}\n'.format(sum_high/highf*100))
