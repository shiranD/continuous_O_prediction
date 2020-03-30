import sys
import cupy as cp
import numpy as np
import torch
import pdb

def evaluate_preds(baseline_valid, baseline, termlist, iteration, use_cuda, name, writer):

    max_rank = 11
    acc = 0
    recall = 0
    mrr = 0
    total = 0
    real_l = 0
    all_loss = []
    type10 = {}
    type1 = {}
    all_ranks = []
    loss = torch.nn.CrossEntropyLoss()
    fname = open(name, 'w')
    # eval mode
    if use_cuda:
        loss = loss.cuda()
        for _context, samp_size, target_idx, seqlen, _, read  in baseline_valid:

            if not read:
                break
            # generate a preds
            hidden = baseline.init_hidden(samp_size)
            pred = baseline(_context, hidden, seqlen)
            r, ntoken = pred.shape

            real_l+=float(loss(pred,torch.autograd.Variable(torch.tensor(target_idx, dtype=torch.long)).cuda().view(r)))
            all_loss.append(real_l)

            preds = pred.cpu().data.numpy()
            preds = cp.array(preds)
            target = cp.array(target_idx)

            pred_idxs = preds.argmax(axis=1)
            acc+= cp.sum(target==pred_idxs)
            
            
            ranks = (cp.negative(preds)).argsort(axis=1)
            ranks_of_best = cp.where(ranks==target.reshape(-1,1))[1]
            recip_ranks = 1.0 / cp.add(ranks_of_best,1)
            mrr+=float(cp.sum(recip_ranks[cp.where(recip_ranks > 0.099)[0]])) # below rank 10 it's 0
 
            acc10 = cp.where(ranks_of_best<10)[0]
            recall+=len(acc10)


            for seqlen, (target_idx, rank) in enumerate(zip(target.reshape(r), ranks_of_best)):

                term = termlist[int(target_idx)]
                if int(rank) < max_rank-1:
                    fname.write("the target's rank {:3d} (within)  ".format(int(rank) + 1))
                    type10[term] = True
                    if int(rank) == 0:
                        type1[term] = True
                    all_ranks.append(int(rank)+1)
                else:
                    fname.write(
                        "the target's rank below {} ".format(max_rank - 1))
                total+=1

                fname.write("predicted word: {:20s} ".format(termlist[int(pred_idxs[seqlen])]))
                fname.write("target word: {:20s} ".format(term))
                fname.write("history length: {:3d}".format(seqlen + 1))
                fname.write('\n')
            #if len(all_loss) > 100:
            #    break
    else:
        for _context, samp_size, target_idx, seqlen, _ in baseline_valid:
            # generate a preds
            hidden = baseline.init_hidden(samp_size)
            pred = baseline(_context, hidden, seqlen, mode='eval')
            r, ntoken = pred.shape
            preds = pred.cpu().data.numpy()
            preds = np.array(preds)

            target = np.array(target_idx)

            pred_idxs = preds.argmax(axis=1)
            acc+= np.sum(target==pred_idxs)
            

            ranks = (np.negative(preds)).argsort(axis=1)
            ranks_of_best = np.where(ranks==target.reshape(-1,1))[1]
            recip_ranks = 1.0 / np.add(ranks_of_best,1) 
            mrr+= np.sum(recip_ranks) # not accurate mrr
            acc10 = np.where(ranks_of_best<10)[0]
            recall+=len(acc10)

            term = termlist[int(target_idx)]
            if ranks_of_best[0] < max_rank-1:
                fname.write("the target's rank {} (within)  ".format(ranks_of_best[0]+1))
                type10[term] = True
                if ranks_of_best[0] == 0:
                    type1[term] = True
                all_ranks.append(ranks_of_best[0]+1)
            else:
                fname.write(
                    "the target's rank below {} ".format(max_rank-1))
            total+=1

            real_l+=loss(pred,torch.autograd.Variable(torch.tensor(target_idx, dtype=torch.long)).view(r))
            all_loss.append(real_l)
            fname.write("target word: {} ".format(term))
            fname.write("history length: {} ".format(seqlen))
            fname.write('\n')
            
    fname.close()
    if writer:
        writer.add_histogram('valid_loss_histogram', np.array(all_loss), global_step=iteration)
        if len(all_ranks) > 1:
            writer.add_histogram('valid_high_ranks_histogram', np.array(all_ranks), global_step=iteration)
        writer.add_scalar('valid_types_10', len(type10) , iteration)
        writer.add_scalar('valid_types_1', len(type1) , iteration)
        writer.add_scalar('valid_top_1', float(acc)/total*100, iteration)
        writer.add_scalar('valid_top_10', float(recall)/total*100, iteration)
        writer.add_scalar('valid_mrr', float(mrr)/total, iteration)

    # send back
    print(
        "total of {} tokens with acc of {:2.4f} mrr of {:2.4f} recall at {} of {:2.4f} valid loss {}".format(
            total,
            float(acc) /
            total *
            100,
            float(mrr) /
            total,
            max_rank-1,
            float(recall) /
            total *
            100, float(real_l) / total))

    sys.stdout.flush()
    return real_l / total, writer 
