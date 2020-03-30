import sys
from space import build_space, cosine_dist
import numpy as np
import pdb

def evaluate_preds(G_valid, netG, veclist, termlist, iteration, emdim, name, writer, pos_dict):

    num_vecs = len(veclist)
    max_rank = 11
    just_in_case_max_rank = 50 
    acc = 0
    recall = 0
    mrr = 0
    total = 0
    acc_pos = 0
    recall_pos = 0
    mrr_pos = 0
    sim_loss  = 0
    space = build_space(emdim, veclist)
    space.build(100)
    all_loss = []
    all_ranks = []
    all_ranks_pos = []
    type1 = {}
    type10 = {}
    type1_pos = {}
    type10_pos = {}
    fname = open(name, 'w')
    fname_pos = open(name+'_pos', 'w')
    for _fake_input, samp_size, target_idx, seqlen, pos, read in G_valid:
   
        if not read:
            break    
        # generate a fake embeddin one pass complete sentence
        hiddenG = netG.init_hidden(samp_size)
        fake = netG(_fake_input, hiddenG, seqlen)
        fake = fake.view(emdim)
        fake = fake.cpu().data.numpy()

        target_idx = target_idx[0]
        # find its nearest neighbor
        nearest, dist= space.get_nns_by_vector(fake, n=just_in_case_max_rank, search_k=-1, include_distances=True)

        # No POS decoding
        rank = -1
        for g, friend in enumerate(
                nearest[:10]):
            if friend == target_idx:
                rank = g + 1
                type10[termlist[target_idx]] = True
                break
        target_term = termlist[target_idx]
    
        # compute and report
        if rank != -1:
            fname.write("the target's rank {:3d} (within)  ".format(rank))
            all_ranks.append(rank)
            recall += 1
            if rank == 1:
                acc += 1
                type1[termlist[target_idx]] = True
            mrr += 1 / rank
        else:
            fname.write(
                "the target's rank below {} ".format(max_rank-1))
        total+=1
        similarity = cosine_dist(fake, veclist[target_idx])
        sim_loss+=1-similarity
        all_loss.append(similarity)
 
        fname.write("predicted word: {:20s} ".format(termlist[nearest[0]]))
        fname.write("target word: {:20s} ".format(target_term))
        fname.write("distance: {:1.9f} ".format(similarity))
        fname.write("history length: {:3d}".format(seqlen+1))
        fname.write('\n')

        
        # POS decoding
        rank = -1
        g = 0
        pred_word = False
        target_pos = pos
        for friend in nearest:
            friend_pos = pos_dict[termlist[friend]]
            if target_pos in friend_pos:
                if friend == target_idx:
                    rank = g + 1
                    type10_pos[termlist[target_idx]] = True
                if g == 0: # closest vector
                    pred_word = friend
                g+=1
                if g == 10:
                    break
        target_term = termlist[target_idx]

        if rank != -1:
            fname_pos.write("the target's rank {:3d} (within)  ".format(rank))
            all_ranks_pos.append(rank)
            recall_pos += 1
            if rank == 1:
                acc_pos += 1
                type1_pos[termlist[target_idx]] = True
            mrr_pos += 1 / rank
        else:
            fname_pos.write(
                "the target's rank below {} ".format(max_rank-1))
        if pred_word:
            fname_pos.write("predicted word: {:20s} ".format(termlist[pred_word]))
        fname_pos.write("target word: {:20s} ".format(target_term))
        fname_pos.write("distance: {:1.9f} ".format(similarity))
        fname_pos.write("history length: {:3d}".format(seqlen+1))
        fname_pos.write('\n')
       # if len(all_loss)>100:
       #     break
    fname.close()
    fname_pos.close()

    if writer:
        writer.add_histogram('valid_loss_histogram', np.array(all_loss), global_step=iteration)
        if len(all_ranks) > 1: 
            writer.add_histogram('valid_high_NOPOS_ranks_histogram', np.array(all_ranks), global_step=iteration)
        if len(all_ranks_pos) > 1: 
            writer.add_histogram('valid_high_ranks_POS_histogram', np.array(all_ranks_pos), global_step=iteration)
        writer.add_scalar('valid_types_10', len(type10) , iteration)
        writer.add_scalar('valid_types_1', len(type1) , iteration)
        writer.add_scalar('valid_types_10_pos', len(type10_pos) , iteration)
        writer.add_scalar('valid_types_1_pos', len(type1_pos) , iteration)
        writer.add_scalar('valid_top_1', acc/total*100, iteration)
        writer.add_scalar('valid_top_10', recall/total*100, iteration)
        writer.add_scalar('valid_top_1_pos', acc_pos/total*100, iteration)
        writer.add_scalar('valid_top_10_pos', recall_pos/total*100, iteration)
        writer.add_scalar('valid_mrr', mrr/total, iteration)
        writer.add_scalar('valid_mrr_pos', mrr_pos/total, iteration)

    print(
        "No POS: total of {} tokens with acc of {:2.4f} mrr of {:2.4f} recall at {} of {:2.4f} cosim loss {:2.4f}".format(
            total,
            acc /
            total *
            100,
            mrr /
            total,
            max_rank-1,
            recall /
            total *
            100,
            sim_loss / 
            total))
    print(
        "   POS: total of {} tokens with acc of {:2.4f} mrr of {:2.4f} recall at {} of {:2.4f} cosim loss {:2.4f}".format( 
            total,
            acc_pos /
            total *
            100,
            mrr_pos /
            total,
            max_rank-1,
            recall_pos /
            total *
            100,
            sim_loss / 
            total))

    sys.stdout.flush()
    # send back
    return sim_loss / total, writer
