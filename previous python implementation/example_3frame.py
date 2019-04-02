#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
last mod 3/11/19
"""

import numpy as np
from time import time
#from daSparse import da, allocateWorkVarsforDA
#from sparsity import sparsify
from daDense import da, allocateWorkVarsforDA
from sspDense import SSP # used for evaluation



ntests = 10
max_ns = 1000
max_nhyp = 1000
s = 10
entryrate = 100 # poisson rate of object entry
fpratio = .005 # poisson rate of fp msmts, wrt entry rate
detect_rate = .995 # detection probability at each time
std = .001 # standard deviation of msmt noise
miss_distance_cutoffs = np.arange(.1,1.01,.1)
np.random.seed(34)


fprate = fpratio*entryrate
def likelihood1(c, msmts1, msmts2):
    # constant term in NLL of normal distribution
    var_constant = 4*std**2 # normalizing out inv(Sigma)
    constant_term = .5*np.log(np.pi*var_constant)
    constant_term += np.log(fpratio/detect_rate+1-detect_rate)*2 + np.log(entryrate)
    # get nll of all pairs from first two measurement sets
    for i,msmti in enumerate(msmts1):
        for j,msmtj in enumerate(msmts2):
            c[i,j] = np.square(msmti[0]-msmtj[0])/var_constant + constant_term

pair_miss_exist_prob = (1-detect_rate)*detect_rate/(fpratio+(1-detect_rate)*detect_rate)
def update1(update_matches, msmts1, msmts2, samples, weights):
    match_var = std**2 / 2
    miss_var = std**2
    for sidx, update_match in enumerate(update_matches):
        i,j = update_match
        if i!=-1 and j!=-1:
            samples[sidx] = ((msmts1[i,0]+msmts2[j,0])*.5, msmts1[i,1], msmts2[j,1],
                              match_var, miss_var, miss_var)
            weights[sidx] = 1.
        elif i!=-1:
            samples[sidx] = (msmts1[i,0], msmts1[i,1], -1,
                             miss_var, miss_var, -1)
            weights[sidx] = pair_miss_exist_prob
        elif j!=-1:
            samples[sidx] = (msmts2[j,0], -1, msmts2[j,1],
                             miss_var, -1, miss_var)
            weights[sidx] = pair_miss_exist_prob
        else:
            # all the null updates should be at the end of the array
            return sidx
    return update_matches.shape[0]

third_miss_loglik = np.log(entryrate) + np.log((1-detect_rate)**2*detect_rate + fpratio)
def likelihood2(c, samples, weights, ns, msmts3):
    twopiterm = np.log(2*np.pi)*.5
    msmt_var = std**2 * 2
    nm = len(msmts3)
    for i in xrange(ns):
        sample = samples[i]
        if sample[5] == -1: # only msmt1, so only match on 2nd dimension
            constant_term_i = third_miss_loglik
            constant_term_i += np.log(1./pair_miss_exist_prob/detect_rate-1)
            constant_term_i += np.log(msmt_var)*.5
            constant_term_i += twopiterm
            c[i,:nm] = np.square(sample[1]-msmts3[:,0])/msmt_var + constant_term_i
        elif sample[4] == -1: # only msmt2, so only match on 3rd dimension
            constant_term_i = third_miss_loglik
            constant_term_i += np.log(1./pair_miss_exist_prob/detect_rate-1)
            constant_term_i += np.log(msmt_var)*.5
            constant_term_i += twopiterm
            c[i,:nm] = np.square(sample[2]-msmts3[:,1])/msmt_var + constant_term_i
        else: # both
            constant_term_i = third_miss_loglik
            constant_term_i += np.log(1./detect_rate-1)
            constant_term_i += np.log(msmt_var)
            constant_term_i += twopiterm*2
            c[i,:nm] = np.square(sample[1]-msmts3[:,0])
            c[i,:nm] += np.square(sample[2]-msmts3[:,1])
            c[i,:nm] /= msmt_var
            c[i,:nm] += constant_term_i
            
# probability of msmt from third set, with no matches, being real and not fp
third_exist_prob = (1-detect_rate)**2*detect_rate
third_exist_prob = third_exist_prob / (third_exist_prob + fpratio)
def update2(update_matches2, update_matches, new_samples, new_weights, msmts1, msmts2, msmts3):
    for sidx, update_match2 in enumerate(update_matches2):
        new_sample = new_samples[sidx]
        id12, id3 = update_match2
        if id12 == -1:
            if id3 == -1:
                return sidx
            else:
                new_weights[sidx] = third_exist_prob
                new_sample[0] = .5
                new_sample[1:3] = msmts3[id3, :2]
        else:
            id1, id2 = update_matches[id12]
            if sum((id1==-1, id2==-1, id3==-1)):
                new_weights[sidx] = third_exist_prob
            else:
                new_weights[sidx] = 1.
            if id1 == -1:
                if id3 == -1:
                    new_sample[0] = msmts2[id2, 0]
                    new_sample[1] = .5
                    new_sample[2] = msmts2[id2, 1]
                else:
                    new_sample[0] = msmts2[id2, 0]
                    new_sample[1] = msmts3[id3, 0]
                    new_sample[2] = msmts2[id2,1] + msmts3[id3,1]
            elif id2 == -1:
                if id3 == -1:
                    new_sample[0] = msmts1[id1, 0]
                    new_sample[1] = msmts1[id1, 1]
                    new_sample[2] = .5
                else:
                    new_sample[0] = msmts1[id1,0]
                    new_sample[1] = msmts1[id1,1] + msmts3[id3,0]
                    new_sample[2] = msmts3[id3,1]
            elif id3 == -1:
                new_sample[0] = msmts1[id1,0] + msmts2[id2,0]
                new_sample[1] = msmts1[id1,1]
                new_sample[2] = msmts2[id2,1]
            else:
                new_sample[0] = msmts1[id1,0] + msmts2[id2,0]
                new_sample[1] = msmts1[id1,1] + msmts3[id3,0]
                new_sample[2] = msmts2[id2,1] + msmts3[id3,1]
                

def scoreObj(tru, est):
    c2 = [[sum(np.square(sample[:3]-truobj)) for sample in est] for truobj in tru]
    c2 = np.sqrt(c2)
    m,n = c2.shape
    x = np.zeros(m, dtype=int)
    y = np.zeros(n, dtype=int)
    pred = np.zeros(n, dtype=int)
    d = np.zeros(n,)
    v = np.zeros(n,)
    rows2use = np.arange(m)
    cols2use = np.arange(n)
    scores = []
    for miss_cutoff in miss_distance_cutoffs:
        x[:] = -1
        y[:] = -1
        v[:] = 0
        SSP(c2 - miss_cutoff, x, y, v, rows2use, m, cols2use, n, d, pred)
        nFN = sum(x==-1)
        nFP = sum(y==-1)
        scores.append((nFN,nFP,m,n))
    return np.array(scores)

def scoreTrack(tru_tracks, tru_m, update_matches, update_matches2):
    track_found = np.zeros(tru_tracks.shape[0], dtype=bool)
    fpcount = 0
    pcount = 0
    for id12, id3 in update_matches2:
        if id12 == -1:
            id1 = -1
            id2 = -1
        else:
            id1, id2 = update_matches[id12]
        if all((id1==-1,id2==-1,id3==-1)) == 3:
            continue
        in_tru_tracks = np.all(tru_tracks == (id1,id2,id3), axis=1)
        if any(in_tru_tracks):
            in_tru_tracks = np.where(in_tru_tracks)[0][0]
            track_found[in_tru_tracks] = True
        else:
            fpcount += 1
        pcount += 1
    fncount = tru_m - np.sum(track_found[:tru_m])
    return fncount, fpcount, tru_m, pcount


max_nm = entryrate + int(fprate*6) + 3 # poisson cdf @ 6 = .99992

timed_total_all = 0.
timed_update_all = 0.
obj_scores_all = np.zeros((miss_distance_cutoffs.shape[0],4), dtype=int)
track_scores_all = np.zeros(4, dtype=int)

samples = np.zeros((max_ns, 6))
weights = np.zeros((max_ns,))
hypotheses = np.zeros((max_nhyp, max_ns), dtype=bool)
hypothesis_weights = np.zeros((max_nhyp,))
ids = np.zeros((max_ns,), dtype=np.uint16)
ns = 0
new_samples = samples.copy()
new_weights = weights.copy()
new_hypotheses = hypotheses.copy()
new_hypothesis_weights = hypothesis_weights.copy()
new_ids = ids.copy()
new_ns = 0
c1 = np.zeros((max_ns, max_nm))
c2 = c1.copy()
update_matches = np.zeros((max_ns, 2), dtype=int)
update_matches2 = np.zeros((max_ns, 2), dtype=int)
workvars = allocateWorkVarsforDA(max_ns, max_nm, max_nhyp)
sols_rows2use, sols_cols2use, sols_elim, sols_x, sols_v, backidx1 = workvars
backidx2 = backidx1.copy()
row_sets = np.zeros((1,max_ns), dtype=bool)
col_sets = np.zeros((1,max_nm), dtype=bool)
includerowsorcols_dummy = np.zeros(1)


for test in xrange(ntests):
    # generate real objects
    tru_m = entryrate#np.random.poisson(entryrate)
    tru = np.random.rand(tru_m, 3)
    tru_tracks = np.zeros((tru_m, 3), dtype=int) - 1
    # generate three sets of measurements
    detected = np.random.rand(tru_m) < detect_rate
    nreal = sum(detected)
    nfalse = np.random.poisson(fprate)        
    msmts1 = tru[detected][:,[0,1]]+np.random.normal(size=(nreal,2))*std
    msmts1 = np.append(msmts1, np.random.rand(nfalse, 2), axis=0)
    tru_tracks[:tru_m][detected,0] = np.arange(nreal)
    tru_tracks_false = np.zeros((nfalse, 3), dtype=int)-1
    tru_tracks_false[:,0] = np.arange(nreal, nreal+nfalse)
    tru_tracks = np.append(tru_tracks, tru_tracks_false, axis=0)
    nm1 = nreal+nfalse
    
    detected = np.random.rand(tru_m) < detect_rate
    nreal = sum(detected)
    nfalse = np.random.poisson(fprate)
    msmts2 = tru[detected][:,[0,2]]+np.random.normal(size=(nreal,2))*std
    msmts2 = np.append(msmts2, np.random.rand(nfalse, 2), axis=0)
    tru_tracks[:tru_m][detected,1] = np.arange(nreal)
    tru_tracks_false = np.zeros((nfalse, 3), dtype=int)-1
    tru_tracks_false[:,1] = np.arange(nreal, nreal+nfalse)
    tru_tracks = np.append(tru_tracks, tru_tracks_false, axis=0)
    nm2 = nreal+nfalse
    
    detected = np.random.rand(tru_m) < detect_rate
    nreal = sum(detected)
    nfalse = np.random.poisson(fprate)
    msmts3 = tru[detected][:,[1,2]]+np.random.normal(size=(nreal,2))*std
    msmts3 = np.append(msmts3, np.random.rand(nfalse, 2), axis=0)
    tru_tracks[:tru_m][detected,2] = np.arange(nreal)
    tru_tracks_false = np.zeros((nfalse, 3), dtype=int)-1
    tru_tracks_false[:,2] = np.arange(nreal, nreal+nfalse)
    tru_tracks = np.append(tru_tracks, tru_tracks_false, axis=0)
    nm3 = nreal+nfalse
    
    # first update
    timed_total = time()
    likelihood1(c1, msmts1, msmts2)
    cs = c1#cs = sparsify(c1, s)
    
    row_sets[0,:nm1] = True
    row_sets[0,nm1:] = False
    col_sets[0,:nm2] = True
    col_sets[0,nm2:] = False
    timed_start = time()
    da(cs, row_sets, includerowsorcols_dummy, col_sets, includerowsorcols_dummy,
           update_matches, hypotheses, hypothesis_weights,
           sols_rows2use, sols_cols2use, sols_elim, sols_x, sols_v, backidx1)
    timed_update = time() - timed_start
    ns = update1(update_matches, msmts1, msmts2, samples, weights)
    
    # find likelihood between updated objects and third set of measurements
    likelihood2(c2, samples, weights, ns, msmts3)
    cs = c2#cs = sparsify(c2, s)
    # account for the fact that each row miss is normalized
    missliks = np.log(1-weights*detect_rate)
    missliks_hyp = np.dot(hypotheses, missliks)
    hypothesis_weights -= missliks_hyp 
    col_sets[0,:nm3] = True
    col_sets[0,nm3:] = False
    
    # second update
    timed_start = time()
    da(cs, hypotheses, hypothesis_weights, col_sets, includerowsorcols_dummy,
           update_matches2, new_hypotheses, new_hypothesis_weights,
           sols_rows2use, sols_cols2use, sols_elim, sols_x, sols_v, backidx2)
    timed_update += time() - timed_start
    new_ns = update2(update_matches2, update_matches, new_samples, new_weights,
                     msmts1, msmts2, msmts3)
    timed_total = time() - timed_total 

    ## analysis of how hypotheses match truth, for debugging purposes
    tru_matches_1_valid = (tru_tracks[:,0] >= 0) | (tru_tracks[:,1] >= 0)
    tru_matches_1 = backidx1[tru_tracks[tru_matches_1_valid,0],
                             tru_tracks[tru_matches_1_valid,1]]
    tru_matches_not_here = sum(tru_matches_1 == -1)
    if tru_matches_not_here == 0:
        tru_hypothesis = np.zeros(hypotheses.shape[1], dtype=bool)
        tru_hypothesis[tru_matches_1] = True
        matching_hypotheses = np.where(np.all(hypotheses==tru_hypothesis,axis=1))[0]
        assert len(matching_hypotheses) <= 1
        if len(matching_hypotheses) == 1:
            matching_hypothesis = matching_hypotheses[0]
            tru_matches_2_score = tru_matches_1_valid & (tru_tracks[:,2] >= 0)
            tru_matches_2in = tru_matches_1[tru_tracks[tru_matches_1_valid,2] >= 0]
            total_prob = -sum(missliks[tru_matches_1])
            total_prob += sum(c2[tru_matches_2in,
                                tru_tracks[tru_matches_2_score,2]])
            tru_matches_1_score = (tru_tracks[:,0] >= 0) & (tru_tracks[:,1] >= 0)
            total_prob += sum(c1[tru_tracks[tru_matches_1_valid,0],
                                 tru_tracks[tru_matches_1_valid,1]])
            if total_prob + 1e-4 < new_hypothesis_weights[0]:
                print("probable error")
            else:
                tru_assignment_rank = np.searchsorted(new_hypothesis_weights, total_prob)
    
    # score
    timed_update_all += timed_update
    timed_total_all += timed_total
    include_samples = new_hypotheses[0] & (new_weights > .5)

    track_scores_all += scoreTrack(tru_tracks, tru_m, update_matches,
                                   update_matches2[new_hypotheses[0]])
    obj_scores_all += scoreObj(tru, new_samples[include_samples])
    
timed_update_all *= 1000./ntests
timed_total_all *= 1000./ntests
obj_score_rates = obj_scores_all[:,:2].astype(float)/obj_scores_all[:,2:]
track_score_rates = track_scores_all[:2].astype(float)/track_scores_all[2:]
#score_rates = track_score_rates
score_rates = np.append(track_score_rates[None,:], obj_score_rates, axis=0)
print("{:.1f} update, {:.1f} total".format(timed_update_all, timed_total_all))
print(score_rates)