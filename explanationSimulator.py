# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 08:31:30 2022

@author: chell

Explanation simulator
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as ptch

'''USER INPUT'''
# Should eventually adapt to take JSON for path/domain data.

'''SCENARIO 1'''
# #Domain (map) dimensions
# domain_x = 10
# domain_y = 11

# # Set up path S*. Each entry is a vertex (y,x).
# pathS = [[2,0],[2,4],[5,4],[5,7],[3,7],[3,10]]

# # Place obstacles with their bounds. 
# omegas = [[[1,1,1],[-1,9,1],[-1,10,-1],[1,5,-1]], #O0
#           [[1,5,1],[-1,6,1],[-1,6,-1],[1,5,-1]], #O1
#           [[1,8,1],[-1,7,1],[-1,8,-1],[1,9,-1]], #O2
#           [[1,9,1],[-1,0,1],[-1,1,-1],[1,10,-1]],#O3
#           [[1,5,1],[-1,2,1],[-1,4,-1],[1,6,-1]], #O4
#           [[1,3,1],[-1,6,1],[-1,7,-1],[1,4,-1]], #O5
#           [[1,2,1],[-1,3,1],[-1,4,-1],[1,3,-1]],  #O6
#           [[1,7,1],[1,8,-1],[-1,0,1],[-1,1,-1]]   #O7
#           ]

'''SCENARIO 2'''

# Domain (map) dimensions
domain_x = 10
domain_y = 11

# Set up path S*. Each entry is a vertex (y,x).
pathS = [[2,0],[2,4],[5,4],[5,7],[3,7],[3,10]]

# Place obstacles with their bounds. 
omegas = [[[1,1,1],[-1,9,1],[-1,10,-1],[1,5,-1]], #O0
          [[1,5,1],[-1,6,1],[-1,6,-1],[1,5,-1]], #O1
          [[1,8,1],[-1,7,1],[-1,8,-1],[1,9,-1]], #O2
          [[1,9,1],[-1,0,1],[-1,1,-1],[1,10,-1]],#O3
          [[1,5,1],[-1,2,1],[-1,4,-1],[1,6,-1]], #O4
          [[1,3,1],[-1,6,1],[-1,7,-1],[1,4,-1]], #O5
          [[1,2,1],[-1,3,1],[-1,4,-1],[1,3,-1]],  #O6
          [[1,8,1],[1,9,-1],[-1,1,1],[-1,2,-1]]   #O7
          ]


# Place reference points and give 1-norm 'radius'
refPoints = [[7,2],[7,8],[1,9]]
refRad = 2

# Set nearness constraints to on or off
useRefPoints = True

# Choose print statements to show
printPathAndHulls = False
printExplDetails = False

'''END USER INPUT'''

###############################################################################
# Global variables
###############################################################################

# Path segments given as two lists [y1,x1],[y2,x2] (endpoints). 
# First coord for y, second coord for x. 
# Origin at bottom left of map.

pathStuples= []
pathSvertSegs = []
pathShorizSegs = []

nearnessHulls = []
obstsInNearHulls = []

hullList = []
hullComplexities = []

refPtsForHulls = []

###############################################################################
# Function declarations
###############################################################################

def addSegment(pt1,pt2): 
    # Creates segment from path vertices and adds to path 
    # points given as python lists [y,x]
    
    # Will use the constraint-formulation segs for _convex hull initiation_
    # Will use the actual segs for _convex hull building_
    if pt1[0] == pt2[0]:
        # segment is horizontal!
        s_tuplePlus = [-1, pt1[0], 1]
        s_tupleMinus = [-1, pt1[0], -1]
        # segment is structured as tensor of z-coord, endpt1, endpt2 in other dim 
        minpt = min(pt1[1], pt2[1])
        maxpt = max(pt1[1], pt2[1])
        #horizSeg = [pt1[0], pt1[1], pt2[1]]
        horizSeg = [pt1[0], minpt, maxpt]
        pathShorizSegs.append(horizSeg)
    else:
        # segment is vertical! (if properly defined)
        s_tuplePlus = [1, pt1[1], 1]
        s_tupleMinus = [1, pt1[1], -1]
        
        minpt = min(pt1[0], pt2[0])
        maxpt = max(pt1[0], pt2[0])
        vertSeg = [pt1[1], minpt, maxpt]
        pathSvertSegs.append(vertSeg)
    
    # pathStuples will be twice as long as actual number of segs in path
    pathStuples.append(s_tuplePlus)
    pathStuples.append(s_tupleMinus)
    
    pass

def buildHulls(sseg, segEndpt1, segEndpt2):
    # Hulls should be built as sets of 4 constraints (fij).
    # For each hull, keep a tab of how many non-boundary (M) sides. 
    # (This will be the same as the complexity!)
    
    # An fij corresponds to a boundary iff the z value is 0 or the 
    #  domain_x/domain_y of the domain dimension!
    f1 = sseg
    f1complexity = 1
    # We use endpoints of the actual _segment_. 
    
    # Horizontal and vert segs must be sorted by ascending z value
    if sseg[0]==-1: # horizontal seg
        minEndpt = min(segEndpt1[1], segEndpt2[1])
        maxEndpt = max(segEndpt1[1], segEndpt2[1])
        
        # Handle the other horizontal side.
        if sseg[2] == 1: # positive ineq., so f1 is the bottom line
            b4 = f1 # b4 is bottom line, so f1 will be b4.
            f4 = [-1, domain_y, -1] # Initialize as top boundary
            f4complexity = 0
            for segToCheck in pathShorizSegs:
                # See if there is overlap between the path seg and the horizontal seg to check
                if not (segToCheck[1] >= maxEndpt or segToCheck[2]<=minEndpt):
                    if segToCheck[0] > sseg[1]:
                        f4 = [-1,segToCheck[0],-1]
                        f4complexity = 1
                        break     
            b2 = f4 # b2 will be top line.
        else: # f1 is the top line
            b2 = f1 # b2 will be top line
            f4 = [-1, 0, 1] # Initialize as bottom boundary
            f4complexity = 0
            for segToCheck in pathShorizSegs:
                # See if there is overlap between the path seg and the horizontal seg to check
                if not (segToCheck[1] >= maxEndpt or segToCheck[2]<=minEndpt):
                    if segToCheck[0] < sseg[1]:
                        f4 = [-1,segToCheck[0],1]
                        f4complexity = 1
            b4 = f4 # b4 will be bottom line.
        
        topVal = max(f1[1],f4[1])
        bottomVal = min(f1[1],f4[1])
        f2 = [1,0,1] # initialize as left-side vertical boundary
        f2complexity = 0
        for segToCheck in pathSvertSegs:
            if not (segToCheck[1]>=topVal) and not (segToCheck[2]<=bottomVal):
                if segToCheck[0]<= minEndpt:
                    f2 = [1,segToCheck[0],1]
                    f2complexity = 1
        b3 = f2 # b3 will be left side.
                
        f3 = [1, domain_x, -1] #initialize as right-side vertical boundary
        f3complexity = 0
        for segToCheck in pathSvertSegs:
            if not (segToCheck[1]>=topVal) and not (segToCheck[2]<=bottomVal):
                if segToCheck[0]>= maxEndpt:
                    f3 = [1,segToCheck[0],-1]
                    f3complexity = 1  
                    break
        b1 = f3 # b1 will be right side.
                        
    
    else: # vertical segment  
        minEndpt = min(segEndpt1[0], segEndpt2[0])
        maxEndpt = max(segEndpt1[0], segEndpt2[0])

        # Handle the other vertical side.
        if sseg[2] == 1: # positive ineq., so f1 is the left line
            b3 = f1 # b3 will be left side.
            f4 = [1, domain_x, -1] # Initialize as right boundary
            f4complexity = 0
            for segToCheck in pathSvertSegs:
                # See if there is overlap between the path seg and the horizontal seg to check
                if not (segToCheck[1] >= maxEndpt or segToCheck[2] <= minEndpt):
                    if segToCheck[0] > sseg[1]:
                        f4 = [1,segToCheck[0],-1]
                        f4complexity = 1
                        break
            b1 = f4 # b1 will be right line.
        else: # f1 is the right line
            b1 = f1 
            f4 = [1, 0, 1] # Initialize as left boundary
            f4complexity = 0
            for segToCheck in pathSvertSegs:
                # See if there is overlap between the path seg and the horizontal seg to check
                if not (segToCheck[1] >= maxEndpt or segToCheck[2]<=minEndpt):
                    if segToCheck[0] < sseg[1]:
                        f4 = [1,segToCheck[0],1]
                        f4complexity = 1  
            b3 = f4
        
        leftVal = min(f1[1], f4[1])
        rightVal = max(f1[1], f4[1])
        f2 = [-1,0,1] # initialize as bottom horizontal boundary
        f2complexity = 0
        for segToCheck in pathShorizSegs:
            if not (segToCheck[1] >= rightVal) and not (segToCheck[2] <= leftVal):
                if segToCheck[0]<= minEndpt:
                    f2 = [-1,segToCheck[0],1]
                    f2complexity = 1
        b4 = f2 # b4 will be bottom line.        
        f3 = [-1, domain_y, -1] #initialize as top horizontal boundary
        f3complexity = 0
        for segToCheck in pathShorizSegs:
            if not (segToCheck[1] >= rightVal) and not (segToCheck[2] <= leftVal):
                if segToCheck[0]>= maxEndpt:
                    f3 = [-1,segToCheck[0],-1]
                    f3complexity = 1
                    break
        b2 = f3 # b2 will be top line.
           
    
    # Only add hull if it doesn't already exist.
    if not ([b1, b2, b3, b4] in hullList):
    # In order: right, top, left, bottom.
        hullList.append([b1, b2, b3, b4])
        hullComplexities.append(f1complexity + f2complexity + f3complexity + f4complexity)
    pass

def checkNearnessHulls(rPts, rad, oList):

    for rPtno, rPt in enumerate(rPts):
        rPtObsts = []
        for oidx, oVerts in enumerate(oList):
            oVertsGood = True
            for vert in oVerts:
                if not ((vert[0]==-1 and abs(vert[1]-rPt[0])<rad) or (vert[0]==1 and abs(vert[1]-rPt[1])<rad)):
                    oVertsGood = False
            if oVertsGood:
                rPtObsts.append(oidx)
        if len(rPtObsts)>0:
            nearnessHulls.append(rPt)
            hullList.append(rPt)
            obstsInNearHulls.append(rPtObsts)
            hullComplexities.append(1)
            refPtsForHulls.append(rPtno)
            
    pass

def isSubsetEq(oSet, bSet):
    # CURRENTLY UNUSED
    # Return 1 if oSet is a subseteq of bSet 
    # Return 0 otherwise
    o1 = oSet[0]
    d11, z11, p11 = o1[0], o1[1], o1[2]
    
    o2 = oSet[1]
    d21, z21, p21 = o2[0], o2[1], o2[2]
    
    o3 = oSet[2]
    d31, z31, p31 = o3[0], o3[1], o3[2]
    
    o4 = oSet[3]
    d41, z41, p41 = o4[0], o4[1], o4[2]
    
    isSubsetEq = 1
    
    for bEqn in bSet:
        d2 = bEqn[0]
        z2 = bEqn[1]
        p2 = bEqn[2]
        
        mu_oset_beqn = (d2*(d2+d11))*(p2+p11)*(z11-z2)+(d2*(d2+d21))*(p2+p21)*(z21-z2)+(d2*(d2+d31))*(p2+p31)*(z31-z2) + (d2*(d2+d41))*(p2+p41)*(z41-z2)
        
        if mu_oset_beqn < 0:
            isSubsetEq = 0
    return isSubsetEq 

def isSafelyContained(oSet, bSet):
    # Return 1 if oSet a subset of bSet s.t. none of oSet's sides coincide with
    #   a side of the path S.
    # Return 0 otherwise
    o1 = oSet[0]
    d11, z11, p11 = o1[0], o1[1], o1[2]
    
    o2 = oSet[1]
    d21, z21, p21 = o2[0], o2[1], o2[2]
    
    o3 = oSet[2]
    d31, z31, p31 = o3[0], o3[1], o3[2]
    
    o4 = oSet[3]
    d41, z41, p41 = o4[0], o4[1], o4[2]
    
    isSubsetEq = 1
    
    for bEqn in bSet:
        d2 = bEqn[0]
        z2 = bEqn[1]
        p2 = bEqn[2]
        
        mu_oset_beqn = (d2*(d2+d11))*(p2+p11)*(z11-z2)+(d2*(d2+d21))*(p2+p21)*(z21-z2)+(d2*(d2+d31))*(p2+p31)*(z31-z2) + (d2*(d2+d41))*(p2+p41)*(z41-z2)
        
        if mu_oset_beqn < 0:
            isSubsetEq = 0

    isSafe = 1    
    if isSubsetEq == 1: # only bother checking for safety if we have a subseteq
    
        for o_side in [o1, o2, o3, o4]:
            # if o_side doesn't correspond to a domain boundary:
            if not o_side[1] == 0 and not ((o_side[0] == 1 and o_side[1] == domain_x) or (o_side[0] == -1 and o_side[1] == domain_y)):
                # if o_side is the right side and coincides with the right hull side:
                if o_side[0]==1 and o_side[2]==-1 and o_side[1]==bSet[0][1]:
                    isSafe = 0 
                # if o_side is the top and coincides with hull top:
                if o_side[0]==-1 and o_side[2]==-1 and o_side[1]==bSet[1][1]:
                    isSafe = 0
                # left side:
                if o_side[0]==1 and o_side[2]==1 and o_side[1]==bSet[2][1]:
                    isSafe = 0
                # bottom side:
                if o_side[0]==-1 and o_side[2]==1 and o_side[1]==bSet[3][1]:
                    isSafe = 0
        
    return isSubsetEq*isSafe
        
def tightenHull(omegaGroup, hull):
    # Based on sides of hull which coincide with domain boundaries, 
    #   -identify which constraint orientations/signs to keep at all
    #   -find each (relevant) constraint f' which satisfies f \subset f' for
    #    all f' in the omegaGroup
    #   -include only these in the final hull given for explanation.
    
    # Compare the constraints of the omegas and find the furthest extremes in
    #   all directions. 
    leftSide = domain_x
    rightSide = 0
    topSide = 0
    bottomSide = domain_y
    
    for omega in omegaGroup:
        for side in omega:
            if side[0]==-1 and side[2]==1: #left side
                leftSide = min(leftSide, side[1])
            elif side[0]==-1 and side[2]==-1: #right side
                rightSide = max(rightSide, side[1])
            elif side[0]==1 and side[2]==1: #bottom side
                bottomSide = min(bottomSide, side[1])
            else: #top side
                topSide = max(topSide, side[1])
    
    tighterHull = []
    
    for hs in hull:
        # Check if hull side corresponds to a boundary. If not...
        if not (hs[1]==0 or (hs[0]==-1 and hs[1]==domain_y) or (hs[0]==1 and hs[1]==domain_x)):
            if hs[0]==-1 and hs[2]==1: #left side
                tighterHull.append([-1,leftSide,1])
            elif hs[0]==-1 and hs[2]==-1: #right side
                tighterHull.append([-1,rightSide,-1])
            elif hs[0]==1 and hs[2]==1: #bottom side
                tighterHull.append([1,bottomSide,1])
            else: #top side
                tighterHull.append([1,topSide,-1])
    return tighterHull
        

###############################################################################
# End function declarations
###############################################################################

# Convert path into segments.
for v in range(0,len(pathS)-1):
    addSegment(pathS[v],pathS[v+1])
    
pathSvertSegs = sorted(pathSvertSegs, key = lambda zvalv: zvalv[0])
pathShorizSegs = sorted(pathShorizSegs, key = lambda zvalh: zvalh[0])

for segidx, sseg in enumerate(pathStuples):
    halvedIdx = int((segidx-segidx % 2)/2)
    ptOnSeg1 = pathS[halvedIdx] # This gets the first vertex of any given seg.
    ptOnSeg2 = pathS[halvedIdx+1]
    buildHulls(sseg, ptOnSeg1, ptOnSeg2) # This will build hulls for both sides of each path seg.

checkNearnessHulls(refPoints, refRad, omegas)
print('refPtsForHulls: ' + str(refPtsForHulls))
numFHulls = len(hullList)-len(nearnessHulls)
print('numFHulls: ' + str(numFHulls))
if printPathAndHulls:
    print('pathStuples: ' + str(pathStuples))
    print('pathSvertSegs (sorted): ' + str(pathSvertSegs))
    print('pathShorizSegs (sorted): ' + str(pathShorizSegs))

    print('hullList: ' + str(hullList))
    print('hullComplexities: ' + str(hullComplexities))
    print ('nearnessHulls: ' + str(nearnessHulls))
    print('obstsInNearHulls: ' + str(obstsInNearHulls))

 
###############################################################################
# BEGIN SOLVER ALGORITHMS
###############################################################################

###############################################################################
# Pre-Processing (B-Reduction)
###############################################################################

# First need to determine which Omega are contained entirely within which hulls (?) 
# If any hulls contain no Omega, then the hulls are useless and needn't be considered.
omegaHullsTensors = [] # which hulls each omega is in, as binary tensors
omegaHullsLists = []   # which hulls each omega is in, as a list per omega
hullsPerOmega = [0]*len(omegas)
omegasPerHull = torch.zeros(len(hullList))
explRefPtNos = []

for o_idx, omega in enumerate(omegas):
    omegaHullsTensors.append(torch.zeros(len(hullList)))
    omegaHullsListThisOmega = []
    for h_idx, hull in enumerate(hullList):
        if h_idx < numFHulls:
            o_in_hull = isSafelyContained(omega, hull)
            omegaHullsTensors[o_idx][h_idx] = o_in_hull
            if o_in_hull == 1:
                omegaHullsListThisOmega.append(h_idx)
        else: # we're considering a nearness constraint
            nearConstIdx = h_idx - numFHulls
            if o_idx in obstsInNearHulls[nearConstIdx]:
                omegaHullsTensors[o_idx][h_idx] = 1
                omegaHullsListThisOmega.append(h_idx)
                o_in_hull = 1
            else:
                omegaHullsTensors[o_idx][h_idx] = 0
                o_in_hull = 0
            
        hullsPerOmega[o_idx] += o_in_hull
    omegasPerHull = omegasPerHull + omegaHullsTensors[o_idx]
    omegaHullsLists.append(omegaHullsListThisOmega)
    
    
       
#print('omegaHullsTensors: ' + str(omegaHullsTensors))
print('omegaHullsLists: ' + str(omegaHullsLists))
print('omegasPerHull: ' + str(omegasPerHull))
print('hullsPerOmega: ' + str(hullsPerOmega))

# Build lists of omegas in each hull.
hullsWithListOfOmegas = []
for h_idx, hull in enumerate(hullList):
    hullsWithListOfOmegas_thisHull = []
    for o_idx, omega in enumerate(omegas):
        if h_idx < numFHulls:
            o_in_hull = isSafelyContained(omega, hull)
        else:
            nearConstIdx = h_idx - numFHulls
            if o_idx in obstsInNearHulls[nearConstIdx]:
                o_in_hull = 1
            else:
                o_in_hull = 0
        if o_in_hull == 1:
            hullsWithListOfOmegas_thisHull.append(o_idx)

    hullsWithListOfOmegas.append(hullsWithListOfOmegas_thisHull)

print('hullsWithListOfOmegas: ' + str(hullsWithListOfOmegas))
explHulls = [] # each entry is a hull which will be used in the explanation
explObstGroups = [] # each entry is list of obstacles contained in the corresponding hull
num_FH_in_expl = 0

#print('Entering loop...')
iterations = 0
while (1 in hullsPerOmega and iterations < 10):
    o_group = []
    # Find an omega which only appears in one hull:
    o_idx = hullsPerOmega.index(1)

    # Find out which hull it appears in:
    h_idx = omegaHullsLists[o_idx][0]

    # Put the omega and its buddies into a group:
    o_group = hullsWithListOfOmegas[h_idx].copy()
    
    # Add the hull to the explanation:
    explHulls.append(hullList[h_idx])
    if h_idx < numFHulls:
        num_FH_in_expl += 1
    else:
        explRefPtNos.append(refPtsForHulls[h_idx-numFHulls])
        print('explRefPtNos: ' + str(explRefPtNos))
        
    # Add the corresponding group of obstacles to the explanation: 
    explObstGroups.append(o_group)
    
    # Now clean up:
    for o_toRemove in o_group:
        #print('o_toRemove: ' + str(o_toRemove))
        #print('o_group: ' + str(o_group))
        # Clean up hullsWithListOfOmegas and omegasPerHull
        for h_no, hull in enumerate(hullsWithListOfOmegas):
            if o_toRemove in hull:
                hull.remove(o_toRemove)   
                omegasPerHull[h_no] -= 1
        # Clean up hullsPerOmega:
        hullsPerOmega[o_toRemove] = 0
        # Clean up omegaHullsLists (remove all hulls for removed omega):
        omegaHullsLists[o_toRemove] = []
        
        
       
    # Clean up omegaHullsLists (remove hull that has been added to explanation):
    for omega_entry in omegaHullsLists:
        if h_idx in omega_entry:
            omega_entry.remove(h_idx)
    iterations += 1
    #print('hullsPerOmega: ' + str(hullsPerOmega))
# print('hullList: ' +str(hullList))
# print('leftover hullsWithListOfOmegas: ' + str(hullsWithListOfOmegas))
# print('leftover hullsPerOmega: ' + str(hullsPerOmega))
# print('leftover omegaHullsLists: ' + str(omegaHullsLists) )
# print('hullComplexities: ' + str(hullComplexities))
# print('omegasPerHull: ' + str(omegasPerHull)) 

###############################################################################
# Greedy
###############################################################################

iterations = 0
# Continue until all omega are cleared
print('max(omegasPerHull): ' + str(max(omegasPerHull)))
while (max(omegasPerHull).item()>0 and iterations < 10):
    o_group = []

    # Find hulls with max number of omegas:
    max_num_omegas_in_hull = max(omegasPerHull).item()
    best_hull = 0
    bestCSoFar = 5 # impossible value, so will always get overwritten
    
    for oHno, oH in enumerate(omegasPerHull):
        
        if oH.item() == max_num_omegas_in_hull and hullComplexities[oHno]<bestCSoFar:
            best_hull = oHno
            bestCSoFar = hullComplexities[best_hull]
    # Put all omegas from the best_hull into a group:
    o_group = hullsWithListOfOmegas[best_hull].copy()
    #print('o_group: ' + str(o_group))
    
    # Add the hull to the explanation:
    explHulls.append(hullList[best_hull])
    print('best_hull is: ' + str(best_hull))
    if best_hull < numFHulls:
        #print('best_hull was: ' + str(best_hull))
        num_FH_in_expl += 1
    else:
        explRefPtNos.append(refPtsForHulls[best_hull-numFHulls])
        print('explRefPtNos: ' + str(explRefPtNos))
    # Add the corresponding group of obstacles to the explanation: 
    explObstGroups.append(o_group)
    
    # Now clean up:
    for o_toRemove in o_group:
        # Clean up hullsWithListOfOmegas and omegasPerHull
        for h_no, hull in enumerate(hullsWithListOfOmegas):
            if o_toRemove in hull:
                hull.remove(o_toRemove)   
                omegasPerHull[h_no] -= 1  
        # Clean up hullsPerOmega:
        hullsPerOmega[o_toRemove] = 0
        # Clean up omegaHullsLists (remove all hulls for removed omega):
        omegaHullsLists[o_toRemove] = []
    #print('hullsPerOmega: ' + str(hullsPerOmega))    
    # Clean up omegaHullsLists (remove hull that has been added to explanation):
    for omega_entry in omegaHullsLists:
        if h_idx in omega_entry:
            omega_entry.remove(h_idx)
    iterations += 1




###############################################################################
# Tighten hulls
###############################################################################
explHulls_FH = explHulls[0:num_FH_in_expl]
explHulls_Near = explHulls[num_FH_in_expl::]
print('explHulls_FH: ' + str(explHulls_FH))

# Tighten each of the hulls in the explanation
tightenedHulls = []
for h_idx, hull in enumerate(explHulls_FH):
    omega_indices = explObstGroups[h_idx]
    omega_group = []
    for index in omega_indices:
        omega_group.append(omegas[index])
    tightenedHulls.append(tightenHull(omega_group, hull))
    
    
if printExplDetails:
    print('explHulls: ' + str(explHulls))
    print('explObstGroups: ' + str(explObstGroups))
    print('tightenedHulls: ' + str(tightenedHulls))

###############################################################################
# Translate constraints into natural language
###############################################################################
 # will later add in the circular constraints 
print('###############################')
print('Text explanation: ')
print('###############################')

# Linear inequality constraint hulls
for g_idx, group in enumerate(tightenedHulls):
    
    numObsts = len(explObstGroups[g_idx])
    if numObsts > 1:
        explStr = str(numObsts) + ' obstacles are'
    else: # not plural
        explStr = str(numObsts) + ' obstacle is'
    
    explItems = []
    for constraint in group:
        if constraint[0]==-1 and constraint[2]==1:
            explItems.append(' north of H' + str(constraint[1]))
        elif constraint[0]==-1 and constraint[2]==-1:
            explItems.append(' south of H' + str(constraint[1]))
        elif constraint[0]==1 and constraint[2]==1:            
            explItems.append(' east of V' + str(constraint[1]))
        else:
            explItems.append(' west of V' + str(constraint[1]))
    
    #print(explItems)
    num_items = len(explItems)
    if num_items > 1:
        for item_no in range(0, num_items):
            if item_no == num_items-1 and num_items > 1:
                explStr += ' and'
            explStr += explItems[item_no]
            if item_no < num_items-1 and num_items > 2:
                explStr += ','

    else:
        explStr += explItems[0]

    print(explStr)
    
# Nearness hulls
for h_idx, nHull in enumerate(explHulls_Near):
    
    numObsts = len(explObstGroups[h_idx+num_FH_in_expl])
    if numObsts > 1:
        explStr = str(numObsts) + ' obstacles are near Reference Point '
    else: # not plural
        explStr = str(numObsts) + ' obstacle is near Reference Point '
    
    explStr += str(explRefPtNos[h_idx])
    
    

    print(explStr)
    
#print('rects: ' + str(rects))
fig = plt.figure(1)
ax = fig.add_subplot(111)
plt.grid()
plt.xlim([0,domain_x])
plt.ylim([0,domain_y])
plt.xticks(range(0,domain_x))
plt.yticks(range(0,domain_y))
yValues = [item[0] for item in pathS]
xValues = [item[1] for item in pathS]

nO = len(omegas)
rects = []
#print('rects: ' + str(rects))
for oidx, obst in enumerate(omegas):
    points = [0,0]*4
    hsides = []
    vsides = []
    for side in obst:
        if side[0]==-1: # horiz
            hsides.append(side[1])
        else:
            vsides.append(side[1])
            
    if min(vsides)-max(vsides)==0 or min(hsides)-max(hsides)==0:
        # Obstacle is just at a point
        ax.plot(vsides[0],hsides[0],marker='s', markersize = 5, color='#ff5733')
    else:
        # Obstacle is a line or rectangle
        rects.append([min(vsides), min(hsides), max(0.1,max(vsides)-min(vsides)),max(0.1,max(hsides)-min(hsides))])


for omRec in rects:
    #print('omRec: ' + str(omRec))
    ax.add_patch(ptch.Rectangle((omRec[0],omRec[1]), omRec[2], omRec[3],lw = .2,color='#ff5733'))

    
for refpt in refPoints:
    ax.plot(refpt[1],refpt[0], marker = '*', markersize=12, color='#c2c21d' )

ax.plot(xValues, yValues, marker='o', color='#0052a9')
plt.xlabel('V-street')
plt.ylabel('H-street')
plt.title('Path and Obstacles in Domain')
plt.show()