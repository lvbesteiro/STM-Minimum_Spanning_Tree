# This script analyzes the degree of order in two-dimensional (2D) polymers from STM experimental images.
# Written by Lucas V. Besteiro, lvbesteiro@protonmail.com.
# Shared under GPL 3.0 license

import os
import numpy as np
from skimage import io, filters, color, morphology, img_as_float
from skimage import exposure, draw
from scipy import ndimage as ndi
import networkx as nx

# Reads image files and scale information. Expected name format: "TEXTscale_XXX.extension"
class img_file:
     def __init__(self,name):
         self.name = name
         self.nm_size = float(name.split('scale_')[1].split('.')[0])
     def load(self):
         img = img_as_float(io.imread(self.name))
         self.px_sizex = img.shape[1]  
         self.px_sizey = img.shape[0]  
         self.area = self.nm_size**2 * self.px_sizey/self.px_sizex
         self.scale = self.px_sizex/self.nm_size
         return img

# Converts image to grayscale, equalize and enhance contrast
def gray_process(img,dsize=25,cutoff=0.5,gain=10):
    img_gr = np.zeros_like(img[:,:,0])
    img_gr = (img[:,:,0]+img[:,:,1]+img[:,:,2])/3
    selem = morphology.disk(dsize)
    img_gr = filters.rank.equalize(img_gr, selem=selem)
    img_gr = img_as_float(img_gr)
    img_gr = exposure.adjust_sigmoid(img_gr,cutoff=cutoff,gain=gain)
    return img_gr
    
# Image segmentation. Finds the different cells
def labels(img_gr,low_th=.1,high_th=.8):
    edges = filters.sobel(img_gr)
    markers = np.zeros_like(img_gr)
    markers [img_gr < low_th] = 2
    markers [img_gr > high_th] = 1
    segmentation = morphology.watershed(edges, markers)
    segmentation = ndi.binary_fill_holes(segmentation-1)
    label_mask, Nlabel = ndi.label(segmentation)
    img_label_overlay = color.label2rgb(label_mask, image=img_gr)
    return img_label_overlay, label_mask, Nlabel

# Populates the network with nodes
def find_nodes(mask,N,px_a_th,scale):
    G = nx.Graph()
    for ii in range (1,N+1):
        # Gets the points for each label
        region = np.column_stack(np.where(mask==ii))
        # Gets the region center of mass, in pixel coordinates
        pcoord = np.floor(sum(region)/len(region))
        area = len(region)
        # Selects cells above a threshold area to avoid image noise
        cond1 = area >= px_a_th
        width = mask.shape[1] * .05
        edge = pcoord[0] < width or pcoord[0] > mask.shape[1] - width 
        edge = edge or pcoord[1] < width or pcoord[1] > mask.shape[1] - width 
        # Reduced area threshold if cell is at the edge of image 
        cond2 = area >= px_a_th/3 and (edge)
        if cond1 or cond2:
            G.add_node(ii, pixel_pos=pcoord, area=area/scale**2)
    return G

# Creates voronoi diagram and connects network. Outputs additional images
def voronoi(img,G):
    psx = img.px_sizex
    psy = img.px_sizey
    scale = img.scale   # px/nm
    img_vor = np.zeros((psy,psx,3),dtype=np.uint8) # Random colors
    img_vor_deg = np.ones((psy,psx,3),dtype=np.uint8) # Color by cell degree
    mask = np.zeros((psy,psx),dtype=np.int16)
    S = np.zeros(G.number_of_nodes())
    nodelist = np.array([n for n, nbrs in G.nodes(data=True)])
    nx = [G.nodes[n]['pixel_pos'][0] for n, nbrs in G.nodes(data=True)]
    ny = [G.nodes[n]['pixel_pos'][1] for n, nbrs in G.nodes(data=True)]
    narea = [G.nodes[n]['area']*scale**2 for n, nbrs in G.nodes(data=True)]
    nr = np.random.randint(256,size=G.number_of_nodes())
    ng = np.random.randint(256,size=G.number_of_nodes())
    nb = np.random.randint(256,size=G.number_of_nodes())
    power = 0.3 # Weighted Wigner-Seitz construction if power not zero
    for x in range(psy):
        for y in range(psx):
            dmin = np.sqrt((psx-1)**2+(psy-1)**2)
            jj = -1
            for ii in range(G.number_of_nodes()):
                if power > 0:
                    dis = np.sqrt((nx[ii]-x)**2+(ny[ii]-y)**2)/np.power(narea[ii],power)
                else:
                    dis = np.sqrt((nx[ii]-x)**2+(ny[ii]-y)**2)
                if dis < dmin:
                    dmin = dis
                    jj = ii
            S[jj] = S[jj] + 1
            mask[x,y] = nodelist[jj]   
    S = S/scale**2
    for ii in range(len(nodelist)):
        G.nodes[nodelist[ii]]['area_vor'] = S[ii]
    # Delete from graph the nodes with cells that neighbor the edge
    on_edge = []
    for x in [0,psy-1]:
        for y in range(psx):
            if mask[x,y] not in on_edge:
                on_edge.append(mask[x,y])
    for y in [0,psx-1]:
        for x in range(psy):
            if mask[x,y] not in on_edge:
                on_edge.append(mask[x,y])
    
    G_inner = G.copy()
    for ii in on_edge:
        G_inner.remove_node(ii)
    # Finds neighbors 
    nodelist_inner = G_inner.nodes()
    for ii in nodelist_inner:
        nbrs = []
        for x, y in np.column_stack(np.where(mask==ii)):
            if (x==0) or (x==mask.shape[0]-1) or (y==0) or (y==mask.shape[1]-1):
                continue
            c1 = mask[x-1,y] != ii
            if c1 and mask[x-1,y] not in nbrs:
                nbrs.append(mask[x-1,y])
            c2 = mask[x+1,y] != ii
            if c2 and mask[x+1,y] not in nbrs:
                nbrs.append(mask[x+1,y])
            c3 = mask[x,y-1] != ii
            if c3 and mask[x,y-1] not in nbrs:
                nbrs.append(mask[x,y-1])
            c4 = mask[x,y+1] != ii
            if c4 and mask[x,y+1] not in nbrs:
                nbrs.append(mask[x,y+1])
        # Counts number of sides of each cell
        G_inner.nodes[ii]['degree'] = len(nbrs)
        for jj in nbrs:
            if jj in nodelist_inner:
                px_dis = np.sqrt(sum((G_inner.nodes[jj]['pixel_pos']-G_inner.nodes[ii]['pixel_pos'])**2))
                G_inner.add_edge(ii,jj, dis=px_dis/scale)
    # Colors the voronoi diagram		
    for x in range(2,psy-1):
        for y in range(2,psx-1):
            if mask[x-1,y] != mask[x,y]:
                continue
            if mask[x+1,y] != mask[x,y]:
                continue
            if mask[x,y-1] != mask[x,y]:
                continue
            if mask[x,y+1] != mask[x,y]:
                continue
            if mask[x,y] in G_inner.nodes(data=True):
                img_vor_deg[x,y,:] = color_by_degree(G_inner.nodes[mask[x,y]]['degree'])
            else:
                img_vor_deg[x,y,:] = [127, 127, 127]
            ii = int(np.where(nodelist==mask[x,y])[0])
            img_vor[x,y,:] = [nr[ii], ng[ii], nb[ii]]   
    return img_vor, img_vor_deg, G, G_inner


# Computes network metrics values
def statistics(G,G_inner,G_MSF):
    Ne = G_MSF.number_of_edges()
    N = G_MSF.number_of_nodes()
    m = 0 # average edge length
    sig = 0 # standard deviation edge length
    deg_list = []
    deg_list = [G_inner.nodes[n]['degree'] for n, tmp in G_inner.nodes(data=True)]
    deg = sum(deg_list) / N # average degree
    defect_ratio = [1 for n in deg_list if n!=6]
    defect_ratio = sum(defect_ratio)/N
    for ei,ef in G_MSF.edges():
        m = m + G_MSF[ei][ef]['dis']
    m = m / Ne
    for ei,ef in G_MSF.edges():
        sig = sig + (G_MSF[ei][ef]['dis'] - m)**2
    sig = np.sqrt(sig/(Ne-1))
    S = sum([G_inner.node[n]['area_vor'] for n, tmp in G_inner.nodes(data=True)])/N
    m = m / np.sqrt(S) * (N-1)/N
    sig = sig / np.sqrt(S) * (N-1)/N
    return deg_list, deg, m, sig, S, defect_ratio

# Overlays nodes and MSF over image 
def overlay(img,G,G_MSF,radius):
    for n, tmp in G.nodes(data=True):
        r = G.nodes[n]['pixel_pos'][0]
        c = G.nodes[n]['pixel_pos'][1]
        [rr,cc]= draw.circle(r,c,radius)
        mask = np.array(rr >= 0)
        mask = mask & np.array(rr<len(img[:,0,0]))
        mask = mask & np.array(cc >= 0)
        mask = mask & np.array(cc<len(img[0,:,0]))
        rr = rr[mask]
        cc = cc[mask]
        img[rr,cc,:] = (1,0,0) 
    for n, nbrs in G.adj.items():
        for nbr, dis in nbrs.items():
            p1 = G.nodes[n]['pixel_pos']
            p2 = G.nodes[nbr]['pixel_pos']
            [rr,cc] = draw.line(int(p1[0]),int(p1[1]),int(p2[0]),int(p2[1]))
            img[rr,cc,:] = (1,0,0) 
    for n, nbrs in G_MSF.adj.items():
        for nbr, dis in nbrs.items():
            p1 = G_MSF.nodes[n]['pixel_pos']
            p2 = G_MSF.nodes[nbr]['pixel_pos']
            [rr,cc] = draw.line(int(p1[0]),int(p1[1]),int(p2[0]),int(p2[1]))
            img[rr,cc,:] = (0,1,0) 
    return img

# Color coding the cells by their coordination number
def color_by_degree(deg):
    if deg == 6:
        return [0, 255, 0]
    elif deg > 6:
        diff = int(255*min(np.sqrt(deg-6)/2,1))
        return [diff,255-diff,0]
    elif deg < 6:
        diff = int(255*min(np.sqrt(6-deg)/2,1))
        return [0,255-diff,diff]

def save_data(G_MSF,S,deg_avg, deg_list, m, sig, filename, defect_ratio):
    with open('data.txt', 'a') as myfile:
        myfile.write('%s  %s  %s  %s  %s  %s  %s\n'\
        % (filename, m, sig, deg_avg,S,S*G_MSF.number_of_nodes(),defect_ratio))
    hist = [G_MSF[ei][ef]['dis'] for ei, ef in G_MSF.edges()]
    np.savetxt('hist_dis_'+filename+'.txt',hist)
    area_list = [G_MSF.node[n]['area_vor'] for n, tmp in G_MSF.nodes(data=True)]
    np.savetxt('hist_S_'+filename+'.txt',area_list)
    deg_list
    np.savetxt('hist_degree_'+filename+'.txt',deg_list)
    
def color_deg6(img, mask, G_inner):
    list_color = []
    for ii in range(1,np.max(mask)):
        if ii in G_inner.nodes():
            list_color.append([x/256 for x in color_by_degree(G_inner.node[ii]['degree'])])
        else:
            list_color.append([.5,.5,.5])
    img_degree = color.label2rgb(mask, image=img, colors=list_color, bg_label=0)
    return img_degree

# MAIN
area_threshold = np.pi*0.3**2  # nm^2

root_fol = os.getcwd()
folders = [fol for fol in os.listdir() if os.path.isdir(fol)]
for fol in folders:
    os.chdir(os.path.abspath(fol))
    old_files = [f for f in os.listdir() if os.path.isfile(f) and ('deg_' in f or 'vor_' in f or 'gr_' in f or '.txt' in f)]
    for f in old_files:
        os.remove(f)

    files = [f for f in os.listdir() if os.path.isfile(f) and ('.jpg' in f or '.png' in f)]
    images = [img_file(f) for f in files]

    with open('data.txt', 'w') as myfile:
        myfile.write('%s  %s  %s  %s  %s  %s  %s\n'\
        % ('filename', 'Mean dis', 'Std dev dis', 'Average degree', 'S', 'Total area', 'Defect ratio'))

    for i in images:
        img = i.load()
        print(i.name)
        scale = i.scale   # px/nm
        px_a_th = area_threshold*scale**2
        radius = int(scale/5)
        img_gr = gray_process(img,dsize=.8*scale,cutoff=0.40) # dsize and cutoff should be adapted to image
        [img_label_overlay, label_mask, Nlabel] = labels(img_gr)
        G = find_nodes(label_mask,Nlabel,px_a_th,scale)
        
        [img_vor, img_vor_deg, G, G_inner] = voronoi(i,G)
        io.imsave('vor_deg'+i.name,img_vor_deg)
        img_vor_deg = overlay(img_vor_deg,G_inner,G_inner,radius)
        io.imsave('vor_deg_net'+i.name,img_vor_deg)
        
        img_vor = overlay(img_vor,G_inner,G_inner,radius)
        io.imsave('vor_'+i.name,img_vor)
        
        img_degree = color_deg6(img_gr, label_mask, G_inner)
        io.imsave('deg_'+i.name,(img_degree*255).astype(np.uint8))
        
        # Calculates minimum spanning tree/forest
        G_MSF = nx.minimum_spanning_tree(G_inner,weight='dis')  
        img_graph = overlay(img_label_overlay,G_inner,G_MSF,radius)
        io.imsave('gr_'+i.name,(img_graph*255).astype(np.uint8))
        
        [deg_list, deg_avg, m, sig, S, defect_ratio] = statistics(G,G_inner,G_MSF)
        save_data(G_MSF, S,deg_avg, deg_list, m, sig, i.name, defect_ratio)

    os.chdir(root_fol)