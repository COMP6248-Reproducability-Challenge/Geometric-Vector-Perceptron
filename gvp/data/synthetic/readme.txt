The synthetic dataset contains 20k "structures" as
described in the manuscript. The voxel-based representation
for the CNN is in cnn.npy, while the graph-based representation
for the standard GNN and GVP-GNN is in synthetic.npy.
The unnormalized ground-truth labels are in answers.npz.

cnn.npy: single tensor with shape (20000, 30, 30, 30, 3)
         corresponding to occupancy grids of size 30 x 30 x 30.
         The last dimension corresponds to the channel,
         with the first channel being the "special" points,
         the second being the "non-special" points,
         and the third being the "sidechains."

synthetic.npy: single tensor with shape (20000, 2, 100, 3).
               In the second dimension, the 1st entry
               stores the position vectors of the 100 points
               and the 2nd entry stores the "sidechain"
               unit vectors for the 100 points.

answers.npz: tensor "off_center" with shape (20000, )
             tensor "perimeter" with shape (20000, )
 

