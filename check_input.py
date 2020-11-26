import numpy as np
import matplotlib.pyplot as plt
from  PIL import  Image
import os

dir_np_bbox_anchor_mask = "./data/np_bbox_anchor_mask/"
dir_np_bbox_anchor_coord = "./data/np_bbox_anchor_coord/"
dir_np_cg_1h = "./data/np_chargrids_1h/"
list_filenames = [f for f in os.listdir("./data/np_gt/") if os.path.isfile(os.path.join("./data/np_gt/", f))]

bbox_anchor_mask = np.load(os.path.join(dir_np_bbox_anchor_mask, list_filenames[0]))
cg = np.load(os.path.join(dir_np_cg_1h, list_filenames[0]))



fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
ax1.imshow(bbox_anchor_mask[:,:,0])
ax2.imshow(bbox_anchor_mask[:,:,1])
ax3.imshow(bbox_anchor_mask[:,:,2])
ax4.imshow(bbox_anchor_mask[:,:,3])
ax4.imshow(cg.argmax(axis=1))



i = 123
fname = list_filenames[i].replace("npy", "jpg")
data_in = Image.open(f"./data/img_inputs/{fname}").convert(mode='L')
data_in = np.array(data_in) / 255
#data = np.sum(data, axis=1)
print(data_in.shape)

# plt.imshow(data_in, cmap=plt.get_cmap('gray'))
# plt.show()
# print(123)


data_label = np.load(os.path.join("./data/np_gt/", list_filenames[i]))

data_label.shape

# plt.imshow(data)
#plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(data_in)  # in_cg[0,:,:,:].argmax(axis=2))    np.reshape(in_cg[0], [256,128])
ax2.imshow(data_label)
plt.show()