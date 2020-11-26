import network
import numpy as np
import matplotlib.pyplot as plt

sample_weight_seg, sample_weight_boxmask = network.get_class_weights()
net = network.initialize_network(sample_weight_seg, sample_weight_boxmask)
net.load_weights("./output/model_ep_0.ckpt")

predict_this, _ = network.get_train_test_sets(['X51008123604.npy', 'X51005441401.npy', 'X51008099087.npy', 'X51005749905.npy', 'X51008164510.npy', 'X51005442378.npy', 'X51005361883.npy', 'X51006620190.npy'])

in_cg, label_segmentation, _, _ = network.extract_batch(predict_this, len(predict_this), network.pad_left_range, network.pad_top_range, network.pad_right_range, network.pad_bot_range)
segmentation, anchor_mask, anchor_coord = net.predict(in_cg)

print("stop")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(in_cg[0,:,:,:].argmax(axis=2))
ax2.imshow(label_segmentation[0,:,:,:].argmax(axis=2))
ax3.imshow(segmentation[0,:,:,:].argmax(axis=2))
plt.show()

# flat_img = segmentation[0,:,:,:].argmax(axis=2)
# plt.imshow(flat_img)
# plt.show()
