import network_ss
import matplotlib.pyplot as plt


sample_weight_seg, sample_weight_boxmask = network_ss.get_class_weights()

sample_weight_seg[:,:,0] = 0.1

net = network_ss.initialize_network(sample_weight_seg, sample_weight_boxmask)
net.load_weights("./output/ss_only/model_ep_7.ckpt")

# 'X51008123604.npy'
predict_this, _ = network_ss.get_train_test_sets(["X00016469612.npy"]) #'X51006008057.npy', 'X51005441402.npy', 'X51005605284(2).npy', 'X51006913031.npy', 'X51005442327.npy',
in_cg, label_segmentation = network_ss.extract_batch(predict_this, len(predict_this), network_ss.pad_left_range, network_ss.pad_top_range, network_ss.pad_right_range, network_ss.pad_bot_range)
segmentation = net.predict(x=in_cg)
print("stop")

# classes: 0 = background, 1 = total, 2 = address , 3 = company ,4 = date
for i in range(0,len(predict_this)):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(in_cg[i,:,:,:].argmax(axis=2))      #in_cg[0,:,:,:].argmax(axis=2))    np.reshape(in_cg[0], [256,128])
    ax2.imshow(label_segmentation[i,:,:,:].argmax(axis=2))
    ax3.imshow(segmentation[i,:,:,:].argmax(axis=2))
    plt.show()




