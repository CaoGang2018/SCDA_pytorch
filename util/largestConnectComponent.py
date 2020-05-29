from skimage.measure import label
import numpy as np

def largestConnectComponent(bw_img, neighbor=2):
    '''
    compute largest Connect component of a binary image
    
    Parameters:
    ---

    bw_img: ndarray
        binary image
	
	Returns:
	---

	lcc: ndarray
		largest connect component.

    Example:
    ---
        >>> lcc = largestConnectComponent(bw_img)

    '''

    labeled_img, num = label(bw_img, connectivity=neighbor, background=0, return_num=True)    
    # plt.figure(), plt.imshow(labeled_img, 'gray')

    max_label = 0
    max_num = 0
    for i in range(1, num+1): # 这里从1开始，防止将背景设置为最大连通域
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)

    return lcc