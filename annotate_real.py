import cv2
import numpy as np
import os

class KeypointsAnnotator:
    def __init__(self, num_keypoints=1):
        self.num_keypoints  = num_keypoints

    def load_image(self, img):
        self.img = img
        self.vis = img.copy()

    def mouse_callback(self, event, x, y, flags, param):
        cv2.imshow("pixel_selector", self.vis)
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.clicks.append([x, y])
            cv2.circle(self.vis, (x, y), 3, (255, 0, 0), -1)

    def run(self, img):
        self.load_image(img)
        self.clicks = []
        self.label = 0
        cv2.namedWindow('pixel_selector')
        cv2.setMouseCallback('pixel_selector', self.mouse_callback)
        while True:
            k = cv2.waitKey(20) & 0xFF
            if k == 27 or len(self.clicks) == self.num_keypoints or (cv2.waitKey(33) == ord('s')):
                break
            if cv2.waitKey(33) == ord('r'):
                self.clicks = []
                self.load_image(img)
                print('Erased annotations for current image')
        print(self.clicks)
        return self.clicks

if __name__ == '__main__':
    pixel_selector = KeypointsAnnotator(num_keypoints=1)

    image_dir = 'split_images/train/images' # Should have images like 00000.jpg, 00001.jpg, ...
    output_dir = 'real_data_train' # Will have real_data/images and real_data/keypoints
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    keypoints_output_dir = os.path.join(output_dir, 'keypoints')
    images_output_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(keypoints_output_dir):
        os.mkdir(keypoints_output_dir)
    if not os.path.exists(images_output_dir):
        os.mkdir(images_output_dir)

    i = 0
    for f in sorted(os.listdir(image_dir)):
        print("Img %d"%i)
        image_path = os.path.join(image_dir, f)
        print(image_path)
        img = cv2.imread(image_path)
        image_outpath = os.path.join(images_output_dir, '%05d.jpg'%i)
        keypoints_outpath = os.path.join(keypoints_output_dir, '%05d.npy'%i)
        annots = pixel_selector.run(img)
        print("---")
        if len(annots)>0:
            annots = np.array(annots)
            cv2.imwrite(image_outpath, img)
            np.save(keypoints_outpath, annots)
            i  += 1
