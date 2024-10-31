import cv2
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model

# Using Hough
def find_roi(gray_img, colored_img, circles):
    rois = []
    coordinates = []
    circles = np.round(circles[0, :]).astype("int")
    # Applying CNN
    new_model = load_model('star_verification_cnn_2.h5')
    for (x, y, r) in circles:
        roi = gray_img[y - (r):y + (r), x - (r):x + (r)]
        roi_redo = cv2.resize(roi, (64, 64))
        # After defining roi, get ready to apply cnn
        roi_redo = cv2.cvtColor(roi_redo, cv2.COLOR_GRAY2BGR)  # Convert to grayscale if necessary
        roi_redo = roi_redo.astype('float32') / 255.0  # Normalize
        roi_redo = np.array([roi_redo])  # Add channel dimension
        print(roi_redo.shape)
        prediction = new_model.predict(roi_redo)
        print(prediction)
        if (prediction[0] >= 0.45):
            label = "Star"
        else:
            label = "Non-Star"
    
        if (label == 'Star'):
            cv2.circle(colored_img, (x, y), r, (0, 255, 255), 2)
       
        cv2.circle(colored_img, (x, y), r, (0, 255, 0) if label == 'Star' else (0, 0, 255), 2)
        cv2.putText(colored_img, label, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if label == 'Star' else (0, 0, 255), 2)

        cv2.imshow("ROI found", roi)
        cv2.imshow("Colored_ROI", colored_img)
        cv2.waitKey(0)
    cv2.waitKey(0)
def detecting_stars(num):
    image = cv2.imread(f'DeepSpaceYoloDataset\DeepSpaceYoloDataset\images\\{num}.jpg', cv2.IMREAD_GRAYSCALE)
    img_blur = cv2.GaussianBlur(image, (9, 9), 2)
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=0.5, minRadius=2, maxRadius=10)
    original_img = cv2.imread(f'DeepSpaceYoloDataset\DeepSpaceYoloDataset\images\\{num}.jpg')
    find_roi(image, original_img, circles)  # ROIs will be used for the CNN

detecting_stars(4320)
# 990, 3283, 4560, 995
